"""
Elevate Emotion Recognition CNN - Training Script
===================================================
Architecture  : MobileNetV2 backbone + custom classification head
Dataset       : dataset/happy | bored | focused | confused | neutral | angry | surprise
Output        : backend/ai_models/emotion_model.h5
                backend/ai_models/emotion_model_info.json
                frontend/js/emotion_tfjs/  (TensorFlow.js web model)

Run from project root:
    pip install tensorflow tensorflowjs scikit-learn matplotlib seaborn pillow
    python train_emotion_model.py

Training takes ~5-15 minutes on CPU, ~2-5 minutes with GPU.
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress verbose TF logs
warnings.filterwarnings("ignore")

# ── TensorFlow imports ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Scikit-learn for evaluation ──────────────────────────────────────────────
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

# ────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these paths if your layout differs)
# ────────────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "backend", "ai_models")
TFJS_OUT_DIR  = os.path.join(os.path.dirname(__file__), "frontend", "js", "emotion_tfjs")

# Training hyper-parameters
IMG_SIZE        = (96, 96)      # MobileNetV2 min is 32×32; 96 is fast & accurate
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 10            # train only the head first
EPOCHS_FINETUNE = 15            # then unfreeze last 30 layers
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 1e-4
VALIDATION_SPLIT = 0.20
RANDOM_SEED      = 42

# ── Canonical labels and dataset folder labels ──────────────────────────────
CLASS_NAMES = ["happy", "bored", "focused", "confused", "neutral", "angry", "surprised"]
DATASET_CLASS_NAMES = ["happy", "bored", "focused", "confused", "neutral", "angry", "surprise"]

# ────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TFJS_OUT_DIR, exist_ok=True)


def verify_dataset():
    """Check that all class folders exist and report image counts."""
    print("\n[1/7] Verifying dataset …")
    counts = {}
    for dataset_cls, canonical_cls in zip(DATASET_CLASS_NAMES, CLASS_NAMES):
        path = os.path.join(DATASET_DIR, dataset_cls)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Missing folder: {path}\n"
                f"Expected dataset layout:\n"
                f"  dataset/happy/\n"
                f"  dataset/bored/\n"
                f"  dataset/focused/\n"
                f"  dataset/confused/\n"
                f"  dataset/neutral/\n"
                f"  dataset/angry/\n"
                f"  dataset/surprise/"
            )
        imgs = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        counts[canonical_cls] = len(imgs)
        print(f"   {canonical_cls:12s}: {len(imgs):,} images")

    total = sum(counts.values())
    print(f"   {'TOTAL':12s}: {total:,} images")
    return counts


def build_data_generators(counts):
    """
    Create augmented train and validation ImageDataGenerators.

    Augmentation rationale
    ----------------------
    - horizontal_flip  : mirrors face → doubles effective data
    - rotation_range 15: head tilts common in natural video
    - brightness_range : handles lighting variance
    - zoom_range 0.10  : partial face crops / camera distances
    - shear_range      : slight perspective shifts
    We do NOT use vertical flips or large rotations because upside-down
    faces are not a natural STEM learning scenario.
    """
    print("\n[2/7] Building data generators …")

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=[0.75, 1.25],
        zoom_range=0.10,
        shear_range=0.08,
        fill_mode="nearest",
    )

    # Validation: only rescale — no augmentation
    val_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_aug.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=DATASET_CLASS_NAMES,
        class_mode="categorical",
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
    )

    val_gen = val_aug.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=DATASET_CLASS_NAMES,
        class_mode="categorical",
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,
    )

    print(f"   Train samples    : {train_gen.samples:,}")
    print(f"   Validation samples: {val_gen.samples:,}")
    print(f"   Class indices    : {train_gen.class_indices}")
    return train_gen, val_gen


def compute_weights(train_gen):
    """
    Compute inverse-frequency class weights to handle imbalance.
    confused (1 285) vs neutral (8 136) → neutral weight ≈ 0.4, confused ≈ 2.5
    """
    print("\n[3/7] Computing class weights for imbalance …")
    labels = train_gen.classes
    unique = np.unique(labels)
    weights_arr = compute_class_weight("balanced", classes=unique, y=labels)
    weight_dict = {int(k): float(v) for k, v in zip(unique, weights_arr)}
    for idx, cls in enumerate(CLASS_NAMES):
        print(f"   {cls:12s}: weight = {weight_dict[idx]:.4f}")
    return weight_dict


def build_model():
    """
    Architecture: MobileNetV2 backbone + custom head

    Why MobileNetV2?
    ----------------
    - Designed for edge/mobile inference (depthwise-separable convolutions)
    - ~3.4M parameters total vs VGG16's 138M → trains faster on laptop CPU
    - Exports cleanly to TensorFlow.js for browser inference
    - Pre-trained on ImageNet — transfers facial texture features well

    Head design:
    - GlobalAveragePooling: spatial → vector without dense overhead
    - Dropout(0.4): regularisation — reduces overfit on small classes
    - Dense(128, relu): task-specific feature compression
    - Dropout(0.3): additional regularisation
    - Dense(N, softmax): project taxonomy probability output
    """
    print("\n[4/7] Building MobileNetV2 model …")

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False      # freeze backbone for phase-1 training

    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.40, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="predictions"
    )(x)

    model = models.Model(inputs, outputs, name="elevate_emotion_cnn")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total   = model.count_params()
    trainable = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    print(f"   Total params     : {total:,}")
    print(f"   Trainable params : {trainable:,}  (backbone frozen)")
    return model, base_model


def get_callbacks(phase: str):
    """Standard callbacks used in both training phases."""
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_weights_{phase}.weights.h5")
    return [
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_phase1(model, train_gen, val_gen, class_weight):
    """Phase 1: train only the classification head (backbone frozen)."""
    print(f"\n[5a/7] Phase 1 — Training head for up to {EPOCHS_FROZEN} epochs …")
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )
    return history


def train_phase2(model, base_model, train_gen, val_gen, class_weight):
    """
    Phase 2: unfreeze the last 30 layers of MobileNetV2 and fine-tune.
    We use a 10× lower learning rate to avoid destroying pre-trained weights.
    """
    print(f"\n[5b/7] Phase 2 — Fine-tuning last 30 backbone layers …")

    # Unfreeze last 30 layers
    for layer in base_model.layers[-30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    trainable_now = sum(
        tf.size(v).numpy() for v in model.trainable_variables
    )
    print(f"   Trainable params now: {trainable_now:,}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )
    return history


def evaluate_and_save(model, val_gen):
    """
    Full evaluation on the validation set.
    Prints per-class precision, recall, F1, and overall AUC.
    Saves confusion matrix PNG for your project report.
    """
    print("\n[6/7] Evaluating model on validation set …")

    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true       = val_gen.classes[: len(y_pred)]   # align lengths

    print("\n── Classification Report ──────────────────────────────────")
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    print(report)

    # Per-class AUC (one-vs-rest)
    try:
        n = len(y_true)
        y_true_oh = np.zeros((n, len(CLASS_NAMES)))
        y_true_oh[np.arange(n), y_true] = 1
        auc = roc_auc_score(y_true_oh, y_pred_proba[:n], multi_class="ovr", average="macro")
        print(f"Macro AUC (OvR): {auc:.4f}")
    except Exception:
        auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Elevate Emotion CNN — Validation Confusion Matrix", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    return report, auc


def plot_training_history(h1, h2):
    """Combined accuracy + loss curves for both phases."""
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    val_loss = h1.history["val_loss"]  + h2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(acc,     label="Train accuracy",  linewidth=2)
    ax1.plot(val_acc, label="Val accuracy",    linewidth=2, linestyle="--")
    ax1.axvline(len(h1.history["accuracy"]) - 1, color="gray",
                linestyle=":", label="Fine-tune start")
    ax1.set_title("Model Accuracy (Phase 1 + Fine-tune)", fontsize=12)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(loss,     label="Train loss",  linewidth=2)
    ax2.plot(val_loss, label="Val loss",    linewidth=2, linestyle="--")
    ax2.axvline(len(h1.history["loss"]) - 1, color="gray",
                linestyle=":", label="Fine-tune start")
    ax2.set_title("Model Loss (Phase 1 + Fine-tune)", fontsize=12)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {curve_path}")


def save_model_and_metadata(model, val_acc, report, auc, counts):
    """
    Save:
      1. emotion_model.h5          — full Keras model (server inference)
      2. emotion_model_info.json   — metadata consumed by Flask API
      3. emotion_tfjs/             — TF.js web model (browser inference)
    """
    print("\n[7/7] Saving model artefacts …")

    # 1. Keras H5
    h5_path = os.path.join(OUTPUT_DIR, "emotion_model.h5")
    model.save(h5_path)
    print(f"   Keras model  → {h5_path}")

    # 2. Metadata JSON
    info = {
        "model_name"     : "elevate_emotion_cnn",
        "architecture"   : "MobileNetV2 + custom head",
        "input_shape"    : [96, 96, 3],
        "class_names"    : CLASS_NAMES,
        "num_classes"    : len(CLASS_NAMES),
        "val_accuracy"   : float(round(val_acc, 4)),
        "macro_auc"      : float(round(auc, 4)) if auc else None,
        "dataset_counts" : counts,
        "total_images"   : sum(counts.values()),
        "img_size"       : list(IMG_SIZE),
        "normalisation"  : "divide_by_255",
        "augmentation"   : [
            "horizontal_flip",
            "rotation_range_15",
            "brightness_0.75-1.25",
            "zoom_0.10",
            "shear_0.08",
        ],
        "training_notes" : (
            "Phase-1: head-only training (backbone frozen). "
            "Phase-2: fine-tune last 30 MobileNetV2 layers at LR=1e-4. "
            "Class weights applied to handle imbalance (confused << neutral)."
        ),
    }
    info_path = os.path.join(OUTPUT_DIR, "emotion_model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"   Metadata     → {info_path}")

    # 3. TensorFlow.js export
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, TFJS_OUT_DIR)
        print(f"   TF.js model  → {TFJS_OUT_DIR}/")
        print("   ✅ TF.js export complete — copy this folder to frontend/js/")
    except ImportError:
        print(
            "   ⚠️  tensorflowjs not installed. Run:\n"
            "      pip install tensorflowjs\n"
            "   then re-run this script, OR convert manually:\n"
            f"      tensorflowjs_converter --input_format=keras {h5_path} {TFJS_OUT_DIR}"
        )

    return h5_path, info_path


# ────────────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ELEVATE — Emotion Recognition CNN Training Pipeline")
    print("=" * 65)
    print(f"  TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU available      : {bool(gpus)} {'(' + gpus[0].name + ')' if gpus else '(using CPU)'}")
    print(f"  Dataset directory  : {DATASET_DIR}")
    print(f"  Output directory   : {OUTPUT_DIR}")
    print("=" * 65)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_dirs()
    counts      = verify_dataset()
    train_gen, val_gen = build_data_generators(counts)
    class_weight = compute_weights(train_gen)
    model, base  = build_model()

    # ── Training ─────────────────────────────────────────────────────────────
    h1 = train_phase1(model, train_gen, val_gen, class_weight)
    h2 = train_phase2(model, base, train_gen, val_gen, class_weight)

    plot_training_history(h1, h2)

    # ── Evaluation ───────────────────────────────────────────────────────────
    _, val_acc = model.evaluate(val_gen, verbose=0)
    report, auc = evaluate_and_save(model, val_gen)

    # ── Save artefacts ───────────────────────────────────────────────────────
    save_model_and_metadata(model, val_acc, report, auc, counts)

    print("\n" + "=" * 65)
    print(f"  ✅  Training complete!")
    print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
    if auc:
        print(f"  Macro AUC (OvR)    : {auc:.4f}")
    print(f"\n  Next step → run:  python backend/routes/emotions.py  (test server-side inference)")
    print(f"  Or open the demo :  frontend/js/emotion_tfjs/model.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
