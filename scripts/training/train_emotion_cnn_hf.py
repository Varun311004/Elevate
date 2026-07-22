"""
High-accuracy emotion training pipeline for Hugging Face / server inference.

Outputs:
  - backend/ai_models/emotion_model.h5
  - backend/ai_models/emotion_model_info.json

This model is consumed by backend/routes/ai_emotion.py for camera inference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import imagehash


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
AI_MODELS_DIR = ROOT / "backend" / "ai_models"
FRONTEND_DIR = ROOT / "frontend"

CLASS_NAMES = ["happy", "bored", "focused", "confused", "neutral", "angry", "surprised"]
CLASS_ALIASES = {
    "surprise": "surprised",
    "surprised": "surprised",
}
CLASS_FOLDER_ALIASES = {
    "happy": ["happy"],
    "bored": ["bored"],
    "focused": ["focused"],
    "confused": ["confused"],
    "neutral": ["neutral"],
    "angry": ["angry"],
    "surprised": ["surprised", "surprise"],
}

IMG_SIZE = (96, 96)
BATCH_SIZE = int(os.environ.get("ELEVATE_EMOTION_BATCH_SIZE", "32"))
EPOCHS_HEAD = int(os.environ.get("ELEVATE_EMOTION_EPOCHS_HEAD", "18"))
EPOCHS_FINETUNE = int(os.environ.get("ELEVATE_EMOTION_EPOCHS_FINETUNE", "14"))
EARLY_STOP_PATIENCE = int(os.environ.get("ELEVATE_EMOTION_EARLY_STOP_PATIENCE", "5"))
FINE_TUNE_AT = 100
SEED = int(os.environ.get("ELEVATE_EMOTION_SEED", "42"))
MIN_PER_CLASS = int(os.environ.get("ELEVATE_EMOTION_MIN_PER_CLASS", "50"))


@dataclass
class DatasetSplit:
    train_files: List[str]
    val_files: List[str]
    test_files: List[str]
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray


def _canonical_name(name: str) -> str:
    n = str(name or "").strip().lower()
    return CLASS_ALIASES.get(n, n)


def _collect_paths() -> Tuple[List[str], np.ndarray]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing dataset directory: {DATASET_DIR}")

    folder_map = {
        p.name.strip().lower(): p
        for p in DATASET_DIR.iterdir()
        if p.is_dir()
    }

    all_files: List[str] = []
    all_labels: List[int] = []

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = None
        for alias in CLASS_FOLDER_ALIASES.get(class_name, [class_name]):
            class_dir = folder_map.get(alias.lower())
            if class_dir is not None:
                break
        if class_dir is None:
            raise FileNotFoundError(
                f"Missing class folder for '{class_name}'. "
                f"Expected one of: {CLASS_FOLDER_ALIASES.get(class_name, [class_name])}"
            )
        files = sorted(
            str(p)
            for p in class_dir.rglob("*") # THE FIX: 'rglob' searches all sub-shards recursively
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(files) < MIN_PER_CLASS:
            raise RuntimeError(
                f"Class '{class_name}' has too few images ({len(files)}). "
                f"Need at least {MIN_PER_CLASS}."
            )
        all_files.extend(files)
        all_labels.extend([idx] * len(files))

    return all_files, np.asarray(all_labels, dtype=np.int32)


def _group_key_for_file(path: str, phash_cache: dict) -> str:
    """
    Returns a group id such that near-duplicate frames (e.g. consecutive
    frames pulled from the same short webcam recording session) always
    get the same id, and therefore always land in the same split.
 
    Perceptual hash with a small Hamming-distance tolerance catches
    near-duplicates that a random/stratified split would otherwise
    scatter across train/val/test — which is what was silently inflating
    the 'bored'/'focused' scores to ~1.0 F1.
    """
    img_hash = imagehash.phash(Image.open(path).convert("L"))
    phash_cache[path] = img_hash
 
    # Bucket by a coarse version of the hash (first 32 bits) so that
    # near-identical frames collapse to the same bucket even with minor
    # compression/lighting noise between frames.
    return str(img_hash)[:8]


def _split_dataset(files: List[str], labels: np.ndarray) -> DatasetSplit:
    phash_cache: dict = {}
    groups = np.array([_group_key_for_file(f, phash_cache) for f in files])
 
    # First carve off test, then val, splitting by GROUP not by file.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(gss_test.split(files, labels, groups=groups))
 
    trainval_files = [files[i] for i in trainval_idx]
    trainval_labels = labels[trainval_idx]
    trainval_groups = groups[trainval_idx]
 
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(
        gss_val.split(trainval_files, trainval_labels, groups=trainval_groups)
    )
 
    return DatasetSplit(
        train_files=[trainval_files[i] for i in train_idx],
        val_files=[trainval_files[i] for i in val_idx],
        test_files=[files[i] for i in test_idx],
        train_labels=trainval_labels[train_idx].astype(np.int32),
        val_labels=trainval_labels[val_idx].astype(np.int32),
        test_labels=labels[test_idx].astype(np.int32),
    )


def _decode_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    # Pretrained ImageNet backbones expect 3 channels — replicate grayscale.
    img = tf.image.grayscale_to_rgb(img)
    return img, label


def _build_tf_dataset(files: List[str], labels: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(files), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.1),   # webcams vary in exposure
        ])
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v3.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _build_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Returns (full_model, backbone) so main() can freeze/unfreeze the backbone."""
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        minimalistic=True,   # smaller export, fewer squeeze-excite blocks
    )
    backbone.trainable = False  # frozen for the head-training phase
 
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
 
    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def _to_class_weight(labels: np.ndarray) -> Dict[int, float]:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _evaluate(model: tf.keras.Model, ds: tf.data.Dataset, y_true: np.ndarray) -> dict:
    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        k: {
            "precision": round(float(v.get("precision", 0.0)), 4),
            "recall": round(float(v.get("recall", 0.0)), 4),
            "f1": round(float(v.get("f1-score", 0.0)), 4),
            "support": int(v.get("support", 0)),
        }
        for k, v in report.items()
        if k in CLASS_NAMES
    }
    return {
        "accuracy": round(float(report.get("accuracy", 0.0)), 4),
        "macro_f1": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "macro_precision": round(float(report.get("macro avg", {}).get("precision", 0.0)), 4),
        "macro_recall": round(float(report.get("macro avg", {}).get("recall", 0.0)), 4),
        "per_class": per_class,
    }

def _write_tfjs_layers_model(model: "tf.keras.Model", tfjs_out_dir: Path) -> None:
    """Write a Keras model in TF.js LayersModel format — no external packages needed.

    Replaces the tensorflowjs Python CLI entirely.  tensorflowjs 4.15.0 is
    fundamentally incompatible with TF 2.16.x: its converter unconditionally
    imports tensorflow_decision_forests, tensorflow_hub, and jax — all of
    which either cannot be installed alongside TF 2.16 or crash on import.

    This function produces the identical output using only:
      - json  (stdlib)
      - numpy (already imported)
      - tensorflow (already imported)

    Output
    ------
    ``<tfjs_out_dir>/model.json``
        Model topology (Keras config) + weights manifest.
    ``<tfjs_out_dir>/group1-shard1of1.bin``
        All weights concatenated as little-endian float32.

    The frontend’s ``_loadTfjsModel()`` already patches
    ``batch_shape → batchInputShape`` before calling ``tf.io.fromMemory()``,
    so we don’t need to handle that here.
    """
    tfjs_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect weights
    weight_specs: list[dict] = []
    weight_bytes_parts: list[bytes] = []
    for w in model.weights:
        arr = w.numpy().astype(np.float32)
        name = w.name[:-2] if w.name.endswith(":0") else w.name
        weight_specs.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
        weight_bytes_parts.append(arr.tobytes())

    # 2. Write binary weight shard
    shard_filename = "group1-shard1of1.bin"
    with open(tfjs_out_dir / shard_filename, "wb") as fbin:
        for chunk in weight_bytes_parts:
            fbin.write(chunk)

    # 3. Build model topology
    #    Wrap get_config() exactly as tf.loadLayersModel expects:
    #    modelTopology → { model_config: { class_name, config }, keras_version, backend }
    keras_version = getattr(tf.keras, "__version__", "unknown")
    model_topology = {
        "model_config": {
            "class_name": model.__class__.__name__,
            "config": model.get_config(),
        },
        "keras_version": keras_version,
        "backend": "tensorflow",
    }

    # 4. Write model.json
    model_artifact = {
        "format": "layers-model",
        "generatedBy": f"TensorFlow {tf.__version__} / Keras {keras_version}",
        "convertedBy": "Elevate custom TFJS exporter",
        "modelTopology": model_topology,
        "weightsManifest": [{"paths": [shard_filename], "weights": weight_specs}],
    }
    (tfjs_out_dir / "model.json").write_text(
        json.dumps(model_artifact, indent=2), encoding="utf-8"
    )

    total_mb = sum(len(b) for b in weight_bytes_parts) / 1_048_576
    print(
        f"[emotion-cnn] TFJS export complete → {tfjs_out_dir}  "
        f"({len(weight_specs)} weight tensors, {total_mb:.1f} MB)"
    )



def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = AI_MODELS_DIR / "emotion_model.h5"
    info_path = AI_MODELS_DIR / "emotion_model_info.json"

    print("[emotion-cnn] Collecting dataset")
    files, labels = _collect_paths()
    split = _split_dataset(files, labels)
    class_weight = _to_class_weight(split.train_labels)

    ds_train = _build_tf_dataset(split.train_files, split.train_labels, training=True)
    ds_val = _build_tf_dataset(split.val_files, split.val_labels, training=False)
    ds_test = _build_tf_dataset(split.test_files, split.test_labels, training=False)

    model, backbone = _build_model(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
 
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    if EARLY_STOP_PATIENCE > 0:
        callbacks.insert(
            0,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
            ),
        )
 
    print("[emotion-cnn] Training head (backbone frozen)")
    hist_head = model.fit(
        ds_train, validation_data=ds_val, epochs=EPOCHS_HEAD,
        class_weight=class_weight, callbacks=callbacks, verbose=1,
    )
 
    print("[emotion-cnn] Fine-tuning backbone (unfreezing top layers)")
    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_AT]:
        layer.trainable = False
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # small LR — don't wreck pretrained weights
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    hist_tune = model.fit(
        ds_train, validation_data=ds_val, epochs=EPOCHS_FINETUNE,
        class_weight=class_weight, callbacks=callbacks, verbose=1,
    )
    epochs_ran_tune = len(hist_tune.history.get("loss", []))
    if epochs_ran_tune < EPOCHS_FINETUNE:
        print(
            "[emotion-cnn] Fine-tuning stopped early at "
            f"{epochs_ran_tune}/{EPOCHS_FINETUNE} epochs "
            f"(EarlyStopping patience={EARLY_STOP_PATIENCE})."
        )

    # print("[emotion-cnn] Fine-tuning backbone")
    # base_model = None
    # for layer in model.layers:
    #     if isinstance(layer, tf.keras.Model) and "efficientnetv2" in layer.name.lower():
    #         base_model = layer
    #         break
    # if base_model is not None:
    #     base_model.trainable = True
    #     for layer in base_model.layers[:FINE_TUNE_AT]:
    #         layer.trainable = False

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=["accuracy"],
    # )
    # hist_tune = model.fit(
    #     ds_train,
    #     validation_data=ds_val,
    #     epochs=EPOCHS_FINETUNE,
    #     class_weight=class_weight,
    #     callbacks=callbacks,
    #     verbose=1,
    # )

    val_metrics = _evaluate(model, ds_val, split.val_labels)
    test_metrics = _evaluate(model, ds_test, split.test_labels)

    model.save(model_path, include_optimizer=False)
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "model_type": "efficientnetv2b0_transfer",
        "architecture": "EfficientNetV2B0 + Dense(256) + Softmax(7)",
        "img_size": list(IMG_SIZE),
        "class_names": CLASS_NAMES,
        "dataset_counts": {
            "train": int(len(split.train_files)),
            "val": int(len(split.val_files)),
            "test": int(len(split.test_files)),
            "total": int(len(files)),
        },
        "class_weight": class_weight,
        "training_history": {
            "epochs_run": len(hist_head.history.get("loss", [])),
            "best_val_accuracy": float(max(hist_head.history.get("val_accuracy", [0.0]))),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_accuracy": val_metrics.get("accuracy", 0.0),
    }
    info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[emotion-cnn] Saved model: {model_path}")
    print(f"[emotion-cnn] Saved metadata: {info_path}")
    print(
        "[emotion-cnn] Test accuracy="
        f"{test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}"
    )
    
    # ==========================================
    # AUTO-CONVERT & GITHUB PUSH AUTOMATION
    # ==========================================
    try:
        print("[emotion-cnn] Converting model to TFJS format...")
        tfjs_out_dir = FRONTEND_DIR / "js" / "emotion_tfjs"
        tfjs_out_dir.mkdir(parents=True, exist_ok=True)

        # Clear out old web weights before writing new ones
        for item in tfjs_out_dir.iterdir():
            if item.is_file():
                item.unlink()

        _write_tfjs_layers_model(model, tfjs_out_dir)
        print("[emotion-cnn] Conversion successful!")

        # Push to GitHub
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            print("[emotion-cnn] Pushing web model to GitHub...")
            # Use the Elevate repo name
            repo_url = f"https://{github_token}@github.com/Varun311004/Elevate.git"
            
            # Configure Git Bot
            subprocess.run(["git", "config", "--global", "user.email", "ai-bot@elevate.com"], check=False)
            subprocess.run(["git", "config", "--global", "user.name", "Elevate"], check=False)
            
            # Add, Commit, Push
            subprocess.run(["git", "add", str(tfjs_out_dir)], cwd=ROOT, check=True)
            subprocess.run(["git", "commit", "-m", "Auto-deploy: Update CNN Emotion Model TFJS weights"], cwd=ROOT, check=True)
            subprocess.run(["git", "push", repo_url, "main"], cwd=ROOT, check=True)
            print("✅ [emotion-cnn] SUCCESSFULLY PUSHED NEW MODEL TO GITHUB!")
        else:
            print("⚠️ [emotion-cnn] Skipping GitHub push: GITHUB_TOKEN not found in environment.")

    except FileNotFoundError as e:
        print(
            "[emotion-cnn] Automation Pipeline Failed: missing executable while running "
            f"TFJS/Git step. filename={getattr(e, 'filename', None)} err={e}"
        )
    except subprocess.CalledProcessError as e:
        print(
            "[emotion-cnn] Automation Pipeline Failed: command exited non-zero. "
            f"returncode={e.returncode} cmd={e.cmd}"
        )
    except Exception as e:
        print(f"[emotion-cnn] Automation Pipeline Failed: {e}")


if __name__ == "__main__":
    main()

