"""
Elevate — Final Emotion CNN Training Pipeline

Purpose:
    Train a 7-class facial/emotional-state classifier and safely deploy
    both the Python/Keras model and its TensorFlow.js equivalent.

Outputs:
    backend/ai_models/emotion_model.h5
    backend/ai_models/emotion_model_info.json

    frontend/js/emotion_tfjs/model.json
    frontend/js/emotion_tfjs/*.bin

Dataset folders:
    dataset/angry
    dataset/Bored
    dataset/confused
    dataset/Focused
    dataset/happy
    dataset/neutral
    dataset/surprise

Canonical class order:
    0 happy
    1 bored
    2 focused
    3 confused
    4 neutral
    5 angry
    6 surprised

IMPORTANT:
    The exported model expects FLOAT32 RGB images in [0, 1].

    Backend:
        uint8 RGB -> /255 -> model

    Browser:
        tf.browser.fromPixels()
        -> /255
        -> model

    The model itself converts [0, 1] -> [-1, 1] before MobileNetV2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "dataset"

AI_MODELS_DIR = ROOT / "backend" / "ai_models"

TFJS_DIR = ROOT / "frontend" / "js" / "emotion_tfjs"

# TensorFlow.js is intentionally NOT installed in the Windows .venv.
# Conversion runs in the dedicated WSL environment so the application
# environment stays limited to TensorFlow/Keras runtime + training deps.
TFJS_VERSION = "4.22.0"
TFJS_WSL_DISTRO = os.environ.get("ELEVATE_TFJS_WSL_DISTRO", "Ubuntu")
TFJS_WSL_VENV = os.environ.get(
    "ELEVATE_TFJS_WSL_VENV",
    "~/.elevate-tfjs/.venv",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "happy",
    "bored",
    "focused",
    "confused",
    "neutral",
    "angry",
    "surprised",
]

CLASS_FOLDER_ALIASES = {
    "happy": ["happy"],
    "bored": ["bored"],
    "focused": ["focused"],
    "confused": ["confused"],
    "neutral": ["neutral"],
    "angry": ["angry"],
    "surprised": ["surprised", "surprise"],
}

IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_CHANNELS = 3

INPUT_SHAPE = (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
)

NUM_CLASSES = len(CLASS_NAMES)

SEED = 42

BATCH_SIZE = int(
    os.environ.get(
        "ELEVATE_EMOTION_BATCH_SIZE",
        "32",
    )
)

HEAD_EPOCHS = int(
    os.environ.get(
        "ELEVATE_EMOTION_HEAD_EPOCHS",
        "12",
    )
)

FINETUNE_EPOCHS = int(
    os.environ.get(
        "ELEVATE_EMOTION_FINETUNE_EPOCHS",
        "10",
    )
)

HEAD_LEARNING_RATE = 1e-3

FINETUNE_LEARNING_RATE = 1e-5

EARLY_STOPPING_PATIENCE = 3

FINE_TUNE_LAYERS = 40

MIN_IMAGES_PER_CLASS = 100

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class DatasetSplit:
    train_files: List[str]
    train_labels: np.ndarray

    val_files: List[str]
    val_labels: np.ndarray

    test_files: List[str]
    test_labels: np.ndarray


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility() -> None:
    """
    Use TensorFlow's official unified seed helper.

    We intentionally do not enable full op determinism because that can
    significantly reduce CPU training performance on this laptop.
    """
    tf.keras.utils.set_random_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_class_folder(class_name: str) -> Path:
    aliases = CLASS_FOLDER_ALIASES[class_name]

    available = {
        p.name.strip().lower(): p
        for p in DATASET_DIR.iterdir()
        if p.is_dir()
    }

    for alias in aliases:
        candidate = available.get(alias.lower())

        if candidate is not None:
            return candidate

    raise FileNotFoundError(
        f"\nMissing dataset folder for class '{class_name}'.\n"
        f"Expected one of: {aliases}\n"
        f"Dataset location: {DATASET_DIR}\n"
        f"Available folders: {sorted(available.keys())}"
    )


def collect_dataset() -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{DATASET_DIR}"
        )

    all_files: List[str] = []
    all_labels: List[int] = []

    counts: Dict[str, int] = {}

    print()
    print("=" * 70)
    print("DATASET DISCOVERY")
    print("=" * 70)
    print(f"Dataset: {DATASET_DIR}")
    print()

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = find_class_folder(class_name)

        files = sorted(
            str(path)
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        if len(files) < MIN_IMAGES_PER_CLASS:
            raise RuntimeError(
                f"Class '{class_name}' contains only {len(files)} images. "
                f"At least {MIN_IMAGES_PER_CLASS} are required."
            )

        counts[class_name] = len(files)

        all_files.extend(files)
        all_labels.extend(
            [class_index] * len(files)
        )

        print(
            f"{class_name:>10}: "
            f"{len(files):>6,} images"
        )

    print()
    print(f"Total images: {len(all_files):,}")

    return (
        all_files,
        np.asarray(
            all_labels,
            dtype=np.int32,
        ),
        counts,
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_images(
    files: List[str],
) -> List[str]:
    """
    Validate image headers before TensorFlow training starts.

    A corrupt image causes a late tf.data failure otherwise.
    """
    print()
    print("=" * 70)
    print("IMAGE VALIDATION")
    print("=" * 70)

    valid_files: List[str] = []

    invalid_files: List[str] = []

    total = len(files)

    for index, path in enumerate(files, start=1):

        try:
            with Image.open(path) as image:
                image.verify()

            valid_files.append(path)

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as exc:

            invalid_files.append(path)

            print(
                f"[INVALID] {path}\n"
                f"          {exc}"
            )

        if index % 1000 == 0 or index == total:
            print(
                f"Validated {index:,}/{total:,}",
                end="\r",
                flush=True,
            )

    print()

    if invalid_files:
        print(
            f"\nFound {len(invalid_files)} invalid images."
        )

        raise RuntimeError(
            "Training stopped because invalid/corrupt images "
            "were found. Remove or repair them and run again."
        )

    print(
        f"All {len(valid_files):,} images passed validation."
    )

    return valid_files


# ============================================================
# EXACT DUPLICATE DETECTION
# ============================================================

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def check_exact_duplicates(
    files: List[str],
    labels: np.ndarray,
) -> None:
    """
    Detect byte-identical images occurring in multiple classes.

    Exact duplicates within the same class are harmless for this purpose.
    Cross-class duplicates indicate potential label leakage.
    """
    print()
    print("=" * 70)
    print("EXACT DUPLICATE CHECK")
    print("=" * 70)

    hashes: Dict[str, Tuple[str, int]] = {}

    cross_class_duplicates = []

    for index, (path, label) in enumerate(
        zip(files, labels),
        start=1,
    ):
        digest = sha256_file(path)

        previous = hashes.get(digest)

        if previous is not None:
            previous_path, previous_label = previous

            if previous_label != int(label):
                cross_class_duplicates.append(
                    (
                        previous_path,
                        path,
                        CLASS_NAMES[previous_label],
                        CLASS_NAMES[int(label)],
                    )
                )
        else:
            hashes[digest] = (
                path,
                int(label),
            )

        if index % 1000 == 0 or index == len(files):
            print(
                f"Hashed {index:,}/{len(files):,}",
                end="\r",
                flush=True,
            )

    print()

    if cross_class_duplicates:
        print(
            "\nCross-class exact duplicates detected:"
        )

        for item in cross_class_duplicates[:20]:
            print(
                f"  {item[2]} <-> {item[3]}\n"
                f"    {item[0]}\n"
                f"    {item[1]}"
            )

        raise RuntimeError(
            "Cross-class duplicate images were detected. "
            "Training stopped to prevent label leakage."
        )

    print("No cross-class exact duplicates detected.")


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def split_dataset(
    files: List[str],
    labels: np.ndarray,
) -> DatasetSplit:

    (
        train_files,
        temp_files,
        train_labels,
        temp_labels,
    ) = train_test_split(
        files,
        labels,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=labels,
        random_state=SEED,
    )

    relative_test_ratio = (
        TEST_RATIO
        / (VAL_RATIO + TEST_RATIO)
    )

    (
        val_files,
        test_files,
        val_labels,
        test_labels,
    ) = train_test_split(
        temp_files,
        temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=SEED,
    )

    split = DatasetSplit(
        train_files=list(train_files),
        train_labels=np.asarray(
            train_labels,
            dtype=np.int32,
        ),
        val_files=list(val_files),
        val_labels=np.asarray(
            val_labels,
            dtype=np.int32,
        ),
        test_files=list(test_files),
        test_labels=np.asarray(
            test_labels,
            dtype=np.int32,
        ),
    )

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Train: {len(split.train_files):,}"
    )
    print(
        f"Val:   {len(split.val_files):,}"
    )
    print(
        f"Test:  {len(split.test_files):,}"
    )

    return split


# ============================================================
# CLASS BALANCING
# ============================================================

def compute_class_weights(
    train_labels: np.ndarray,
) -> Dict[int, float]:

    counts = np.bincount(
        train_labels,
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    total = float(np.sum(counts))

    raw_weights = np.sqrt(
        total
        / (
            NUM_CLASSES
            * np.maximum(counts, 1.0)
        )
    )

    # Normalize to weighted mean ~= 1.
    weighted_mean = (
        np.sum(raw_weights * counts)
        / total
    )

    normalized = raw_weights / weighted_mean

    weights = {
        index: float(normalized[index])
        for index in range(NUM_CLASSES)
    }

    print()
    print("=" * 70)
    print("CLASS BALANCE WEIGHTS")
    print("=" * 70)

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:>10}: "
            f"{weights[index]:.4f}"
        )

    return weights


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_image(
    path: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
    )

    # External model contract:
    # float32 values in [0, 1].
    image = tf.cast(
        image,
        tf.float32,
    ) / 255.0

    return image, label


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_augmentation() -> tf.keras.Sequential:
    """
    Conservative augmentation for webcam facial-expression data.

    We deliberately avoid aggressive geometric transformations because
    facial expression recognition depends on subtle spatial features.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal",
                seed=SEED,
            ),
            tf.keras.layers.RandomRotation(
                factor=0.06,
                fill_mode="reflect",
                seed=SEED,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.10),
                width_factor=(-0.08, 0.10),
                fill_mode="reflect",
                seed=SEED,
            ),
            tf.keras.layers.RandomContrast(
                factor=0.12,
                seed=SEED,
            ),
        ],
        name="emotion_augmentation",
    )


# ============================================================
# TF.DATA DATASET
# ============================================================

def build_dataset(
    files: List[str],
    labels: np.ndarray,
    training: bool,
    class_weights: Dict[int, float] | None = None,
) -> tf.data.Dataset:

    ds = tf.data.Dataset.from_tensor_slices(
        (
            files,
            labels,
        )
    )

    if training:
        ds = ds.shuffle(
            buffer_size=len(files),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = create_augmentation()

        def augment(
            image: tf.Tensor,
            label: tf.Tensor,
        ):
            return (
                augmentation(
                    image,
                    training=True,
                ),
                label,
            )

        ds = ds.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if class_weights is not None:

        weight_vector = tf.constant(
            [
                class_weights[index]
                for index in range(NUM_CLASSES)
            ],
            dtype=tf.float32,
        )

        def add_weight(
            image: tf.Tensor,
            label: tf.Tensor,
        ):
            weight = tf.gather(
                weight_vector,
                label,
            )

            return (
                image,
                tf.one_hot(
                    label,
                    depth=NUM_CLASSES,
                ),
                weight,
            )

        ds = ds.map(
            add_weight,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    else:

        def one_hot(
            image: tf.Tensor,
            label: tf.Tensor,
        ):
            return (
                image,
                tf.one_hot(
                    label,
                    depth=NUM_CLASSES,
                ),
            )

        ds = ds.map(
            one_hot,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds


# ============================================================
# MODEL
# ============================================================

def build_model(
    weights: str | None = "imagenet",
) -> Tuple[
    tf.keras.Model,
    tf.keras.Model,
]:

    inputs = tf.keras.Input(
        shape=INPUT_SHAPE,
        dtype=tf.float32,
        name="image",
    )

    # The model receives [0, 1].
    #
    # MobileNetV2 expects [-1, 1].
    #
    # Therefore:
    #     x -> 2x - 1
    #
    x = tf.keras.layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenetv2_normalization",
    )(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights=weights,
        pooling=None,
    )

    backbone.trainable = False

    x = backbone(
        x,
        training=False,
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        name="emotion_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="emotion_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="elevate_emotion_mobilenetv2",
    )

    return model, backbone


# ============================================================
# COMPILE
# ============================================================

def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(
                name="accuracy"
            )
        ],
    )


# ============================================================
# FINE-TUNING
# ============================================================

def configure_fine_tuning(
    backbone: tf.keras.Model,
) -> None:

    backbone.trainable = True

    total_layers = len(backbone.layers)

    freeze_until = max(
        0,
        total_layers - FINE_TUNE_LAYERS,
    )

    for index, layer in enumerate(
        backbone.layers
    ):

        if index < freeze_until:
            layer.trainable = False

        else:
            layer.trainable = True

        # BatchNorm statistics should remain frozen during fine-tuning.
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False

    trainable_layers = sum(
        1
        for layer in backbone.layers
        if layer.trainable
    )

    print()
    print(
        f"MobileNetV2 total layers: "
        f"{total_layers}"
    )

    print(
        f"MobileNetV2 trainable layers: "
        f"{trainable_layers}"
    )


# ============================================================
# CALLBACKS
# ============================================================

def make_callbacks(
    checkpoint_path: Path,
) -> List[tf.keras.callbacks.Callback]:

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_labels: np.ndarray,
) -> Dict:

    probabilities = model.predict(
        test_ds,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    y_true = np.asarray(
        test_labels,
        dtype=np.int32,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_precision = precision_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(NUM_CLASSES)),
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


# ============================================================
# TF.JS VALIDATION
# ============================================================

def validate_tfjs_artifact(
    directory: Path,
) -> Dict:

    model_json_path = (
        directory / "model.json"
    )

    if not model_json_path.is_file():
        raise RuntimeError(
            "TF.js conversion did not create model.json."
        )

    data = json.loads(
        model_json_path.read_text(
            encoding="utf-8"
        )
    )

    manifest = data.get(
        "weightsManifest"
    )

    if not isinstance(
        manifest,
        list,
    ) or not manifest:
        raise RuntimeError(
            "TF.js model.json has no weightsManifest."
        )

    total_expected_bytes = 0
    total_actual_bytes = 0

    for group in manifest:

        paths = group.get(
            "paths",
            [],
        )

        if not paths:
            raise RuntimeError(
                "TF.js weight manifest group has no paths."
            )

        for relative_path in paths:

            shard = directory / relative_path

            if not shard.is_file():
                raise RuntimeError(
                    "TF.js manifest references missing "
                    f"weight shard: {relative_path}"
                )

            total_actual_bytes += (
                shard.stat().st_size
            )

        for weight in group.get(
            "weights",
            [],
        ):

            shape = weight.get(
                "shape",
                [],
            )

            elements = int(
                np.prod(shape)
            )

            total_expected_bytes += (
                elements * 4
            )

    if total_actual_bytes < total_expected_bytes:
        raise RuntimeError(
            "TF.js weight shards appear truncated.\n"
            f"Expected at least: {total_expected_bytes:,} bytes\n"
            f"Found:              {total_actual_bytes:,} bytes"
        )

    topology = (
        data.get("modelTopology")
        or {}
    )

    model_config = (
        topology.get("model_config")
        or {}
    )

    config = (
        model_config.get("config")
        or {}
    )

    serialized = json.dumps(
        data,
        separators=(",", ":"),
    ).lower()

    forbidden = [
        "dtypepolicy",
        "hardsilu",
        "hard_silu",
    ]

    for token in forbidden:

        if token in serialized:
            raise RuntimeError(
                f"Unexpected TF.js serialization token: {token}"
            )

    class_count = NUM_CLASSES

    output_shape = None

    layers = config.get(
        "layers",
        [],
    )

    for layer in layers:

        if layer.get("name") == "emotion_output":

            layer_config = (
                layer.get("config")
                or {}
            )

            output_shape = layer_config.get(
                "units"
            )

    if output_shape != class_count:
        raise RuntimeError(
            "TF.js output layer does not contain "
            f"{class_count} classes."
        )

    print(
        "[emotion-cnn] TF.js artifact validation passed."
    )

    return {
        "model_json": str(
            model_json_path
        ),
        "weight_shards": sum(
            len(group.get("paths", []))
            for group in manifest
        ),
        "expected_weight_bytes": (
            total_expected_bytes
        ),
        "actual_weight_bytes": (
            total_actual_bytes
        ),
    }


# ============================================================
# TF.JS CONVERSION
# ============================================================

def _wsl_path(path: Path) -> str:
    """Convert a Windows project path to an absolute WSL path."""
    completed = subprocess.run(
        [
            "wsl.exe",
            "-d",
            TFJS_WSL_DISTRO,
            "--",
            "wslpath",
            "-a",
            str(path.resolve()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "Unable to convert a Windows path to WSL. "
            f"Ensure WSL distribution '{TFJS_WSL_DISTRO}' is installed. "
            f"Details: {detail}"
        )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("wslpath returned an empty path.")
    return value


def _run_wsl_tfjs_converter(
    input_h5: Path,
    output_directory: Path,
) -> None:
    """Run the pinned TF.js converter inside the dedicated WSL venv."""
    if os.name != "nt":
        raise RuntimeError(
            "The local emotion TF.js converter is configured for Windows + WSL. "
            "Run the training pipeline from the supported Windows development environment."
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    wsl_h5 = _wsl_path(input_h5)
    wsl_output = _wsl_path(output_directory)

    # Use bash so $HOME in the configured venv path is resolved by WSL, not Windows.
    command = (
        f"{TFJS_WSL_VENV}/bin/tensorflowjs_converter "
        f"--input_format=keras "
        f"{shlex.quote(wsl_h5)} "
        f"{shlex.quote(wsl_output)}"
    )

    completed = subprocess.run(
        [
            "wsl.exe",
            "-d",
            TFJS_WSL_DISTRO,
            "--",
            "bash",
            "-lc",
            command,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        details = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        raise RuntimeError(
            "TensorFlow.js conversion failed inside WSL. "
            "The dedicated WSL converter environment must contain "
            f"tensorflow==2.15.0, tensorflow-decision-forests==1.8.1, "
            f"and tensorflowjs=={TFJS_VERSION}.\n"
            f"{details}"
        )


def _convert_h5_to_tfjs(
    input_h5: Path,
    output_directory: Path,
) -> Dict:
    if not input_h5.is_file():
        raise RuntimeError(f"Keras H5 model not found: {input_h5}")

    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    print()
    print(
        "[emotion-cnn] Exporting model to TensorFlow.js "
        f"via WSL (tensorflowjs {TFJS_VERSION})..."
    )

    _run_wsl_tfjs_converter(input_h5, output_directory)
    return validate_tfjs_artifact(output_directory)


def convert_to_tfjs(
    model: tf.keras.Model,
    output_directory: Path,
) -> Dict:
    """Save a temporary H5 model and convert it through WSL."""
    with tempfile.TemporaryDirectory(prefix="elevate_emotion_tfjs_bridge_") as temp:
        temporary_h5 = Path(temp) / "emotion_model.h5"
        model.save(temporary_h5, include_optimizer=False)
        return _convert_h5_to_tfjs(temporary_h5, output_directory)


# ============================================================
# PRE-FLIGHT TEST
# ============================================================

def run_tfjs_preflight() -> None:

    print()
    print("=" * 70)
    print("TF.JS PRODUCTION ARCHITECTURE PREFLIGHT")
    print("=" * 70)

    model, _ = build_model(
        weights=None
    )

    with tempfile.TemporaryDirectory(
        prefix="elevate_emotion_preflight_"
    ) as temp:

        output = (
            Path(temp)
            / "emotion_tfjs"
        )

        convert_to_tfjs(
            model,
            output,
        )

        # Also verify the model itself can perform inference.
        dummy = np.zeros(
            (
                1,
                IMG_HEIGHT,
                IMG_WIDTH,
                IMG_CHANNELS,
            ),
            dtype=np.float32,
        )

        prediction = model.predict(
            dummy,
            verbose=0,
        )

        if prediction.shape != (
            1,
            NUM_CLASSES,
        ):
            raise RuntimeError(
                "Production model produced unexpected "
                f"output shape: {prediction.shape}"
            )

        if not np.isfinite(
            prediction
        ).all():
            raise RuntimeError(
                "Production model produced NaN/Inf values."
            )

        if not np.isclose(
            np.sum(prediction),
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Production softmax output does not sum to 1."
            )

    tf.keras.backend.clear_session()

    print(
        "[emotion-cnn] TF.js production preflight PASSED."
    )


# ============================================================
# ATOMIC DEPLOYMENT
# ============================================================

def deploy_artifacts(
    staged_h5: Path,
    staged_info: Path,
    staged_tfjs: Path,
) -> None:

    AI_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFJS_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_h5 = (
        AI_MODELS_DIR
        / "emotion_model.h5"
    )

    final_info = (
        AI_MODELS_DIR
        / "emotion_model_info.json"
    )

    final_tfjs = TFJS_DIR

    backup_root = (
        ROOT
        / ".emotion_model_previous"
    )

    if backup_root.exists():
        shutil.rmtree(
            backup_root
        )

    backup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_h5 = (
        backup_root
        / "emotion_model.h5"
    )

    previous_info = (
        backup_root
        / "emotion_model_info.json"
    )

    previous_tfjs = (
        backup_root
        / "emotion_tfjs"
    )

    # Backup current artifacts.
    if final_h5.exists():
        shutil.copy2(
            final_h5,
            previous_h5,
        )

    if final_info.exists():
        shutil.copy2(
            final_info,
            previous_info,
        )

    if final_tfjs.exists():
        shutil.copytree(
            final_tfjs,
            previous_tfjs,
        )

    try:

        staged_h5.replace(
            final_h5
        )

        staged_info.replace(
            final_info
        )

        if final_tfjs.exists():
            shutil.rmtree(
                final_tfjs
            )

        staged_tfjs.replace(
            final_tfjs
        )

    except Exception:

        # Restore previous artifacts.
        if previous_h5.exists():
            previous_h5.replace(
                final_h5
            )

        if previous_info.exists():
            previous_info.replace(
                final_info
            )

        if previous_tfjs.exists():

            if final_tfjs.exists():
                shutil.rmtree(
                    final_tfjs
                )

            previous_tfjs.replace(
                final_tfjs
            )

        raise

    print()
    print("=" * 70)
    print("MODEL DEPLOYMENT COMPLETE")
    print("=" * 70)

    print(
        f"Keras model: {final_h5}"
    )

    print(
        f"Metadata:    {final_info}"
    )

    print(
        f"TF.js:       {final_tfjs}"
    )


# ============================================================
# METADATA
# ============================================================

def write_metadata(
    path: Path,
    dataset_counts: Dict[str, int],
    split: DatasetSplit,
    class_weights: Dict[int, float],
    test_metrics: Dict,
    head_history: Dict,
    fine_history: Dict,
) -> None:

    metadata = {
        "model_name": "elevate_emotion_mobilenetv2",
        "model_type": "mobilenetv2_transfer_learning",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "tensorflow_version": tf.__version__,
        "tensorflowjs_version": TFJS_VERSION,

        "input": {
            "shape": [
                IMG_HEIGHT,
                IMG_WIDTH,
                IMG_CHANNELS,
            ],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "channels": "RGB",
        },

        "preprocessing": {
            "external_range": "[0,1]",
            "internal_operation": (
                "x * 2.0 - 1.0"
            ),
            "backbone_expected_range": "[-1,1]",
        },

        "class_names": CLASS_NAMES,

        "class_folder_mapping": {
            class_name: CLASS_FOLDER_ALIASES[
                class_name
            ]
            for class_name in CLASS_NAMES
        },

        "dataset_counts": dataset_counts,

        "split_counts": {
            "train": len(
                split.train_files
            ),
            "validation": len(
                split.val_files
            ),
            "test": len(
                split.test_files
            ),
            "total": (
                len(split.train_files)
                + len(split.val_files)
                + len(split.test_files)
            ),
        },

        "class_balance_weights": {
            CLASS_NAMES[index]: float(
                class_weights[index]
            )
            for index in range(NUM_CLASSES)
        },

        "training": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "head_epochs_requested": HEAD_EPOCHS,
            "fine_tune_epochs_requested": FINETUNE_EPOCHS,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "fine_tune_learning_rate": FINETUNE_LEARNING_RATE,
            "fine_tune_layers": FINE_TUNE_LAYERS,
        },

        "training_history": {
            "head_epochs_run": len(
                head_history.get(
                    "loss",
                    [],
                )
            ),
            "fine_tune_epochs_run": len(
                fine_history.get(
                    "loss",
                    [],
                )
            ),
            "best_head_val_loss": (
                float(
                    min(
                        head_history.get(
                            "val_loss",
                            [float("inf")],
                        )
                    )
                )
                if head_history.get("val_loss")
                else None
            ),
            "best_fine_tune_val_loss": (
                float(
                    min(
                        fine_history.get(
                            "val_loss",
                            [float("inf")],
                        )
                    )
                )
                if fine_history.get("val_loss")
                else None
            ),
        },

        "test_metrics": test_metrics,

        # Existing backend expects this field.
        "val_accuracy": float(
            test_metrics["accuracy"]
        ),

        "tfjs": {
            "format": "layers_model",
            "loader": "tf.loadLayersModel",
            "path": "/js/emotion_tfjs/model.json",
        },
    }

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train(
    skip_image_validation: bool = False,
) -> None:

    set_reproducibility()

    print()
    print("=" * 70)
    print("ELEVATE EMOTION CNN")
    print("=" * 70)

    print(
        f"Python:       {sys.version.split()[0]}"
    )

    print(
        f"TensorFlow:   {tf.__version__}"
    )

    print(
        f"TF.js converter: {TFJS_VERSION} (WSL)"
    )

    print(
        f"Dataset:      {DATASET_DIR}"
    )

    print(
        f"Input:        {INPUT_SHAPE}"
    )

    print(
        f"Classes:      {NUM_CLASSES}"
    )

    # --------------------------------------------------------
    # 1. Collect dataset
    # --------------------------------------------------------

    files, labels, dataset_counts = (
        collect_dataset()
    )

    # --------------------------------------------------------
    # 2. Validate image files
    # --------------------------------------------------------

    if not skip_image_validation:
        validate_images(files)

    # --------------------------------------------------------
    # 3. Detect exact duplicate leakage
    # --------------------------------------------------------

    check_exact_duplicates(
        files,
        labels,
    )

    # --------------------------------------------------------
    # 4. Stratified split
    # --------------------------------------------------------

    split = split_dataset(
        files,
        labels,
    )

    # --------------------------------------------------------
    # 5. Class weighting
    # --------------------------------------------------------

    class_weights = (
        compute_class_weights(
            split.train_labels
        )
    )

    # --------------------------------------------------------
    # 6. Build datasets
    # --------------------------------------------------------

    print()
    print(
        "[emotion-cnn] Building tf.data pipelines..."
    )

    train_ds = build_dataset(
        split.train_files,
        split.train_labels,
        training=True,
        class_weights=class_weights,
    )

    val_ds = build_dataset(
        split.val_files,
        split.val_labels,
        training=False,
    )

    test_ds = build_dataset(
        split.test_files,
        split.test_labels,
        training=False,
    )

    # --------------------------------------------------------
    # 7. Build exact production model
    # --------------------------------------------------------

    print()
    print(
        "[emotion-cnn] Loading MobileNetV2 ImageNet weights..."
    )

    model, backbone = build_model(
        weights="imagenet"
    )

    model.summary()

    # --------------------------------------------------------
    # 8. Compile classifier head
    # --------------------------------------------------------

    compile_model(
        model,
        HEAD_LEARNING_RATE,
    )

    # --------------------------------------------------------
    # 9. Temporary working directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="elevate_emotion_training_",
        dir=str(ROOT),
    ) as working:

        working_dir = Path(
            working
        )

        checkpoint_head = (
            working_dir
            / "best_head.weights.h5"
        )

        checkpoint_fine = (
            working_dir
            / "best_finetune.weights.h5"
        )

        print()
        print("=" * 70)
        print("STAGE 1 — CLASSIFIER HEAD")
        print("=" * 70)

        head_callbacks = make_callbacks(
            checkpoint_head
        )

        history_head = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=HEAD_EPOCHS,
            callbacks=head_callbacks,
            verbose=1,
        )

        if checkpoint_head.exists():
            model.load_weights(
                checkpoint_head
            )

        # ----------------------------------------------------
        # 10. Fine tuning
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("STAGE 2 — MOBILE NET FINE-TUNING")
        print("=" * 70)

        configure_fine_tuning(
            backbone
        )

        compile_model(
            model,
            FINETUNE_LEARNING_RATE,
        )

        fine_callbacks = make_callbacks(
            checkpoint_fine
        )

        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINETUNE_EPOCHS,
            callbacks=fine_callbacks,
            verbose=1,
        )

        if checkpoint_fine.exists():
            model.load_weights(
                checkpoint_fine
            )

        # ----------------------------------------------------
        # 11. Final test evaluation
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL TEST EVALUATION")
        print("=" * 70)

        test_metrics = evaluate_model(
            model,
            test_ds,
            split.test_labels,
        )

        print()
        print(
            f"Accuracy:        "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"Macro Precision: "
            f"{test_metrics['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall:    "
            f"{test_metrics['macro_recall']:.4f}"
        )

        print(
            f"Macro F1:        "
            f"{test_metrics['macro_f1']:.4f}"
        )

        print()
        print("Per-class results:")

        report = (
            test_metrics[
                "classification_report"
            ]
        )

        for class_name in CLASS_NAMES:

            row = report[class_name]

            print(
                f"{class_name:>10}: "
                f"precision={row['precision']:.3f} "
                f"recall={row['recall']:.3f} "
                f"f1={row['f1-score']:.3f}"
            )

        print()
        print("Confusion matrix:")

        print(
            np.asarray(
                test_metrics[
                    "confusion_matrix"
                ]
            )
        )

        # ----------------------------------------------------
        # 12. Save everything into staging
        # ----------------------------------------------------

        staging = (
            working_dir
            / "staging"
        )

        staging.mkdir(
            parents=True,
            exist_ok=True,
        )

        staged_h5 = (
            staging
            / "emotion_model.h5"
        )

        staged_info = (
            staging
            / "emotion_model_info.json"
        )

        staged_tfjs = (
            staging
            / "emotion_tfjs"
        )

        print()
        print(
            "[emotion-cnn] Saving final Keras model..."
        )

        model.save(
            staged_h5,
            include_optimizer=False,
        )

        # ----------------------------------------------------
        # 13. Verify saved Keras model
        # ----------------------------------------------------

        print(
            "[emotion-cnn] Re-loading saved Keras model..."
        )

        reloaded = tf.keras.models.load_model(
            staged_h5,
            compile=False,
        )

        dummy = np.zeros(
            (
                1,
                IMG_HEIGHT,
                IMG_WIDTH,
                IMG_CHANNELS,
            ),
            dtype=np.float32,
        )

        reloaded_prediction = (
            reloaded.predict(
                dummy,
                verbose=0,
            )
        )

        if reloaded_prediction.shape != (
            1,
            NUM_CLASSES,
        ):
            raise RuntimeError(
                "Reloaded Keras model has "
                f"unexpected output shape: "
                f"{reloaded_prediction.shape}"
            )

        print(
            "[emotion-cnn] Saved Keras model "
            "reload test passed."
        )

        # ----------------------------------------------------
        # 14. Convert to TF.js
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("TF.JS EXPORT")
        print("=" * 70)

        tfjs_result = _convert_h5_to_tfjs(
            staged_h5,
            staged_tfjs,
        )

        # ----------------------------------------------------
        # 15. Metadata
        # ----------------------------------------------------

        write_metadata(
            staged_info,
            dataset_counts,
            split,
            class_weights,
            test_metrics,
            history_head.history,
            history_fine.history,
        )

        # ----------------------------------------------------
        # 16. Final staging validation
        # ----------------------------------------------------

        if not staged_h5.is_file():
            raise RuntimeError(
                "Staged H5 model is missing."
            )

        if not staged_info.is_file():
            raise RuntimeError(
                "Staged metadata is missing."
            )

        if not (
            staged_tfjs
            / "model.json"
        ).is_file():
            raise RuntimeError(
                "Staged TF.js model.json is missing."
            )

        print()
        print("=" * 70)
        print("ALL ARTIFACT CHECKS PASSED")
        print("=" * 70)

        print(
            f"Keras: {staged_h5}"
        )

        print(
            f"Metadata: {staged_info}"
        )

        print(
            f"TF.js: {staged_tfjs}"
        )

        print(
            f"TF.js shards: "
            f"{tfjs_result['weight_shards']}"
        )

        # ----------------------------------------------------
        # 17. Deploy atomically
        # ----------------------------------------------------

        deploy_artifacts(
            staged_h5,
            staged_info,
            staged_tfjs,
        )

    tf.keras.backend.clear_session()

    print()
    print("=" * 70)
    print("EMOTION TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train and deploy the Elevate emotion CNN."
        )
    )

    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help=(
            "Skip the PIL corruption check. "
            "Not recommended for the first production run."
        ),
    )

    parser.add_argument(
        "--preflight-tfjs",
        action="store_true",
        help=(
            "Test the exact production architecture "
            "and TF.js conversion without training."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.preflight_tfjs:
        set_reproducibility()
        run_tfjs_preflight()
        return

    train(
        skip_image_validation=(
            args.skip_image_validation
        )
    )


if __name__ == "__main__":
    main()