"""
Elevate — final server-side emotion inference.

The browser and server use the exact same model artifact and preprocessing
contract:

    RGB pixels [0,255] -> model's input_rescale -> [-1,1] -> MobileNetV2 -> 7 classes
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time

import numpy as np
from flask import Blueprint, jsonify, request

try:
    from PIL import Image
    import tensorflow as tf
except ImportError:
    Image = None
    tf = None

ai_emotion_bp = Blueprint("ai_emotion", __name__)
logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
_MODEL_PATH = os.path.join(_BACKEND_DIR, "ai_models", "emotion_model.h5")
_INFO_PATH = os.path.join(_BACKEND_DIR, "ai_models", "emotion_model_info.json")

CLASS_NAMES = [
    "happy", "bored", "focused", "confused",
    "neutral", "angry", "surprised"
]

IMG_SIZE = (96, 96)
MAX_B64_CHARS = 2 * 1024 * 1024

_model = None
_info = {}
_load_error = None


def _load_once():
    global _model, _info, _load_error

    if _model is not None or _load_error is not None:
        return

    if tf is None:
        _load_error = "TensorFlow is not installed."
        return

    if Image is None:
        _load_error = "Pillow is not installed."
        return

    if not os.path.exists(_MODEL_PATH):
        _load_error = f"Emotion model not found: {_MODEL_PATH}"
        return

    try:
        if os.path.exists(_INFO_PATH):
            with open(_INFO_PATH, encoding="utf-8") as f:
                _info = json.load(f)

        _model = tf.keras.models.load_model(_MODEL_PATH, compile=False)

        if tuple(_model.input_shape[1:]) != (96, 96, 3):
            raise RuntimeError(f"Unexpected model input shape: {_model.input_shape}")

        if tuple(_model.output_shape[1:]) != (7,):
            raise RuntimeError(f"Unexpected model output shape: {_model.output_shape}")

        logger.info(
            "[EmotionDeploy] Loaded emotion model: input=%s output=%s",
            _model.input_shape,
            _model.output_shape,
        )
    except Exception as exc:
        _model = None
        _load_error = f"Failed to load emotion model: {exc}"
        logger.exception("[EmotionDeploy] emotion model load failed")


def _decode_image(data_uri: str) -> np.ndarray:
    if not isinstance(data_uri, str) or not data_uri.strip():
        raise ValueError("image is required")

    if len(data_uri) > MAX_B64_CHARS:
        raise ValueError("image is too large")

    raw = data_uri.split(",", 1)[1] if "," in data_uri else data_uri

    try:
        binary = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise ValueError(f"invalid base64 image: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(binary)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"invalid image: {exc}") from exc

    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)

    # Production model contract:
    # float32 RGB values in [0,1].
    # The Keras model internally converts [0,1] -> [-1,1].
    arr = np.asarray(image, dtype=np.float32) / 255.0
    
    return arr[None, ...]

@ai_emotion_bp.post("/predict")
def predict_emotion():
    _load_once()

    if _load_error:
        return jsonify({"error": _load_error}), 503

    body = request.get_json(silent=True) or {}

    try:
        started = time.perf_counter()
        image = _decode_image(body.get("image", ""))

        probs = np.asarray(_model(image, training=False)).reshape(-1)

        if probs.shape != (7,):
            raise RuntimeError(f"Model returned unexpected shape: {probs.shape}")

        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0:
            raise RuntimeError("Model returned an invalid probability vector")
        probs /= total

        index = int(np.argmax(probs))
        emotion = CLASS_NAMES[index]
        confidence = float(probs[index])

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "all_scores": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(CLASS_NAMES)
            },
            "model_info": {
                "model_type": _info.get("model_type"),
                "architecture": _info.get("architecture"),
                "test_metrics": _info.get("test_metrics"),
                "val_accuracy": (_info.get("validation_metrics") or {}).get("accuracy"),
            },
            "latency_ms": int((time.perf_counter() - started) * 1000),
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("[EmotionDeploy] inference error")
        return jsonify({"error": f"Inference failed: {exc}"}), 500


@ai_emotion_bp.get("/status")
def emotion_status():
    _load_once()

    return jsonify({
        "model_loaded": _model is not None,
        "model_exists": os.path.exists(_MODEL_PATH),
        "model_path": _MODEL_PATH,
        "inference_mode": "server" if _model is not None else "unavailable",
        "error": _load_error,
        "class_names": CLASS_NAMES,
        "info": _info,
        "tensorflow_ver": tf.__version__ if tf is not None else None,
    })


def inspect_emotion_artifacts() -> dict:
    project_root = os.path.dirname(_BACKEND_DIR)
    tfjs_dir = os.path.join(project_root, "frontend", "js", "emotion_tfjs")

    model_json = os.path.join(tfjs_dir, "model.json")

    shard_files = []
    if os.path.isdir(tfjs_dir):
        shard_files = [
            f for f in os.listdir(tfjs_dir)
            if f.endswith(".bin")
        ]

    info = {}
    if os.path.exists(_INFO_PATH):
        try:
            with open(_INFO_PATH, encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}

    return {
        "backend_model_exists": os.path.exists(_MODEL_PATH),
        "backend_model_path": _MODEL_PATH,
        "metadata_exists": os.path.exists(_INFO_PATH),
        "metadata_path": _INFO_PATH,
        "tfjs_model_exists": os.path.exists(model_json),
        "tfjs_model_path": model_json,
        "tfjs_weights_exists": bool(shard_files),
        "tfjs_weights": shard_files,
        "training_summary": {
            "model_type": info.get("model_type"),
            "accuracy": (info.get("validation_metrics") or {}).get("accuracy"),
            "class_names": info.get("class_names"),
            "timestamp": info.get("timestamp"),
        },
    }
