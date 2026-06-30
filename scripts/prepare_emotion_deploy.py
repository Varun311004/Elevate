"""Strict verification for emotion model deployment artifacts.

This script does not support fallback modes. Missing required artifacts causes
an immediate non-zero exit.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIN_EXPECTED_TEST_ACCURACY = 0.90

REQUIRED_ARTIFACTS = [
    ROOT / "backend" / "ai_models" / "emotion_model_info.json",
    ROOT / "frontend" / "js" / "emotion_tfjs" / "model.json",
    ROOT / "frontend" / "js" / "emotion_tfjs" / "group1-shard1of1.bin",
]


def _load_metadata_summary() -> dict:
    info_path = ROOT / "backend" / "ai_models" / "emotion_model_info.json"
    if not info_path.exists():
        return {}

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[emotion-deploy] ERROR: unable to parse {info_path}: {exc}")
        return {}

    validation = info.get("validation_metrics") if isinstance(info.get("validation_metrics"), dict) else {}
    test_metrics = info.get("test_metrics") if isinstance(info.get("test_metrics"), dict) else {}
    accuracy = test_metrics.get("accuracy")
    if accuracy is None:
        accuracy = validation.get("accuracy")

    class_names = info.get("class_names")
    if not isinstance(class_names, list):
        class_names = []

    return {
        "timestamp": info.get("timestamp"),
        "model_type": info.get("model_type") or info.get("model_name"),
        "accuracy": accuracy,
        "class_names": class_names,
    }


def main() -> int:
    print("[emotion-deploy] Strict artifact verification started")
    print(f"[emotion-deploy] root={ROOT}")

    missing_required = []
    for path in REQUIRED_ARTIFACTS:
        exists = path.exists()
        print(f"[emotion-deploy] required {'OK  ' if exists else 'MISS'} {path}")
        if not exists:
            missing_required.append(path)

    if missing_required:
        print("[emotion-deploy] ERROR: required artifacts are missing.")
        return 1

    summary = _load_metadata_summary()
    if not summary:
        print("[emotion-deploy] ERROR: missing or invalid emotion metadata summary")
        return 1

    print(
        "[emotion-deploy] metadata summary "
        f"model_type={summary.get('model_type')} accuracy={summary.get('accuracy')} "
        f"timestamp={summary.get('timestamp')} classes={summary.get('class_names')}"
    )

    try:
        accuracy = float(summary.get("accuracy"))
    except (TypeError, ValueError):
        print("[emotion-deploy] ERROR: model accuracy is missing/non-numeric in metadata")
        return 1

    if accuracy < MIN_EXPECTED_TEST_ACCURACY:
        print(
            "[emotion-deploy] ERROR: emotion model test accuracy below strict threshold "
            f"({accuracy:.4f} < {MIN_EXPECTED_TEST_ACCURACY:.4f})"
        )
        return 1

    print("[emotion-deploy] Strict verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())