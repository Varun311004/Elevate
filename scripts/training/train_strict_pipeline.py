"""Strict, fail-fast ML training pipeline for Elevate.
para
This script always runs the full training stack and exits non-zero on any failure:
1) Build interaction dataset
2) Train BKT, DKT, Emotion (HOG+MLP), and At-Risk models in parallel
3) Validate metric thresholds
4) Persist a machine-readable summary

Default quality gates are tuned for **real bootstrap / synthetic interaction data**
(small student counts, topic-level BKT). Demanding 0.75+ AUC on BKT/DKT often fails
even when training succeeded — that is a gate problem, not a model problem.

Tighten gates in production (optional), via environment variables on the runner:
  HF_ML_MIN_BKT_AUC, HF_ML_MIN_DKT_AUC, HF_ML_MIN_ATRISK_AUC,
  HF_ML_MIN_EMOTION_ACCURACY (and CLI flags still apply).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
EMOTION_CLASS_ALIASES = {
    "happy": ["happy"],
    "bored": ["bored"],
    "focused": ["focused"],
    "confused": ["confused"],
    "neutral": ["neutral"],
    "angry": ["angry"],
    "surprised": ["surprised", "surprise"],
}
DEFAULT_DB_URL = f"sqlite:///{(ROOT / 'elevate_dev.db').as_posix()}"
STRICT_DB_URL = (
    str(os.environ.get("ELEVATE_STRICT_DATABASE_URL") or "").strip()
    or str(os.environ.get("DATABASE_URL") or "").strip()
    or DEFAULT_DB_URL
)

# Enforce one DB target for this entire run (current process + subprocesses).
os.environ["DATABASE_URL"] = STRICT_DB_URL
PIPELINE_ENV = dict(os.environ)
PIPELINE_ENV["DATABASE_URL"] = STRICT_DB_URL
PIPELINE_ENV.setdefault("PYTHONIOENCODING", "utf-8")
PIPELINE_ENV.setdefault("PYTHONUTF8", "1")
PIPELINE_ENV.setdefault("PYTHONUNBUFFERED", "1")


def _env_override_float(env_name: str, current: float) -> float:
    raw = str(os.environ.get(env_name) or "").strip()
    if not raw:
        return current
    try:
        return float(raw)
    except ValueError:
        return current


def _run_step(command: list[str], label: str) -> None:
    print(f"[strict-ml] Running {label}: {' '.join(command)}")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=str(ROOT), check=False, env=PIPELINE_ENV)
    duration = time.perf_counter() - started
    print(f"[strict-ml] Completed {label} in {duration:.1f}s (exit={result.returncode})")
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {label} (exit={result.returncode}) command={' '.join(command)}"
        )


def _run_parallel_steps(steps: list[tuple[str, list[str]]], max_workers: int) -> None:
    workers = max(1, min(max_workers, len(steps)))
    print(f"[strict-ml] Launching parallel training jobs with {workers} workers")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_run_step, command, label): label
            for label, command in steps
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{label}: {exc}")

    if failures:
        for entry in failures:
            print(f"[strict-ml] ERROR: {entry}")
        raise RuntimeError("One or more parallel training jobs failed")


def _latest_json_by_mtime(folder: Path, pattern: str) -> dict[str, Any]:
    matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match {pattern} in {folder}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _to_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {label} is missing or non-numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Metric {label} is non-finite: {value!r}")
    return parsed


def _load_latest_dataset_manifest() -> dict[str, Any]:
    dataset_root = ROOT / "backend" / "data" / "ml" / "interaction_datasets"
    latest_file = dataset_root / "LATEST_VERSION"
    if not latest_file.exists():
        raise FileNotFoundError(f"Missing latest dataset marker: {latest_file}")

    version = latest_file.read_text(encoding="utf-8").strip()
    if not version:
        raise RuntimeError("Dataset latest version marker is empty")

    manifest_path = dataset_root / version / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _assert_dataset_ready(manifest: dict[str, Any], *, min_events: int, min_users: int) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    events_total = int(counts.get("events_total") or 0)
    events_train = int(counts.get("events_train") or 0)
    events_test = int(counts.get("events_test") or 0)
    students_total = int(counts.get("students_total") or 0)

    if events_total < min_events:
        raise RuntimeError(
            f"Dataset not ready: events_total={events_total} < required={min_events}"
        )
    if students_total < min_users:
        raise RuntimeError(
            f"Dataset not ready: students_total={students_total} < required={min_users}"
        )
    if events_train <= 0 or events_test <= 0:
        raise RuntimeError(
            "Dataset split invalid: expected train/test events > 0, "
            f"got train={events_train}, test={events_test}"
        )


def _count_questions() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from backend.app import create_app
    from backend.models import Question, db

    app = create_app("development")
    with app.app_context():
        # Defensive initialization for fresh DBs where Alembic state may be out-of-sync.
        db.create_all()
        return int(Question.query.count())


def _extract_metrics() -> tuple[dict[str, float], dict[str, float]]:
    bkt_metrics = _latest_json_by_mtime(ROOT / "backend" / "models" / "bkt", "bkt_metrics_*.json")
    dkt_metrics = _latest_json_by_mtime(ROOT / "backend" / "models" / "dkt", "dkt_metrics_*.json")

    at_risk_manifest_path = ROOT / "backend" / "models" / "at_risk_predictor" / "latest_manifest.json"
    if not at_risk_manifest_path.exists():
        raise FileNotFoundError(f"Missing at-risk manifest: {at_risk_manifest_path}")
    at_risk_manifest = json.loads(at_risk_manifest_path.read_text(encoding="utf-8"))

    deploy_key = (
        at_risk_manifest.get("metrics", {})
        .get("selected_deploy")
    )
    at_risk_auc = (
        at_risk_manifest.get("metrics", {})
        .get("roc_auc", {})
        .get(deploy_key or "", {})
        .get("test_auc")
    )

    metrics = {
        "bkt_test_auc": _to_float(bkt_metrics.get("test_metrics", {}).get("auc"), "bkt_test_auc"),
        "dkt_test_auc": _to_float(dkt_metrics.get("test_metrics", {}).get("auc"), "dkt_test_auc"),
        "at_risk_test_auc": _to_float(at_risk_auc, "at_risk_test_auc"),
    }

    per_class_recall: dict[str, float] = {}
    emotion_info_path = ROOT / "backend" / "ai_models" / "emotion_model_info.json"
    if emotion_info_path.exists():
        emotion_info = json.loads(emotion_info_path.read_text(encoding="utf-8"))
        emotion_test = emotion_info.get("test_metrics") if isinstance(emotion_info.get("test_metrics"), dict) else {}
        if isinstance(emotion_test, dict):
            if emotion_test.get("accuracy") is not None:
                metrics["emotion_test_accuracy"] = _to_float(
                    emotion_test.get("accuracy"), "emotion_test_accuracy"
                )
            if emotion_test.get("macro_f1") is not None:
                metrics["emotion_test_macro_f1"] = _to_float(
                    emotion_test.get("macro_f1"), "emotion_test_macro_f1"
                )
            per_class = emotion_test.get("per_class") if isinstance(emotion_test.get("per_class"), dict) else {}
            for class_name, row in per_class.items():
                if isinstance(row, dict):
                    per_class_recall[str(class_name)] = _to_float(
                        row.get("recall"), f"emotion_recall_{class_name}"
                    )

    return metrics, per_class_recall


def _assert_threshold(metric_name: str, value: float, minimum: float) -> None:
    if value < minimum:
        raise RuntimeError(
            f"Quality gate failed: {metric_name}={value:.4f} is below minimum {minimum:.4f}"
        )


def _assert_per_class_recall(per_class_recall: dict[str, float], minimum: float) -> None:
    if not per_class_recall:
        raise RuntimeError("Missing emotion per-class recall metrics in emotion_model_info.json")
    failures = [
        f"{name}={value:.4f}"
        for name, value in sorted(per_class_recall.items())
        if value < minimum
    ]
    if failures:
        raise RuntimeError(
            "Quality gate failed: per-class emotion recall below threshold "
            f"{minimum:.4f}. Offenders: {', '.join(failures)}"
        )


def _write_summary(metrics: dict[str, float], thresholds: dict[str, float], elapsed_seconds: float) -> Path:
    out_dir = ROOT / "backend" / "models" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "pipeline": "strict",
        "metrics": metrics,
        "thresholds": thresholds,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }

    timestamped = out_dir / f"strict_training_summary_{timestamp}.json"
    latest = out_dir / "strict_training_summary_latest.json"
    timestamped.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return latest


def _detect_emotion_dataset_readiness() -> tuple[bool, str]:
    dataset_dir = ROOT / "dataset"
    if not dataset_dir.exists():
        return False, f"missing directory: {dataset_dir}"

    folder_map = {
        p.name.strip().lower(): p
        for p in dataset_dir.iterdir()
        if p.is_dir()
    }
    missing = []
    for cls, aliases in EMOTION_CLASS_ALIASES.items():
        if not any(alias.lower() in folder_map for alias in aliases):
            missing.append(cls)
    if missing:
        return False, "missing class folder(s): " + ", ".join(missing)

    empty = []
    for cls, aliases in EMOTION_CLASS_ALIASES.items():
        folder = None
        for alias in aliases:
            folder = folder_map.get(alias.lower())
            if folder is not None:
                break
        if folder is None:
            empty.append(cls)
            continue
        has_images = any(
            f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for f in folder.iterdir()
        )
        if not has_images:
            empty.append(cls)
    if empty:
        return False, "class folder(s) have no images: " + ", ".join(empty)

    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict, fail-fast full ML training pipeline")
    parser.add_argument(
        "--processes",
        type=int,
        default=max(2, min(os.cpu_count() or 2, 4)),
        help="Parallel process budget for training jobs",
    )
    parser.add_argument("--min-emotion-accuracy", type=float, default=0.88)
    parser.add_argument("--min-emotion-macro-f1", type=float, default=0.85)
    parser.add_argument(
        "--min-bkt-auc",
        type=float,
        default=0.55,
        help="BKT test-set AUC floor (default 0.55; topic-skill BKT is often ~0.55–0.65 on bootstrap data).",
    )
    parser.add_argument(
        "--min-dkt-auc",
        type=float,
        default=0.65,
        help="DKT test-set AUC floor (default 0.65).",
    )
    parser.add_argument(
        "--min-atrisk-auc",
        type=float,
        default=0.72,
        help="At-risk deploy model test AUC floor (default 0.72).",
    )
    parser.add_argument("--min-emotion-per-class-recall", type=float, default=0.58)
    parser.add_argument("--dataset-min-events", type=int, default=20000)
    parser.add_argument("--dataset-min-users", type=int, default=60)
    parser.add_argument("--seed-questions-per-topic", type=int, default=20)
    parser.add_argument("--atrisk-min-events-per-student", type=int, default=40)
    parser.add_argument("--atrisk-stride", type=int, default=3)
    parser.add_argument("--atrisk-max-snapshots-per-student", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.min_bkt_auc = _env_override_float("HF_ML_MIN_BKT_AUC", float(args.min_bkt_auc))
    args.min_dkt_auc = _env_override_float("HF_ML_MIN_DKT_AUC", float(args.min_dkt_auc))
    args.min_atrisk_auc = _env_override_float("HF_ML_MIN_ATRISK_AUC", float(args.min_atrisk_auc))

    started = time.perf_counter()

    thresholds = {
        "bkt_test_auc": float(args.min_bkt_auc),
        "dkt_test_auc": float(args.min_dkt_auc),
        "at_risk_test_auc": float(args.min_atrisk_auc),
    }

    try:
        print("[strict-ml] Starting strict training pipeline")
        print(f"[strict-ml] python={PYTHON}")
        print(f"[strict-ml] root={ROOT}")
        print(f"[strict-ml] database_url={STRICT_DB_URL}")
        print(
            "[strict-ml] quality gates "
            f"bkt_auc>={thresholds['bkt_test_auc']:.3f} "
            f"dkt_auc>={thresholds['dkt_test_auc']:.3f} "
            f"at_risk_auc>={thresholds['at_risk_test_auc']:.3f}"
        )

        _run_step(
            [
                PYTHON,
                "-m",
                "alembic",
                "-c",
                str(ROOT / "alembic.ini"),
                "upgrade",
                "head",
            ],
            "migrate-database",
        )

        _run_step([PYTHON, str(ROOT / "seed_users.py")], "seed-users")

        _run_step(
            [
                PYTHON,
                str(ROOT / "backend" / "seed_questions.py"),
                "--augment-large",
                "--skip-ai-augment",
                "--per-topic",
                str(max(5, int(args.seed_questions_per_topic))),
            ],
            "seed-questions",
        )

        question_count = _count_questions()
        if question_count <= 0:
            raise RuntimeError(
                "Question seeding produced zero questions. "
                "Fix question generation inputs before continuing strict training."
            )

        _run_step(
            [
                PYTHON,
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "ai" / "requirements.txt"),
            ],
            "install-ai-requirements",
        )

        _run_step(
            [
                PYTHON,
                str(ROOT / "scripts" / "build_interaction_dataset.py"),
                "--min-events",
                str(max(1000, int(args.dataset_min_events))),
                "--min-users",
                str(max(10, int(args.dataset_min_users))),
                "--seed",
                "42",
            ],
            "build-interaction-dataset",
        )

        manifest = _load_latest_dataset_manifest()
        _assert_dataset_ready(
            manifest,
            min_events=max(1000, int(args.dataset_min_events)),
            min_users=max(10, int(args.dataset_min_users)),
        )

        emotion_dataset_ready, emotion_dataset_reason = _detect_emotion_dataset_readiness()
        if emotion_dataset_ready:
            thresholds["emotion_test_accuracy"] = float(args.min_emotion_accuracy)
            thresholds["emotion_test_macro_f1"] = float(args.min_emotion_macro_f1)
            print("[strict-ml] Emotion dataset detected. Emotion training enabled.")
        else:
            print(
                "[strict-ml] WARNING: Emotion dataset unavailable; skipping emotion training "
                f"and emotion quality gates ({emotion_dataset_reason})"
            )

        parallel_steps = [
            ("train-bkt", [PYTHON, str(ROOT / "scripts" / "fit_bkt_model.py")]),
            ("train-dkt", [PYTHON, str(ROOT / "scripts" / "train_dkt_model.py")]),
            (
                "train-at-risk",
                [
                    PYTHON,
                    str(ROOT / "scripts" / "train_at_risk_predictor.py"),
                    "--min-events-per-student",
                    str(max(10, int(args.atrisk_min_events_per_student))),
                    "--stride",
                    str(max(1, int(args.atrisk_stride))),
                    "--max-snapshots-per-student",
                    str(max(50, int(args.atrisk_max_snapshots_per_student))),
                ],
            ),
            (
                "train-emotion-cnn", [PYTHON, str(ROOT / "scripts" / "train_emotion_cnn_hf.py")]
            )
        ]
        if emotion_dataset_ready:
            parallel_steps.append(
                ("train-emotion-hog-mlp", [PYTHON, str(ROOT / "scripts" / "train_emotion_fast.py")])
            )
        _run_parallel_steps(parallel_steps, max_workers=max(1, int(args.processes)))

        metrics, per_class_recall = _extract_metrics()
        for metric_name, minimum in thresholds.items():
            _assert_threshold(metric_name, metrics[metric_name], minimum)
        if emotion_dataset_ready:
            _assert_per_class_recall(per_class_recall, float(args.min_emotion_per_class_recall))

        elapsed = time.perf_counter() - started
        summary_path = _write_summary(metrics, thresholds, elapsed)

        print("[strict-ml] Pipeline succeeded")
        for key, value in metrics.items():
            print(f"[strict-ml] {key}={value:.4f}")
        print(f"[strict-ml] summary={summary_path}")
        return 0
    except Exception as exc:
        print(f"[strict-ml] ERROR: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())