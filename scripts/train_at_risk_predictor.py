from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.at_risk_features import FeatureConfig, build_feature_vector, label_from_horizon, difficulty_to_numeric, emotion_to_bucket


DATA_ROOT = ROOT / "backend" / "data" / "ml" / "interaction_datasets"
ARTIFACT_DIR = ROOT / "backend" / "models" / "at_risk_predictor"
LATEST_MANIFEST_PATH = ARTIFACT_DIR / "latest_manifest.json"


def _stable_student_id(student_key: str, provided_student_id: Any) -> int:
    if provided_student_id is not None:
        try:
            return int(provided_student_id)
        except Exception:
            pass
    digest = hashlib.sha1(str(student_key).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train at-risk predictor (Task 7)")
    parser.add_argument("--window-attempts", type=int, default=50)
    parser.add_argument("--horizon-attempts", type=int, default=10)
    parser.add_argument("--at-risk-threshold", type=float, default=0.55)
    parser.add_argument("--min-events-per-student", type=int, default=80)
    parser.add_argument("--stride", type=int, default=5, help="Sample every N steps within each student")
    parser.add_argument("--max-snapshots-per-student", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--background-size", type=int, default=250, help="SHAP background sample size")
    return parser.parse_args()


def _parse_iso_ts(value: str | None) -> float:
    if not value:
        return datetime.now(timezone.utc).timestamp()
    try:
        iso = str(value)
        if iso.endswith("Z"):
            iso = iso.replace("Z", "+00:00")
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


def _load_latest_version() -> str:
    latest_path = DATA_ROOT / "LATEST_VERSION"
    if not latest_path.exists():
        raise SystemExit("No interaction dataset latest version file found.")
    return latest_path.read_text(encoding="utf-8").strip()


def _iter_events(events_path: Path) -> Iterable[dict]:
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_events_by_student(version: str) -> Dict[int, List[dict]]:
    events_path = DATA_ROOT / version / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"Missing events.jsonl at: {events_path}")

    by_student: Dict[int, List[dict]] = {}
    for payload in _iter_events(events_path):
        student_key = payload.get("student_key") or "unknown"
        student_id = _stable_student_id(student_key, payload.get("student_id"))

        split = payload.get("split") or "train"
        answered_at = _parse_iso_ts(payload.get("answered_at"))

        # Keep only fields needed for feature engineering + labeling.
        record = {
            "student_id": student_id,
            "split": split,
            "answered_at": answered_at,
            "is_correct": bool(payload.get("is_correct", False)),
            "time_spent_sec": float(payload.get("time_spent_sec") or payload.get("time_spent") or 0.0),
            "difficulty": payload.get("difficulty"),
            "emotion": payload.get("emotion"),
        }

        by_student.setdefault(student_id, []).append(record)

    # Sort each student's history.
    for sid in list(by_student.keys()):
        by_student[sid].sort(key=lambda r: r["answered_at"])
    return by_student


def _build_snapshots_for_student(
    history: List[dict],
    *,
    config: FeatureConfig,
    seed: int,
    stride: int,
    max_snapshots: int,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    if len(history) < config.window_attempts + config.horizon_attempts + 1:
        return [], [], []

    rng = random.Random(seed + history[0]["student_id"])

    correctness_all = np.asarray([1 if r["is_correct"] else 0 for r in history], dtype=np.int32)
    time_all = np.asarray([float(r["time_spent_sec"]) for r in history], dtype=np.float32)
    difficulty_all = np.asarray([difficulty_to_numeric(r.get("difficulty")) for r in history], dtype=np.float32)
    emotion_all = np.asarray([emotion_to_bucket(r.get("emotion")) for r in history], dtype=object)
    split = history[0]["split"]

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    splits: List[str] = []

    # Snapshot index is the "history end" inclusive.
    # We need: [i-window+1 ... i] as history and [i+1 ... i+horizon] as future horizon.
    i_start = config.window_attempts - 1
    i_end = len(history) - config.horizon_attempts - 1
    if i_end < i_start:
        return [], [], []

    sampled_indices = list(range(i_start, i_end + 1, max(1, stride)))
    rng.shuffle(sampled_indices)
    sampled_indices = sampled_indices[: max_snapshots]

    for i in sampled_indices:
        hist_slice = slice(max(0, i - config.window_attempts + 1), i + 1)
        fut_slice = slice(i + 1, i + 1 + config.horizon_attempts)

        X_feat, _ = build_feature_vector(
            correctness=correctness_all[hist_slice],
            time_spent=time_all[hist_slice],
            difficulty_numeric=difficulty_all[hist_slice],
            emotion_buckets=emotion_all[hist_slice],
            config=config,
        )

        y = label_from_horizon(correctness_future=correctness_all[fut_slice], threshold=config.at_risk_accuracy_threshold)

        X_rows.append(X_feat)
        y_rows.append(int(y))
        splits.append(split)

    return X_rows, y_rows, splits


def _stack_xy(rows: List[np.ndarray], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(rows, axis=0).astype(np.float32, copy=False) if rows else np.zeros((0, 17), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32) if labels else np.zeros((0,), dtype=np.int32)
    return X, y


def _compute_class_weight(y: np.ndarray) -> float:
    if y.size == 0:
        return 1.0
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return neg / max(1.0, pos)


def _train_lr(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    scaler = StandardScaler(with_mean=True, with_std=True)
    model = LogisticRegression(
        max_iter=500,
        solver="liblinear",
        class_weight="balanced",
        random_state=1,
    )
    return Pipeline([("scaler", scaler), ("model", model)])


def _train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=420,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=1,
        n_jobs=-1,
    )


def _train_mlp(X_train: np.ndarray, y_train: np.ndarray, *, seed: int) -> Pipeline:
    # MLP can emit convergence warnings; suppress during training.
    scaler = StandardScaler(with_mean=True, with_std=True)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )
    pipe = Pipeline([("scaler", scaler), ("model", clf)])
    # Use explicit class weights via sample weights (no external deps).
    pos_weight = _compute_class_weight(y_train)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipe


def _roc_auc_safe(model, X: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = FeatureConfig(
        window_attempts=args.window_attempts,
        horizon_attempts=args.horizon_attempts,
        at_risk_accuracy_threshold=args.at_risk_threshold,
    )

    version = _load_latest_version()
    print(f"[AtRisk] Using interaction dataset version: {version}")
    by_student = _load_events_by_student(version)

    X_train_rows: List[np.ndarray] = []
    y_train_rows: List[int] = []
    X_val_rows: List[np.ndarray] = []
    y_val_rows: List[int] = []
    X_test_rows: List[np.ndarray] = []
    y_test_rows: List[int] = []

    for sid, history in by_student.items():
        if len(history) < args.min_events_per_student:
            continue

        X_rows, y_rows, splits = _build_snapshots_for_student(
            history,
            config=config,
            seed=args.seed,
            stride=args.stride,
            max_snapshots=args.max_snapshots_per_student,
        )
        if not X_rows:
            continue

        # Attach to split based on the student's split.
        for X_feat, y, sp in zip(X_rows, y_rows, splits):
            if sp == "train":
                X_train_rows.append(X_feat)
                y_train_rows.append(y)
            elif sp == "val":
                X_val_rows.append(X_feat)
                y_val_rows.append(y)
            else:
                X_test_rows.append(X_feat)
                y_test_rows.append(y)

    X_train, y_train = _stack_xy(X_train_rows, y_train_rows)
    X_val, y_val = _stack_xy(X_val_rows, y_val_rows)
    X_test, y_test = _stack_xy(X_test_rows, y_test_rows)

    if X_train.shape[0] < 500 or X_test.shape[0] < 200:
        raise SystemExit(
            f"Not enough snapshot samples for stable training: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}"
        )

    print(f"[AtRisk] Snapshots => train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")
    print(f"[AtRisk] Positive ratio => train={float(np.mean(y_train)):.3f} val={float(np.mean(y_val)):.3f} test={float(np.mean(y_test)):.3f}")

    # Train baseline models.
    models: Dict[str, Any] = {}
    models["logreg"] = _train_lr(X_train, y_train)
    models["logreg"].fit(X_train, y_train)

    models["rf"] = _train_rf(X_train, y_train)
    models["rf"].fit(X_train, y_train)

    # MLP is only for comparison per spec (deployment uses LR/RF for SHAP stability).
    models["mlp"] = _train_mlp(X_train, y_train, seed=args.seed)

    aucs: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        aucs[name] = {
            "val_auc": _roc_auc_safe(m, X_val, y_val),
            "test_auc": _roc_auc_safe(m, X_test, y_test),
        }
        print(f"[AtRisk][AUC] {name}: val={aucs[name]['val_auc']:.4f} test={aucs[name]['test_auc']:.4f}")

    # Select deployment model among LR/RF.
    deploy_candidates = ["logreg", "rf"]
    selected_deploy = max(deploy_candidates, key=lambda n: (aucs[n]["val_auc"] if not math.isnan(aucs[n]["val_auc"]) else -1.0))
    deploy_kind = selected_deploy
    deploy_model = models[selected_deploy]
    print(f"[AtRisk] Deployment model selected: {deploy_kind}")

    # Fit deployment model on train+val for stronger generalization.
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    if deploy_kind == "logreg":
        # Re-train pipeline to include val signals.
        deploy_model = _train_lr(X_trainval, y_trainval)
        deploy_model.fit(X_trainval, y_trainval)
        scaler = deploy_model.named_steps["scaler"]
        model_obj = deploy_model.named_steps["model"]
        explainer_kind = "linear"
    elif deploy_kind == "rf":
        deploy_model = _train_rf(X_trainval, y_trainval)
        deploy_model.fit(X_trainval, y_trainval)
        scaler = None
        model_obj = deploy_model
        explainer_kind = "tree"
    else:
        # Should never happen (we restrict to logreg/rf).
        explainer_kind = "linear"
        scaler = None
        model_obj = deploy_model

    # SHAP background sample (raw feature space).
    rng = np.random.default_rng(args.seed)
    bg_size = min(args.background_size, X_trainval.shape[0])
    background_idx = rng.choice(np.arange(X_trainval.shape[0]), size=bg_size, replace=False)
    background_X_raw = X_trainval[background_idx].astype(np.float32, copy=False)

    # Persist artifact.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_id = f"at_risk_{timestamp}"

    model_bundle_path = ARTIFACT_DIR / f"{artifact_id}.joblib"
    joblib.dump(
        {
            "model_type": deploy_kind,
            "scaler": scaler,
            "model": model_obj,
            "feature_names": [
                "n_attempts_window",
                "correct_rate_window",
                "incorrect_rate_window",
                "avg_time_spent_window",
                "time_std_window",
                "avg_difficulty_window",
                "emotion_confused_rate",
                "emotion_frustrated_rate",
                "emotion_bored_rate",
                "emotion_focused_rate",
                "emotion_happy_rate",
                "emotion_other_rate",
                "emotion_unknown_rate",
                "correctness_trend",
                "max_correct_streak_last20",
                "correct_rate_last10",
                "frustrated_rate_repeat",
            ],
            "explainer_kind": explainer_kind,
            "background_X_raw": background_X_raw,
        },
        model_bundle_path,
    )

    metrics = {
        "roc_auc": {k: {kk: float(vv) for kk, vv in vv.items()} for k, vv in aucs.items()},
        "selected_deploy": deploy_kind,
        "config": asdict(config),
        "counts": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
            "trainval": int(X_trainval.shape[0]),
        },
    }

    manifest = {
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_bundle_path": model_bundle_path.name,
        "feature_config": {
            "window_attempts": config.window_attempts,
            "horizon_attempts": config.horizon_attempts,
            "streak_window": config.streak_window,
            "at_risk_accuracy_threshold": config.at_risk_accuracy_threshold,
        },
        "metrics": metrics,
    }
    LATEST_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[AtRisk] Artifact saved:", model_bundle_path)
    print("[AtRisk] Deployment metrics:", metrics.get("roc_auc", {}))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

