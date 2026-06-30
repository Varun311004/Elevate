from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from flask import current_app
from sqlalchemy import desc

from .at_risk_features import FeatureConfig, build_feature_vector, difficulty_to_numeric, emotion_to_bucket
from .models import AnswerLog, Question, db


ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "at_risk_predictor"
LATEST_MANIFEST_PATH = ARTIFACT_DIR / "latest_manifest.json"


@dataclass
class AtRiskPrediction:
    student_id: int
    at_risk_probability: float
    shap_top_features: list[dict[str, Any]]


class _AtRiskRuntime:
    def __init__(self, manifest: dict[str, Any], model_bundle_path: Path) -> None:
        bundle = joblib.load(model_bundle_path)

        self.artifact_id: str = str(manifest.get("artifact_id", model_bundle_path.stem))
        self.trained_at: str = str(manifest.get("created_at", ""))
        self.metrics: dict[str, Any] = dict(manifest.get("metrics", {}))

        self.feature_config = FeatureConfig(**manifest.get("feature_config", {}))
        self.feature_names: list[str] = list(bundle.get("feature_names") or [])

        self.model_type: str = str(bundle.get("model_type", "logreg"))
        self.scaler = bundle.get("scaler")  # may be None
        self.model = bundle["model"]
        self.explainer_kind: str = str(bundle.get("explainer_kind", "linear"))

        self.background_X_raw = np.asarray(bundle.get("background_X_raw", []), dtype=np.float32)
        self._background_X_scaled = self._transform_X(self.background_X_raw) if self.background_X_raw.size else np.zeros((1, len(self.feature_names)), dtype=np.float32)

        self._explainer = None

    def _transform_X(self, X_raw: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X_raw.astype(np.float32, copy=False)
        return self.scaler.transform(X_raw.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X_scaled = self._transform_X(X_raw)
        proba = self.model.predict_proba(X_scaled)
        # sklearn classifiers return [:, 1] for positive class.
        return proba[:, 1].astype(np.float32, copy=False)

    def _get_shap_explainer(self):
        if self._explainer is not None:
            return self._explainer

        # SHAP can be noisy; silence warnings during explainer creation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import shap  # local import to keep runtime lighter

        if self.explainer_kind == "linear":
            # feature_perturbation is deprecated in newer SHAP versions;
            # rely on SHAP defaults to avoid FutureWarning noise.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._explainer = shap.LinearExplainer(self.model, self._background_X_scaled)
        elif self.explainer_kind == "tree":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._explainer = shap.TreeExplainer(self.model, model_output="probability")
        else:
            # Fallback: let SHAP auto-detect.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._explainer = shap.Explainer(self.model, self._background_X_scaled)

        return self._explainer

    def explain_top_features(self, X_raw: np.ndarray, top_k: int = 5) -> list[list[dict[str, Any]]]:
        if X_raw.size == 0:
            return []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import shap  # noqa: F401

            explainer = self._get_shap_explainer()
            X_scaled = self._transform_X(X_raw)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_scaled)

            # Some SHAP APIs return list per class for classifiers.
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = np.asarray(shap_values, dtype=np.float32)

            # Pick top-k by abs(shap_value).
            top_features_per_row: list[list[dict[str, Any]]] = []
            for row_idx in range(shap_values.shape[0]):
                row_shap = shap_values[row_idx]
                order = np.argsort(-np.abs(row_shap))[: max(1, top_k)]
                top_list = [
                    {
                        "feature": self.feature_names[int(i)] if i < len(self.feature_names) else f"f{int(i)}",
                        "shap_value": float(row_shap[int(i)]),
                    }
                    for i in order
                ]
                top_features_per_row.append(top_list)
            return top_features_per_row
        except Exception:
            # Never fail the main endpoint because SHAP is unavailable.
            return [[ ] for _ in range(X_raw.shape[0])]


def _load_runtime() -> Optional[_AtRiskRuntime]:
    if not LATEST_MANIFEST_PATH.exists():
        return None
    manifest = json.loads(LATEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    model_rel_path = manifest.get("model_bundle_path")
    if not model_rel_path:
        return None
    model_bundle_path = ARTIFACT_DIR / str(model_rel_path)
    if not model_bundle_path.exists():
        return None
    return _AtRiskRuntime(manifest, model_bundle_path)


@lru_cache(maxsize=1)
def _get_runtime() -> Optional[_AtRiskRuntime]:
    return _load_runtime()


def _fetch_student_history_for_features(student_id: int, *, cutoff: datetime, config: FeatureConfig) -> dict[str, np.ndarray]:
    # Use AnswerLog within cutoff, ordered ascending; then take last window_attempts.
    # We query in descending order to cheaply limit DB load.
    rows = (
        db.session.query(AnswerLog, Question.difficulty)
        .join(Question, AnswerLog.question_id == Question.id)
        .filter(
            AnswerLog.user_id == int(student_id),
            AnswerLog.answered_at >= cutoff,
        )
        .order_by(desc(AnswerLog.answered_at))
        .limit(max(10, config.window_attempts))
        .all()
    )
    # Reverse to chronological order.
    rows = list(reversed(rows))

    correctness = np.asarray([1 if bool(r.is_correct) else 0 for (r, _qd) in rows], dtype=np.int32)
    time_spent = np.asarray([float(r.time_spent or 0) for (r, _qd) in rows], dtype=np.float32)

    difficulties: list[float] = []
    emotions: list[str] = []
    for r, question_difficulty in rows:
        raw_diff = r.difficulty_at_time if r.difficulty_at_time is not None else question_difficulty
        difficulties.append(difficulty_to_numeric(raw_diff))
        emotions.append(emotion_to_bucket(r.emotion_at_time))

    difficulty_numeric = np.asarray(difficulties, dtype=np.float32)
    emotion_buckets = np.asarray(emotions, dtype=object)

    return {
        "correctness": correctness,
        "time_spent": time_spent,
        "difficulty_numeric": difficulty_numeric,
        "emotion_buckets": emotion_buckets,
    }


def get_at_risk_predictions_for_students(
    student_ids: list[int],
    *,
    cutoff: datetime,
    top_k_shap: int = 5,
) -> dict[str, Any]:
    """
    Returns a teacher-friendly payload:
    - at_risk_students: one entry per student_id
    - meta: artifact/metrics for debugging and transparency
    """
    runtime = _get_runtime()
    if runtime is None:
        return {"at_risk_students": [], "meta": {"reason": "artifact_missing"}}

    if not student_ids:
        return {"at_risk_students": [], "meta": {"reason": "no_students"}}

    # Build X_raw in the same order as student_ids.
    X_raw_list: list[np.ndarray] = []
    for sid in student_ids:
        hist = _fetch_student_history_for_features(int(sid), cutoff=cutoff, config=runtime.feature_config)
        feats, _ = build_feature_vector(
            correctness=hist["correctness"],
            time_spent=hist["time_spent"],
            difficulty_numeric=hist["difficulty_numeric"],
            emotion_buckets=hist["emotion_buckets"],
            config=runtime.feature_config,
        )
        X_raw_list.append(feats)

    X_raw = np.stack(X_raw_list, axis=0).astype(np.float32, copy=False)

    proba = runtime.predict_proba(X_raw)
    safe_top_k = max(0, int(top_k_shap or 0))
    if safe_top_k > 0:
        shap_top_features = runtime.explain_top_features(X_raw, top_k=safe_top_k)
    else:
        shap_top_features = [[] for _ in range(X_raw.shape[0])]

    at_risk_students: list[dict[str, Any]] = []
    for idx, sid in enumerate(student_ids):
        at_risk_students.append(
            {
                "student_id": int(sid),
                "at_risk_probability": float(proba[idx]),
                "explanation": {
                    "top_features": shap_top_features[idx] if idx < len(shap_top_features) else [],
                },
            }
        )

    return {
        "at_risk_students": at_risk_students,
        "meta": {
            "artifact_id": runtime.artifact_id,
            "trained_at": runtime.trained_at,
            "metrics": runtime.metrics,
        },
    }

