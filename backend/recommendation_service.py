from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
from sqlalchemy import func

from .adaptive_engine import compute_adaptive_signals
from .models import AnswerLog, Question, UserProgress, db
from .validation import sanitize_string

DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]
ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "recommender"
LATEST_MANIFEST_PATH = ARTIFACT_DIR / "latest_manifest.json"
MAX_CANDIDATES = 450
MIN_RESULTS = 3


@dataclass
class RecommendationCandidate:
    question: Question
    expected_gain: float
    predicted_success: float
    novelty_bonus: float
    difficulty_alignment: float
    popularity_bonus: float
    score: float


@dataclass
class RecommendationBundle:
    items: list[RecommendationCandidate]
    metadata: dict


class RecommendationArtifact:
    def __init__(self, manifest: dict, weights_path: Path) -> None:
        data = np.load(weights_path, allow_pickle=False)
        self.artifact_id: str = manifest.get("artifact_id", weights_path.stem)
        self.metrics: dict = manifest.get("metrics", {})
        self.created_at: str = manifest.get("created_at", "")

        self.user_ids = data["user_ids"].astype(np.int64)
        self.question_ids = data["question_ids"].astype(np.int64)
        self.user_factors = data["user_factors"].astype(np.float32)
        self.question_factors = data["question_factors"].astype(np.float32)
        self.user_bias = data["user_bias"].astype(np.float32)
        self.question_bias = data["question_bias"].astype(np.float32)
        self.global_bias = float(data["global_bias"][0])
        self.question_popularity = data["question_popularity"].astype(np.float32) if "question_popularity" in data.files else None
        self.question_success = data["question_success"].astype(np.float32) if "question_success" in data.files else None

        self.user_index = {int(uid): idx for idx, uid in enumerate(self.user_ids)}
        self.question_index = {int(qid): idx for idx, qid in enumerate(self.question_ids)}

        self.user_backoff = self._mean_vector(self.user_factors)
        self.question_backoff = self._mean_vector(self.question_factors)
        self.popularity_backoff = float(np.mean(self.question_popularity)) if self.question_popularity is not None else 0.5
        self.success_backoff = float(np.mean(self.question_success)) if self.question_success is not None else 0.5

    @staticmethod
    def _mean_vector(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            width = matrix.shape[1] if matrix.ndim == 2 and matrix.shape[1] else 1
            return np.zeros((width,), dtype=np.float32)
        return np.mean(matrix, axis=0)

    def predict(self, user_id: int, question_id: int) -> float:
        user_idx = self.user_index.get(int(user_id))
        question_idx = self.question_index.get(int(question_id))

        user_vec = self.user_factors[user_idx] if user_idx is not None else self.user_backoff
        question_vec = self.question_factors[question_idx] if question_idx is not None else self.question_backoff

        bias = self.global_bias
        if user_idx is not None:
            bias += float(self.user_bias[user_idx])
        if question_idx is not None:
            bias += float(self.question_bias[question_idx])

        score = bias + float(np.dot(user_vec, question_vec))
        return _sigmoid(score)

    def popularity(self, question_id: int) -> float:
        if self.question_popularity is None:
            return self.popularity_backoff
        idx = self.question_index.get(int(question_id))
        if idx is None:
            return self.popularity_backoff
        return float(self.question_popularity[idx])

    def success_rate(self, question_id: int) -> float:
        if self.question_success is None:
            return self.success_backoff
        idx = self.question_index.get(int(question_id))
        if idx is None:
            return self.success_backoff
        return float(self.question_success[idx])


@dataclass
class _RecommendationRuntime:
    artifact: RecommendationArtifact
    manifest: dict


def recommend_questions_for_user(
    user_id: int,
    *,
    subject: str,
    grade: str | None,
    topic: str | None,
    count: int = 10,
    difficulty_hint: str | None = None,
    exclude_answered: bool = True,
) -> RecommendationBundle:
    runtime = _load_runtime()
    if runtime is None:
        return RecommendationBundle(items=[], metadata={"reason": "artifact_missing"})

    artifact = runtime.artifact
    started = perf_counter()

    normalized_subject = _normalize_subject(subject)
    normalized_grade = _normalize_grade(grade)
    normalized_topic = _normalize_topic_token(topic)

    candidate_query = Question.query
    if normalized_subject:
        candidate_query = candidate_query.filter(func.lower(Question.subject) == normalized_subject)
    if normalized_grade:
        candidate_query = candidate_query.filter(Question.grade == normalized_grade)
    if normalized_topic:
        candidate_query = candidate_query.filter(func.lower(Question.syllabus_topic) == normalized_topic)

    candidate_query = candidate_query.order_by(func.random()).limit(MAX_CANDIDATES * 2)
    rows: list[Question] = candidate_query.all()

    answered_ids: set[int] = set()
    if exclude_answered:
        answered_ids = {
            row.question_id
            for row in db.session.query(AnswerLog.question_id).filter(AnswerLog.user_id == user_id)
        }

    candidates: list[Question] = []
    for row in rows:
        if exclude_answered and row.id in answered_ids:
            continue
        candidates.append(row)
        if len(candidates) >= MAX_CANDIDATES:
            break

    if not candidates:
        return RecommendationBundle(
            items=[],
            metadata={
                "artifact_id": artifact.artifact_id,
                "reason": "no_candidates",
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            },
        )

    subject_accuracy = _subject_accuracy(user_id, normalized_subject)
    adaptive_hint = None
    if normalized_subject:
        adaptive_hint = compute_adaptive_signals(user_id, normalized_subject, normalized_topic or topic or None)
    target_difficulty = difficulty_hint or (adaptive_hint.get("recommended_difficulty") if adaptive_hint else None)
    target_rank = _difficulty_rank(target_difficulty)

    items: list[RecommendationCandidate] = []
    mastery_gap = 1.0 - subject_accuracy

    for question in candidates:
        predicted = artifact.predict(user_id, question.id)
        novelty = 1.0 if question.id not in answered_ids else 0.35
        diff_align = _alignment_score(question.difficulty, target_rank)
        popularity = artifact.popularity(question.id)
        success = artifact.success_rate(question.id)

        expected_gain = (predicted * (0.55 + 0.35 * diff_align)) + (mastery_gap * 0.2 * diff_align)
        expected_gain = _clamp(expected_gain, 0.01, 1.25)
        final_score = expected_gain + 0.08 * novelty + 0.07 * popularity + 0.04 * (1.0 - success)

        items.append(
            RecommendationCandidate(
                question=question,
                expected_gain=expected_gain,
                predicted_success=predicted,
                novelty_bonus=novelty,
                difficulty_alignment=diff_align,
                popularity_bonus=popularity,
                score=final_score,
            )
        )

    items.sort(key=lambda item: item.score, reverse=True)
    limit = min(len(items), max(count, MIN_RESULTS))
    selected = items[:limit]

    metadata = {
        "artifact_id": artifact.artifact_id,
        "trained_at": artifact.created_at,
        "metrics": artifact.metrics,
        "took_ms": round((perf_counter() - started) * 1000, 2),
        "candidate_pool": len(candidates),
    }

    return RecommendationBundle(items=selected, metadata=metadata)


def get_recommender_metadata() -> Optional[dict]:
    runtime = _load_runtime()
    if runtime is None:
        return None
    return {
        "artifact_id": runtime.artifact.artifact_id,
        "trained_at": runtime.artifact.created_at,
        "metrics": runtime.artifact.metrics,
    }


@lru_cache(maxsize=1)
def _load_runtime() -> Optional[_RecommendationRuntime]:
    if not LATEST_MANIFEST_PATH.exists():
        return None
    manifest = json.loads(LATEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    weights_name = manifest.get("weights_path")
    if not weights_name:
        return None
    weights_path = ARTIFACT_DIR / weights_name
    if not weights_path.exists():
        return None
    artifact = RecommendationArtifact(manifest, weights_path)
    return _RecommendationRuntime(artifact=artifact, manifest=manifest)


def _alignment_score(question_difficulty: str | None, target_rank: int) -> float:
    q_rank = _difficulty_rank(question_difficulty)
    span = max(1, len(DIFFICULTY_ORDER) - 1)
    distance = abs(q_rank - target_rank) / span
    return _clamp(1.0 - distance, 0.2, 1.0)


def _subject_accuracy(user_id: int, subject: str | None) -> float:
    if not subject:
        return 0.4
    progress = UserProgress.query.filter_by(user_id=user_id, subject=subject).first()
    if not progress or not progress.total_questions:
        return 0.4
    accuracy = progress.correct_answers / max(1, progress.total_questions)
    return _clamp(accuracy, 0.0, 1.0)


def _normalize_topic_token(topic_value: str | None) -> str:
    if not topic_value:
        return ""
    normalized = sanitize_string(topic_value).strip().lower()
    if not normalized:
        return ""
    normalized = normalized.replace("-", " ")
    normalized = "_".join(normalized.split())
    return normalized


def _normalize_subject(subject: str | None) -> str:
    if not subject:
        return ""
    return sanitize_string(subject).strip().lower()


def _normalize_grade(grade: str | None) -> str:
    if not grade:
        return ""
    return sanitize_string(grade).strip().lower()


def _difficulty_rank(label: str | None) -> int:
    if not label:
        return 1
    label = sanitize_string(label).strip().lower()
    if label in DIFFICULTY_ORDER:
        return DIFFICULTY_ORDER.index(label)
    return 1


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
