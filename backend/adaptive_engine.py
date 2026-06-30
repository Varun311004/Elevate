"""Adaptive learning engine (BKT + IRT) for runtime question difficulty routing.

This module keeps model logic centralized so routes can remain thin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .models import AnswerLog, EmotionLog, Question


DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]
DIFFICULTY_TO_B = {
    "easy": -1.25,
    "medium": -0.1,
    "hard": 0.95,
    "expert": 1.55,
}


@dataclass
class BKTParams:
    guess: float
    slip: float
    learn: float
    forget: float = 0.01


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _difficulty_rank(level: str | None) -> int:
    level = (level or "medium").lower()
    return DIFFICULTY_ORDER.index(level) if level in DIFFICULTY_ORDER else 1


def _difficulty_from_rank(rank: int) -> str:
    return DIFFICULTY_ORDER[_clamp(int(rank), 0, len(DIFFICULTY_ORDER) - 1)]


def bkt_params_for_difficulty(difficulty: str | None) -> BKTParams:
    d = (difficulty or "medium").lower()
    if d == "easy":
        return BKTParams(guess=0.28, slip=0.08, learn=0.24)
    if d == "hard":
        return BKTParams(guess=0.16, slip=0.18, learn=0.14)
    if d == "expert":
        return BKTParams(guess=0.12, slip=0.24, learn=0.1)
    return BKTParams(guess=0.2, slip=0.12, learn=0.18)


def bkt_update(prior: float, is_correct: bool, params: BKTParams) -> float:
    p_known = _clamp(prior, 0.001, 0.999)

    if is_correct:
        numerator = p_known * (1.0 - params.slip)
        denominator = numerator + (1.0 - p_known) * params.guess
    else:
        numerator = p_known * params.slip
        denominator = numerator + (1.0 - p_known) * (1.0 - params.guess)

    posterior = numerator / denominator if denominator > 0 else p_known

    # Transition: forgetting + learning
    transitioned = posterior * (1.0 - params.forget) + (1.0 - posterior) * params.learn
    return _clamp(transitioned, 0.001, 0.999)


def estimate_irt_theta(responses: Iterable[tuple[bool, str]], iterations: int = 7) -> float:
    """Estimate ability parameter theta via 1PL Newton-Raphson.

    responses: iterable of (is_correct, difficulty)
    """
    rows = list(responses)
    if not rows:
        return 0.0

    theta = 0.0
    for _ in range(iterations):
        grad = 0.0
        hess = 0.0
        for is_correct, difficulty in rows:
            b = DIFFICULTY_TO_B.get((difficulty or "medium").lower(), 0.0)
            p = _sigmoid(theta - b)
            u = 1.0 if is_correct else 0.0
            grad += (u - p)
            hess -= p * (1.0 - p)

        if abs(hess) < 1e-6:
            break
        theta -= grad / hess
        theta = _clamp(theta, -3.0, 3.0)

    return round(theta, 4)


def compute_adaptive_signals(user_id: int, subject: str, concept: str | None = None) -> dict:
    """Compute BKT mastery, IRT ability, and recommended next difficulty."""
    subj = (subject or "").strip()
    concept_key = (concept or "").strip() or None

    answer_query = (
        AnswerLog.query.join(Question, AnswerLog.question_id == Question.id)
        .filter(AnswerLog.user_id == user_id, Question.subject == subj)
        .order_by(AnswerLog.answered_at.asc())
    )
    if concept_key:
        answer_query = answer_query.filter(Question.syllabus_topic == concept_key)

    concept_answers = answer_query.limit(300).all()

    p_known = 0.22
    for answer in concept_answers:
        params = bkt_params_for_difficulty(answer.difficulty_at_time or answer.question.difficulty)
        p_known = bkt_update(p_known, bool(answer.is_correct), params)

    # Use broader subject responses for robust theta.
    theta_answers = (
        AnswerLog.query.join(Question, AnswerLog.question_id == Question.id)
        .filter(AnswerLog.user_id == user_id, Question.subject == subj)
        .order_by(AnswerLog.answered_at.desc())
        .limit(80)
        .all()
    )
    theta = estimate_irt_theta((bool(a.is_correct), a.difficulty_at_time or a.question.difficulty) for a in theta_answers)

    # Emotion-aware modifier from recent context logs.
    recent_emotions = (
        EmotionLog.query.filter(
            EmotionLog.user_id == user_id,
            EmotionLog.context.in_([f"answering_{subj}", "answering"]),
        )
        .order_by(EmotionLog.timestamp.desc())
        .limit(8)
        .all()
    )
    emotion_modifier = 0.0
    for emo in recent_emotions:
        tag = (emo.emotion or "").lower()
        if tag in {"confused", "angry"}:
            emotion_modifier -= 0.17
        elif tag in {"focused", "happy"}:
            emotion_modifier += 0.08
        elif tag == "bored":
            emotion_modifier -= 0.09

    theta_norm = (theta + 3.0) / 6.0  # [0, 1]
    composite = (0.6 * theta_norm) + (0.4 * p_known) + emotion_modifier
    composite = _clamp(composite, 0.0, 1.0)

    recommended_rank = int(round(composite * (len(DIFFICULTY_ORDER) - 1)))

    # Guardrails for early stage users.
    if len(theta_answers) < 5 and recommended_rank > 1:
        recommended_rank = 1
    if p_known < 0.35:
        recommended_rank = min(recommended_rank, 1)

    return {
        "subject": subj,
        "concept": concept_key,
        "bkt_p_known": round(p_known, 4),
        "irt_theta": theta,
        "composite_skill": round(composite, 4),
        "answered_in_concept": len(concept_answers),
        "answered_in_subject": len(theta_answers),
        "recommended_difficulty": _difficulty_from_rank(recommended_rank),
    }


def difficulty_fallback_sequence(target: str | None) -> list[str]:
    """Return nearest-neighbor difficulty fallback order."""
    rank = _difficulty_rank(target)
    offsets = [0, -1, 1, -2, 2, -3, 3]
    sequence = []
    for offset in offsets:
        candidate_rank = rank + offset
        if 0 <= candidate_rank < len(DIFFICULTY_ORDER):
            level = DIFFICULTY_ORDER[candidate_rank]
            if level not in sequence:
                sequence.append(level)
    return sequence
