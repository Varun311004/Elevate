from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


DIFFICULTY_TO_NUMERIC = {
    "easy": -1.0,
    "medium": 0.0,
    "hard": 1.0,
    "expert": 1.5,
}

EMOTIONS = ["confused", "frustrated", "bored", "focused", "happy"]


@dataclass(frozen=True)
class FeatureConfig:
    window_attempts: int = 50
    horizon_attempts: int = 10
    streak_window: int = 20

    # If the student's future correctness-rate in the horizon window
    # drops below this threshold, we label them "at-risk".
    at_risk_accuracy_threshold: float = 0.55


def difficulty_to_numeric(difficulty: str | None) -> float:
    if not difficulty:
        return 0.0
    return float(DIFFICULTY_TO_NUMERIC.get(str(difficulty).strip().lower(), 0.0))


def emotion_to_bucket(emotion: str | None) -> str:
    if not emotion:
        return "unknown"
    e = str(emotion).strip().lower()
    if e in EMOTIONS:
        return e
    return "other"


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values))


def _max_correct_streak(correctness: np.ndarray, window: int) -> float:
    if correctness.size == 0:
        return 0.0
    seq = correctness[-window:] if correctness.size > window else correctness
    best = 0
    cur = 0
    for v in seq:
        if int(v) == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best)


def build_feature_vector(
    *,
    correctness: np.ndarray,
    time_spent: np.ndarray,
    difficulty_numeric: np.ndarray,
    emotion_buckets: np.ndarray,
    config: FeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    """
    Build engineered rolling features from a student's window history.

    Inputs must already be ordered by time ascending (oldest -> newest).
    """
    correctness = np.asarray(correctness, dtype=np.float32)
    time_spent = np.asarray(time_spent, dtype=np.float32)
    difficulty_numeric = np.asarray(difficulty_numeric, dtype=np.float32)
    emotion_buckets = np.asarray(emotion_buckets, dtype=object)

    n = int(correctness.size)
    if n == 0:
        # Keep vector shape stable even for empty windows.
        feats = np.zeros((17,), dtype=np.float32)
        return feats, get_feature_names()

    last10 = min(10, n)
    first_half = correctness[: n // 2]
    second_half = correctness[n // 2 :]

    correct_rate = _safe_mean(correctness)
    incorrect_rate = float(1.0 - correct_rate)
    avg_time = _safe_mean(time_spent)
    time_std = _safe_std(time_spent)
    avg_diff = _safe_mean(difficulty_numeric)
    acc_last10 = _safe_mean(correctness[-last10:])

    trend = float(_safe_mean(second_half) - _safe_mean(first_half))

    max_streak_last20 = _max_correct_streak(correctness, config.streak_window)

    # Emotion rates over the whole window (not just last10).
    emotion_total = float(max(1, n))
    rates = {
        e: float(np.sum(emotion_buckets == e)) / emotion_total for e in EMOTIONS
    }
    other_rate = float(
        np.sum(
            (emotion_buckets != "unknown")
            & (emotion_buckets != "other")
            & (~np.isin(emotion_buckets, EMOTIONS))
        )
    )
    # The above is conservative; we'll use "other"/"unknown" buckets explicitly instead.
    unknown_rate = float(np.sum(emotion_buckets == "unknown")) / emotion_total
    explicit_other_rate = float(np.sum(emotion_buckets == "other")) / emotion_total

    feats = np.array(
        [
            float(n),
            correct_rate,
            incorrect_rate,
            avg_time,
            time_std,
            avg_diff,
            rates["confused"],
            rates["frustrated"],
            rates["bored"],
            rates["focused"],
            rates["happy"],
            explicit_other_rate,
            unknown_rate,
            trend,
            max_streak_last20,
            acc_last10,
            float(np.sum(emotion_buckets == "frustrated") / emotion_total),
        ],
        dtype=np.float32,
    )
    return feats, get_feature_names()


def get_feature_names() -> list[str]:
    return [
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
        "frustrated_rate_repeat",  # duplicate-like feature; kept for stability across versions
    ]


def label_from_horizon(correctness_future: np.ndarray, threshold: float) -> int:
    if correctness_future.size == 0:
        return 0
    future_acc = float(np.mean(correctness_future))
    return int(future_acc < threshold)


def build_window_arrays(
    *,
    correctness: Iterable[bool | int],
    time_spent: Iterable[float | int],
    difficulty: Iterable[str | None],
    emotions: Iterable[str | None],
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(list(correctness), dtype=np.int32)
    t = np.asarray(list(time_spent), dtype=np.float32)
    d = np.asarray([difficulty_to_numeric(x) for x in difficulty], dtype=np.float32)
    e = np.asarray([emotion_to_bucket(x) for x in emotions], dtype=object)

    if c.size > config.window_attempts:
        c = c[-config.window_attempts :]
        t = t[-config.window_attempts :]
        d = d[-config.window_attempts :]
        e = e[-config.window_attempts :]

    return c, t, d, e

