"""Task 6: Collaborative question recommendation via latent factors.

Builds a student-question interaction matrix, fits a truncated-SVD model,
performs offline evaluation against a random baseline, and exports reusable
artifacts consumed by the Flask API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "backend" / "data" / "ml" / "interaction_datasets"
ARTIFACT_DIR = ROOT / "backend" / "models" / "recommender"


@dataclass
class EventRecord:
    user_id: int
    question_id: int
    rating: float
    is_correct: bool
    difficulty: str
    subject: str
    grade: str
    topic: str
    answered_at: float
    time_spent: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-factor question recommender (Task 6)")
    parser.add_argument("--components", type=int, default=32, help="Latent dimensions for SVD")
    parser.add_argument("--min-user-events", type=int, default=8, help="Minimum events required per user")
    parser.add_argument("--min-question-events", type=int, default=5, help="Minimum events per question")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each user's history reserved for evaluation")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for hit-rate evaluation")
    parser.add_argument("--eval-candidate-pool", type=int, default=60, help="Candidate pool size per user during evaluation")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    events, source_version = load_events()
    if not events:
        raise SystemExit("No interaction events were found. Build datasets or seed the database first.")

    print(f"[Recommender] Loaded {len(events):,} events from source={source_version}")

    train_events, test_events = split_events(events, args.min_user_events, args.test_fraction)
    if len(train_events) < 50 or len(test_events) < 20:
        raise SystemExit("Not enough events after filtering to train recommender.")

    train_events, test_events = filter_questions(train_events, test_events, args.min_question_events)
    user_ids = sorted({evt.user_id for evt in train_events})
    question_ids = sorted({evt.question_id for evt in train_events})

    if len(user_ids) < 5 or len(question_ids) < 25:
        raise SystemExit("Dataset too small after filtering. Collect more interactions.")

    matrix, mappings = build_matrix(train_events, user_ids, question_ids)
    latent_dim = max(1, min(args.components, min(matrix.shape) - 1))
    if latent_dim < 1:
        latent_dim = 1

    user_factors, question_factors = fit_svd(matrix, latent_dim)
    stats = compute_biases(train_events, mappings, matrix.data)

    predictor = build_predictor(user_factors, question_factors, stats)
    offline_metrics = evaluate_model(
        predictor,
        train_events,
        test_events,
        mappings,
        args.top_k,
        args.eval_candidate_pool,
        seed=args.seed,
    )

    manifest = save_artifact(
        user_factors,
        question_factors,
        mappings,
        stats,
        offline_metrics,
        args,
        source_version,
    )

    print("[Recommender] Artifact saved", manifest["weights_path"])
    print(json.dumps({"metrics": manifest["metrics"]}, indent=2))


def load_events() -> Tuple[List[EventRecord], str]:
    snapshot_events, version = load_snapshot_events()
    if snapshot_events:
        return snapshot_events, version
    return load_db_events(), "live_db"


def load_snapshot_events() -> Tuple[List[EventRecord], str]:
    latest_version_path = DATA_ROOT / "LATEST_VERSION"
    if not latest_version_path.exists():
        return [], ""
    version = latest_version_path.read_text(encoding="utf-8").strip()
    events_path = DATA_ROOT / version / "events.jsonl"
    if not events_path.exists():
        return [], ""

    events: List[EventRecord] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = event_from_payload(payload)
            if record:
                events.append(record)
    return events, version


def load_db_events() -> List[EventRecord]:
    create_app, AnswerLog, Question, db = _load_backend_models()
    app = create_app("development")
    with app.app_context():
        rows = (
            db.session.query(AnswerLog, Question)
            .join(Question, AnswerLog.question_id == Question.id)
            .order_by(AnswerLog.answered_at.asc())
            .all()
        )

    events: List[EventRecord] = []
    for answer, question in rows:
        record = EventRecord(
            user_id=int(answer.user_id),
            question_id=int(answer.question_id),
            rating=learning_gain(answer.is_correct, question.difficulty, answer.time_spent or 0),
            is_correct=bool(answer.is_correct),
            difficulty=str(question.difficulty or "medium").strip().lower(),
            subject=str(question.subject or "general").strip().lower(),
            grade=str(question.grade or "").strip().lower(),
            topic=str(question.syllabus_topic or "general").strip().lower(),
            answered_at=(answer.answered_at or datetime.now(timezone.utc)).timestamp(),
            time_spent=float(answer.time_spent or 0),
        )
        events.append(record)
    return events


def event_from_payload(payload: Dict) -> EventRecord | None:
    question_id = payload.get("question_id")
    if question_id is None:
        return None
    student_key = str(payload.get("student_key") or f"u_{payload.get('student_id')}")
    user_id = stable_student_id(student_key, payload.get("student_id"))
    answered_at = parse_timestamp(payload.get("answered_at"))
    rating = learning_gain(payload.get("is_correct", False), payload.get("difficulty"), payload.get("time_spent_sec") or payload.get("time_spent") or 0)
    return EventRecord(
        user_id=user_id,
        question_id=int(question_id),
        rating=rating,
        is_correct=bool(payload.get("is_correct", False)),
        difficulty=str(payload.get("difficulty") or "medium").strip().lower(),
        subject=str(payload.get("subject") or "general").strip().lower(),
        grade=str(payload.get("grade") or "").strip().lower(),
        topic=str(payload.get("topic") or "general").strip().lower(),
        answered_at=answered_at,
        time_spent=float(payload.get("time_spent_sec") or payload.get("time_spent") or 0),
    )


def split_events(events: List[EventRecord], min_user_events: int, test_fraction: float) -> Tuple[List[EventRecord], List[EventRecord]]:
    per_user: Dict[int, List[EventRecord]] = defaultdict(list)
    for event in events:
        per_user[event.user_id].append(event)

    train: List[EventRecord] = []
    test: List[EventRecord] = []
    for user_id, records in per_user.items():
        if len(records) < min_user_events:
            continue
        records.sort(key=lambda evt: evt.answered_at)
        holdout = max(1, int(len(records) * test_fraction))
        test_chunk = records[-holdout:]
        train_chunk = records[:-holdout]
        if not train_chunk:
            continue
        train.extend(train_chunk)
        test.extend(test_chunk)
    return train, test


def filter_questions(
    train_events: List[EventRecord],
    test_events: List[EventRecord],
    min_question_events: int,
) -> Tuple[List[EventRecord], List[EventRecord]]:
    counts: Dict[int, int] = defaultdict(int)
    for event in train_events:
        counts[event.question_id] += 1
    eligible = {qid for qid, cnt in counts.items() if cnt >= min_question_events}
    filtered_train = [evt for evt in train_events if evt.question_id in eligible]
    filtered_test = [evt for evt in test_events if evt.question_id in eligible]
    return filtered_train, filtered_test


def build_matrix(
    train_events: List[EventRecord],
    user_ids: List[int],
    question_ids: List[int],
) -> Tuple[sparse.csr_matrix, dict]:
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    question_index = {qid: idx for idx, qid in enumerate(question_ids)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for event in train_events:
        rows.append(user_index[event.user_id])
        cols.append(question_index[event.question_id])
        data.append(event.rating)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(question_ids)))
    return matrix, {
        "user_index": user_index,
        "question_index": question_index,
        "user_ids": user_ids,
        "question_ids": question_ids,
    }


def fit_svd(matrix: sparse.csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    u, s, vt = svds(matrix, k=k)
    u = u[:, ::-1]
    s = s[::-1]
    vt = vt[::-1, :]
    sqrt_s = np.sqrt(s)
    user_factors = (u * sqrt_s).astype(np.float32)
    question_factors = (vt.T * sqrt_s).astype(np.float32)
    return user_factors, question_factors


def compute_biases(train_events: List[EventRecord], mappings: dict, values: np.ndarray) -> dict:
    user_index = mappings["user_index"]
    question_index = mappings["question_index"]
    global_bias = float(np.mean(values)) if values.size else 0.5

    per_user: Dict[int, List[float]] = defaultdict(list)
    per_question: Dict[int, List[float]] = defaultdict(list)
    question_correct: Dict[int, int] = defaultdict(int)
    question_counts: Dict[int, int] = defaultdict(int)
    for event in train_events:
        per_user[event.user_id].append(event.rating)
        per_question[event.question_id].append(event.rating)
        question_counts[event.question_id] += 1
        if event.is_correct:
            question_correct[event.question_id] += 1

    user_bias = np.zeros(len(user_index), dtype=np.float32)
    for user_id, values in per_user.items():
        user_bias[user_index[user_id]] = float(np.mean(values) - global_bias)

    question_bias = np.zeros(len(question_index), dtype=np.float32)
    popularity = np.zeros(len(question_index), dtype=np.float32)
    success = np.zeros(len(question_index), dtype=np.float32)
    max_pop = max(question_counts.values()) if question_counts else 1
    for question_id, values in per_question.items():
        idx = question_index[question_id]
        question_bias[idx] = float(np.mean(values) - global_bias)
        popularity[idx] = question_counts[question_id] / max_pop
        success[idx] = question_correct.get(question_id, 0) / max(1, question_counts[question_id])

    return {
        "global_bias": global_bias,
        "user_bias": user_bias,
        "question_bias": question_bias,
        "question_popularity": popularity,
        "question_success": success,
    }


def build_predictor(user_factors: np.ndarray, question_factors: np.ndarray, stats: dict):
    global_bias = stats["global_bias"]
    user_bias = stats["user_bias"]
    question_bias = stats["question_bias"]
    popularity = stats["question_popularity"]
    success = stats["question_success"]

    user_mean = np.mean(user_factors, axis=0) if user_factors.size else np.zeros((1,), dtype=np.float32)
    question_mean = np.mean(question_factors, axis=0) if question_factors.size else np.zeros((1,), dtype=np.float32)

    def predictor(user_idx: int | None, question_idx: int | None) -> Tuple[float, float, float]:
        user_vec = user_factors[user_idx] if user_idx is not None else user_mean
        question_vec = question_factors[question_idx] if question_idx is not None else question_mean
        bias = global_bias
        if user_idx is not None:
            bias += float(user_bias[user_idx])
        if question_idx is not None:
            bias += float(question_bias[question_idx])
        score = bias + float(np.dot(user_vec, question_vec))
        prob = sigmoid(score)
        pop = float(popularity[question_idx]) if question_idx is not None else 0.5
        succ = float(success[question_idx]) if question_idx is not None else 0.5
        return prob, pop, succ

    return predictor


def evaluate_model(
    predictor,
    train_events: List[EventRecord],
    test_events: List[EventRecord],
    mappings: dict,
    top_k: int,
    candidate_pool: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    user_index = mappings["user_index"]
    question_index = mappings["question_index"]

    y_true: List[float] = []
    y_pred: List[float] = []
    for event in test_events:
        user_idx = user_index.get(event.user_id)
        question_idx = question_index.get(event.question_id)
        pred, _, _ = predictor(user_idx, question_idx)
        y_true.append(event.rating)
        y_pred.append(pred)

    rmse = math.sqrt(np.mean([(p - t) ** 2 for p, t in zip(y_pred, y_true)])) if y_true else 0.0
    mae = float(np.mean([abs(p - t) for p, t in zip(y_pred, y_true)])) if y_true else 0.0

    train_by_user: Dict[int, set] = defaultdict(set)
    for evt in train_events:
        train_by_user[evt.user_id].add(evt.question_id)
    test_by_user: Dict[int, set] = defaultdict(set)
    for evt in test_events:
        test_by_user[evt.user_id].add(evt.question_id)

    hits_model = 0
    hits_random = 0
    considered_users = 0
    coverage_users = 0
    all_questions = set(question_index.keys())

    for user_id, test_items in test_by_user.items():
        if not test_items:
            continue
        available = list(all_questions - train_by_user.get(user_id, set()))
        if not available:
            continue
        pool = list(test_items)
        remainder = [qid for qid in available if qid not in test_items]
        rng.shuffle(remainder)
        pool.extend(remainder[: max(0, candidate_pool - len(pool))])
        if not pool:
            continue
        considered_users += 1
        if len(pool) >= top_k:
            coverage_users += 1

        scored = []
        for qid in pool:
            user_idx = user_index.get(user_id)
            question_idx = question_index.get(qid)
            score, pop, succ = predictor(user_idx, question_idx)
            novelty = 1.0 if qid not in train_by_user.get(user_id, set()) else 0.3
            final = score + 0.05 * pop + 0.05 * (1.0 - succ) + 0.05 * novelty
            scored.append((qid, final))
        scored.sort(key=lambda item: item[1], reverse=True)
        top_pred = [qid for qid, _ in scored[: min(top_k, len(scored))]]
        if any(qid in test_items for qid in top_pred):
            hits_model += 1

        random_top = rng.sample(pool, min(top_k, len(pool)))
        if any(qid in test_items for qid in random_top):
            hits_random += 1

    hit_rate_model = hits_model / considered_users if considered_users else 0.0
    hit_rate_random = hits_random / considered_users if considered_users else 0.0
    coverage = coverage_users / considered_users if considered_users else 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "hit_rate_model": round(hit_rate_model, 4),
        "hit_rate_random": round(hit_rate_random, 4),
        "hit_rate_lift": round(hit_rate_model - hit_rate_random, 4),
        "coverage": round(coverage, 4),
        "users_evaluated": considered_users,
        "test_events": len(test_events),
    }


def save_artifact(
    user_factors: np.ndarray,
    question_factors: np.ndarray,
    mappings: dict,
    stats: dict,
    metrics: dict,
    args: argparse.Namespace,
    dataset_version: str,
) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    weights_path = ARTIFACT_DIR / f"question_recommender_{timestamp}.npz"

    np.savez_compressed(
        weights_path,
        user_ids=np.array(mappings["user_ids"], dtype=np.int64),
        question_ids=np.array(mappings["question_ids"], dtype=np.int64),
        user_factors=user_factors,
        question_factors=question_factors,
        user_bias=stats["user_bias"],
        question_bias=stats["question_bias"],
        global_bias=np.array([stats["global_bias"]], dtype=np.float32),
        question_popularity=stats["question_popularity"],
        question_success=stats["question_success"],
    )

    manifest = {
        "artifact_id": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights_path": weights_path.name,
        "dataset_version": dataset_version,
        "metrics": metrics,
        "counts": {
            "users": len(mappings["user_ids"]),
            "questions": len(mappings["question_ids"]),
        },
        "config": {
            "components": args.components,
            "min_user_events": args.min_user_events,
            "min_question_events": args.min_question_events,
            "test_fraction": args.test_fraction,
            "top_k": args.top_k,
            "candidate_pool": args.eval_candidate_pool,
            "seed": args.seed,
        },
    }

    manifest_path = ARTIFACT_DIR / f"question_recommender_{timestamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LATEST_PATH = ARTIFACT_DIR / "latest_manifest.json"
    LATEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_timestamp(value: str | None) -> float:
    if not value:
        return datetime.now(timezone.utc).timestamp()
    try:
        iso = value.replace("Z", "+00:00")
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


def learning_gain(is_correct: bool, difficulty: str | None, time_spent: float) -> float:
    base = 1.0 if is_correct else 0.35
    diff_weights = {
        "easy": 0.9,
        "medium": 1.0,
        "hard": 1.15,
        "expert": 1.3,
    }
    difficulty_weight = diff_weights.get((difficulty or "medium").strip().lower(), 1.0)
    time_adjust = 1.05 - max(-0.35, min(0.35, (float(time_spent) - 40.0) / 160.0))
    value = base * difficulty_weight * time_adjust
    return float(max(0.05, min(1.0, value)))


def stable_student_id(student_key: str, provided_id) -> int:
    if provided_id is not None:
        return int(provided_id)
    digest = hashlib.sha1(student_key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _load_backend_models():
    import sys

    sys.path.insert(0, str(ROOT))
    from backend.app import create_app
    from backend.models import AnswerLog, Question, db

    return create_app, AnswerLog, Question, db


if __name__ == "__main__":
    main()
