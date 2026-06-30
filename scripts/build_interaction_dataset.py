"""Build versioned interaction datasets for AI training.

Features:
- Extracts real events from database answer logs.
- Generates synthetic student events when real data is sparse.
- Splits by student (no user leakage between train/val/test).
- Writes versioned JSONL + manifest artifacts under backend/data/ml/interaction_datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "backend" / "data" / "ml" / "interaction_datasets"
SCHEMA_VERSION = "1.0.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_iso(value: datetime | None) -> str:
    if value is None:
        return _utc_now().isoformat()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _student_split(student_key: str) -> str:
    hashed = hashlib.sha256(student_key.encode("utf-8")).hexdigest()
    bucket = int(hashed[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _load_backend_models():
    import sys

    sys.path.insert(0, str(ROOT))
    from backend.app import create_app
    from backend.models import AnswerLog, Question, User, db

    return create_app, AnswerLog, Question, User, db


@dataclass
class BuildConfig:
    min_events: int
    min_users: int
    seed: int


def extract_real_events() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    create_app, AnswerLog, Question, User, db = _load_backend_models()

    app = create_app("development")
    with app.app_context():
        users = User.query.filter(User.role == "student").all()

        rows = (
            db.session.query(AnswerLog, Question, User)
            .join(Question, AnswerLog.question_id == Question.id)
            .join(User, AnswerLog.user_id == User.id)
            .order_by(AnswerLog.answered_at.asc(), AnswerLog.id.asc())
            .all()
        )

        events: list[dict[str, Any]] = []
        for answer, question, user in rows:
            options = question.options if isinstance(question.options, list) else []
            correct_index = int(question.correct_index)
            selected_index = int(answer.selected_index)
            event = {
                "event_id": f"real-{answer.id}",
                "source": "real",
                "student_key": f"u_{answer.user_id}",
                "student_id": int(answer.user_id),
                "question_id": int(answer.question_id),
                "subject": str(question.subject or "").strip().lower(),
                "grade": str(question.grade or "").strip().lower(),
                "topic": str(question.syllabus_topic or "general").strip().lower(),
                "difficulty": str(question.difficulty or "medium").strip().lower(),
                "selected_index": selected_index,
                "correct_index": correct_index,
                "is_correct": bool(answer.is_correct),
                "time_spent_sec": int(max(0, answer.time_spent or 0)),
                "emotion": str(answer.emotion_at_time or "unknown").strip().lower(),
                "options_count": len(options),
                "answered_at": _to_utc_iso(answer.answered_at),
            }
            events.append(event)

        student_profiles = [
            {
                "student_key": f"u_{u.id}",
                "student_id": int(u.id),
                "grade": str(u.grade or "").strip().lower() or None,
                "name": str(u.name or "").strip() or None,
            }
            for u in users
        ]

    return events, student_profiles


def _difficulty_to_scalar(level: str) -> float:
    return {
        "easy": -0.8,
        "medium": 0.0,
        "hard": 0.9,
        "expert": 1.5,
    }.get(level, 0.0)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def generate_synthetic_events(
    existing_events: list[dict[str, Any]],
    existing_students: list[dict[str, Any]],
    config: BuildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    create_app, _AnswerLog, Question, _User, _db = _load_backend_models()

    app = create_app("development")
    with app.app_context():
        questions = Question.query.all()

    if not questions:
        return [], []

    rng = random.Random(config.seed)

    question_bank: list[dict[str, Any]] = []
    for q in questions:
        options = q.options if isinstance(q.options, list) else []
        if len(options) < 2:
            continue
        question_bank.append(
            {
                "question_id": int(q.id),
                "subject": str(q.subject or "").strip().lower(),
                "grade": str(q.grade or "").strip().lower(),
                "topic": str(q.syllabus_topic or "general").strip().lower(),
                "difficulty": str(q.difficulty or "medium").strip().lower(),
                "correct_index": int(q.correct_index),
                "options_count": len(options),
            }
        )

    if not question_bank:
        return [], []

    existing_keys = {row["student_key"] for row in existing_events}
    target_user_count = max(config.min_users, len(existing_keys))

    synthetic_students: list[dict[str, Any]] = []
    while len(existing_keys) + len(synthetic_students) < target_user_count:
        sid = len(synthetic_students) + 1
        student_key = f"sim_{sid:04d}"
        synthetic_students.append(
            {
                "student_key": student_key,
                "student_id": None,
                "grade": rng.choice(["elementary", "middle", "high", "college"]),
                "name": None,
            }
        )

    total_needed_events = max(0, config.min_events - len(existing_events))
    if total_needed_events == 0:
        return [], synthetic_students

    # Distribute events across synthetic students.
    if not synthetic_students:
        synthetic_students = [
            {
                "student_key": "sim_0001",
                "student_id": None,
                "grade": "high",
                "name": None,
            }
        ]

    subject_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in question_bank:
        subject_groups[row["subject"]].append(row)

    synthetic_events: list[dict[str, Any]] = []
    start_time = _utc_now() - timedelta(days=90)

    per_student = max(40, total_needed_events // len(synthetic_students))
    event_counter = 0

    for profile in synthetic_students:
        ability = rng.uniform(-1.2, 1.4)
        preferred_subjects = rng.sample(list(subject_groups.keys()), k=min(2, len(subject_groups)))
        current_time = start_time + timedelta(minutes=rng.randint(0, 7200))

        for _ in range(per_student):
            if len(synthetic_events) >= total_needed_events:
                break

            subject = rng.choice(preferred_subjects) if preferred_subjects else rng.choice(list(subject_groups.keys()))
            pool = subject_groups.get(subject) or question_bank
            question = rng.choice(pool)

            difficulty_scalar = _difficulty_to_scalar(question["difficulty"])
            p_correct = _sigmoid(ability - difficulty_scalar + rng.uniform(-0.35, 0.35))
            is_correct = rng.random() < p_correct

            options_count = max(2, int(question["options_count"]))
            correct_index = int(question["correct_index"])
            if is_correct:
                selected_index = correct_index
            else:
                wrong = [idx for idx in range(options_count) if idx != correct_index]
                selected_index = rng.choice(wrong)

            base_time = {
                "easy": 24,
                "medium": 39,
                "hard": 57,
                "expert": 72,
            }.get(question["difficulty"], 40)
            if not is_correct:
                base_time += 8
            time_spent = max(6, int(rng.gauss(base_time, 7)))

            if is_correct and time_spent <= 35:
                emotion = rng.choice(["focused", "engaged", "neutral"])
            elif is_correct:
                emotion = rng.choice(["neutral", "focused"])
            else:
                emotion = rng.choice(["confused", "frustrated", "neutral"])

            event_counter += 1
            synthetic_events.append(
                {
                    "event_id": f"sim-{event_counter}",
                    "source": "simulated",
                    "student_key": profile["student_key"],
                    "student_id": None,
                    "question_id": question["question_id"],
                    "subject": question["subject"],
                    "grade": question["grade"],
                    "topic": question["topic"],
                    "difficulty": question["difficulty"],
                    "selected_index": selected_index,
                    "correct_index": correct_index,
                    "is_correct": bool(is_correct),
                    "time_spent_sec": int(time_spent),
                    "emotion": emotion,
                    "options_count": options_count,
                    "answered_at": _to_utc_iso(current_time),
                }
            )

            current_time += timedelta(minutes=max(1, int(rng.gauss(11, 3))))

    # If rounding left a gap, fill from pooled synthetic users.
    while len(synthetic_events) < total_needed_events:
        profile = rng.choice(synthetic_students)
        pool = subject_groups[rng.choice(list(subject_groups.keys()))]
        question = rng.choice(pool)
        event_counter += 1
        synthetic_events.append(
            {
                "event_id": f"sim-{event_counter}",
                "source": "simulated",
                "student_key": profile["student_key"],
                "student_id": None,
                "question_id": question["question_id"],
                "subject": question["subject"],
                "grade": question["grade"],
                "topic": question["topic"],
                "difficulty": question["difficulty"],
                "selected_index": 0,
                "correct_index": question["correct_index"],
                "is_correct": False,
                "time_spent_sec": 45,
                "emotion": "neutral",
                "options_count": question["options_count"],
                "answered_at": _to_utc_iso(start_time + timedelta(minutes=event_counter)),
            }
        )

    return synthetic_events, synthetic_students


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _latest_manifest_path() -> Path | None:
    if not DATA_ROOT.exists():
        return None
    manifests = sorted(DATA_ROOT.glob("*/manifest.json"), reverse=True)
    return manifests[0] if manifests else None


def _is_stale(max_age_hours: int) -> bool:
    manifest = _latest_manifest_path()
    if manifest is None:
        return True
    modified = datetime.fromtimestamp(manifest.stat().st_mtime, tz=timezone.utc)
    return (_utc_now() - modified) > timedelta(hours=max_age_hours)


def build_dataset(config: BuildConfig) -> dict[str, Any]:
    real_events, real_students = extract_real_events()
    synthetic_events, synthetic_students = generate_synthetic_events(real_events, real_students, config)

    all_events = real_events + synthetic_events
    all_events.sort(key=lambda row: row["answered_at"])

    student_keys = sorted({event["student_key"] for event in all_events})
    split_by_student = {student_key: _student_split(student_key) for student_key in student_keys}

    # Ensure every split has at least one student when we have enough students.
    if len(student_keys) >= 3:
        split_counts = Counter(split_by_student.values())
        required_splits = ("train", "val", "test")
        for missing_split in required_splits:
            if split_counts.get(missing_split, 0) > 0:
                continue

            donor_split = max(required_splits, key=lambda split_name: split_counts.get(split_name, 0))
            if split_counts.get(donor_split, 0) <= 1:
                continue

            donor_students = sorted(
                student_key
                for student_key, split_name in split_by_student.items()
                if split_name == donor_split
            )
            if not donor_students:
                continue

            moved_student = donor_students[-1]
            split_by_student[moved_student] = missing_split
            split_counts[donor_split] -= 1
            split_counts[missing_split] += 1

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for event in all_events:
        split = split_by_student.get(event["student_key"], _student_split(event["student_key"]))
        event["split"] = split
        by_split[split].append(event)

    version = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = DATA_ROOT / version
    splits_dir = out_dir / "splits"

    _write_jsonl(out_dir / "events.jsonl", all_events)
    _write_jsonl(splits_dir / "train.jsonl", by_split["train"])
    _write_jsonl(splits_dir / "val.jsonl", by_split["val"])
    _write_jsonl(splits_dir / "test.jsonl", by_split["test"])

    users_by_split = {"train": set(), "val": set(), "test": set()}
    for event in all_events:
        users_by_split[event["split"]].add(event["student_key"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "created_at": _to_utc_iso(_utc_now()),
        "config": {
            "min_events": config.min_events,
            "min_users": config.min_users,
            "seed": config.seed,
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        },
        "paths": {
            "events": "events.jsonl",
            "train": "splits/train.jsonl",
            "val": "splits/val.jsonl",
            "test": "splits/test.jsonl",
        },
        "counts": {
            "events_total": len(all_events),
            "events_real": len(real_events),
            "events_simulated": len(synthetic_events),
            "students_real": len({row["student_key"] for row in real_events}),
            "students_simulated": len(synthetic_students),
            "students_total": len({row["student_key"] for row in all_events}),
            "events_train": len(by_split["train"]),
            "events_val": len(by_split["val"]),
            "events_test": len(by_split["test"]),
            "students_train": len(users_by_split["train"]),
            "students_val": len(users_by_split["val"]),
            "students_test": len(users_by_split["test"]),
        },
        "student_leakage_check": {
            "train_val_overlap": sorted(users_by_split["train"] & users_by_split["val"]),
            "train_test_overlap": sorted(users_by_split["train"] & users_by_split["test"]),
            "val_test_overlap": sorted(users_by_split["val"] & users_by_split["test"]),
        },
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATA_ROOT / "LATEST_VERSION").write_text(version, encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned interaction dataset snapshots.")
    parser.add_argument("--if-stale", action="store_true", help="Skip build if latest snapshot is still fresh.")
    parser.add_argument("--max-age-hours", type=int, default=24, help="Freshness threshold when using --if-stale.")
    parser.add_argument("--min-events", type=int, default=20000, help="Minimum total events in output dataset.")
    parser.add_argument("--min-users", type=int, default=60, help="Minimum distinct students in output dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for simulation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.if_stale and not _is_stale(max(1, args.max_age_hours)):
        manifest = _latest_manifest_path()
        print(f"[DATASET] Fresh snapshot already exists: {manifest}")
        return 0

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    config = BuildConfig(
        min_events=max(1000, args.min_events),
        min_users=max(10, args.min_users),
        seed=args.seed,
    )
    manifest = build_dataset(config)

    overlaps = manifest["student_leakage_check"]
    has_leakage = any(overlaps[key] for key in overlaps)
    if has_leakage:
        print("[DATASET] ERROR: Student leakage detected across splits.")
        return 2

    print(
        "[DATASET] Built interaction dataset "
        f"{manifest['version']} with {manifest['counts']['events_total']} events "
        f"({manifest['counts']['events_real']} real / {manifest['counts']['events_simulated']} simulated)."
    )
    print(
        "[DATASET] Split events -> "
        f"train: {manifest['counts']['events_train']}, "
        f"val: {manifest['counts']['events_val']}, "
        f"test: {manifest['counts']['events_test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
