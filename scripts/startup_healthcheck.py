"""One-command startup bootstrap for Elevate.

This script is intentionally idempotent:
- Runs Alembic migrations before any DB access (auto-applies new columns).
- Seeds users only when missing.
- Ensures question bank size is healthy.
- Supports optional strict rebuild mode via env var.

Responsibility boundary: this file only verifies/repairs application state
(DB schema, seed data, question bank, ML artifacts). It has no opinion about
how things are printed — all console styling lives in scripts/bootstrap/logger.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Needed so `from scripts.bootstrap.logger import ...` resolves no matter how
# this script is invoked (it is normally launched with an absolute path via
# subprocess, which does not put PROJECT_ROOT on sys.path by itself).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.bootstrap.logger import info, success, warning, error, stage, init as init_logger
except Exception:
    # Defensive fallback: never let a logging import break the health check itself.
    def init_logger() -> None:
        return None

    def info(message: str) -> None:
        print(f"[BOOTSTRAP] {message}")

    def success(message: str) -> None:
        print(f"[BOOTSTRAP] {message}")

    def warning(message: str) -> None:
        print(f"[BOOTSTRAP] WARNING: {message}")

    def error(message: str) -> None:
        print(f"[BOOTSTRAP] ERROR: {message}")

    import contextlib

    @contextlib.contextmanager
    def stage(name: str):
        print(f"[BOOTSTRAP] -- {name} --")
        yield


PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
_APP_CACHE = None
EMOTION_DATASET_DIR = ROOT / "dataset"
EMOTION_CLASS_FOLDERS = [
    "happy",
    "bored",
    "focused",
    "confused",
    "neutral",
    "angry",
    "surprised",
]

# ---------------------------------------------------------------------------
# Script locations (kept in one place so the folder layout only needs to be
# updated here if scripts/ is ever reorganized again).
# ---------------------------------------------------------------------------
RUN_MIGRATIONS_SCRIPT = ROOT / "scripts" / "database" / "run_migrations.py"
SEED_USERS_SCRIPT = "scripts/database/seed_users.py"
BUILD_INTERACTION_DATASET_SCRIPT = "scripts/datasets/build_interaction_dataset.py"
FIT_BKT_SCRIPT = "scripts/training/fit_bkt_model.py"
TRAIN_DKT_SCRIPT = "scripts/training/train_dkt_model.py"
TRAIN_EMOTION_SCRIPT = "scripts/training/train_emotion_cnn_hf.py"
TRAIN_AT_RISK_SCRIPT = "scripts/training/train_at_risk_predictor.py"
TRAIN_STRICT_PIPELINE_SCRIPT = "scripts/training/train_strict_pipeline.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env into os.environ so DATABASE_URL is available for migrations."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        warning(f"Could not load .env: {exc}")


def _run_migrations() -> None:
    """Apply all pending Alembic migrations against the configured database.

    Uses run_migrations.py which calls Alembic Python API directly — no
    Flask CLI needed. Is idempotent when schema is already at head.
    Must run BEFORE any SQLAlchemy model query so columns exist in Postgres.
    """
    _load_dotenv()

    if not RUN_MIGRATIONS_SCRIPT.exists():
        # Previously this was a silent warning-and-skip. Skipping migrations
        # silently means the app can boot against a stale schema, which is
        # worse than failing loudly here.
        raise RuntimeError(
            f"run_migrations.py not found at {RUN_MIGRATIONS_SCRIPT}. "
            "Database migrations cannot be skipped — fix the path or restore the file."
        )

    info("Running database migrations (alembic upgrade head)...")
    result = subprocess.run(
        [str(PYTHON_EXE), str(RUN_MIGRATIONS_SCRIPT)],
        cwd=ROOT,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Database migration failed. "
            "Check your DATABASE_URL in .env and that the PostgreSQL server is reachable."
        )
    success("Migrations applied successfully.")


def _run(command: list[str], description: str) -> None:
    info(description)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed step: {description}")


def _is_truthy_env(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _assert_gemini_key_if_required() -> None:
    is_production_runtime = (
        str(os.environ.get("FLASK_ENV", "")).strip().lower() == "production"
        or bool(str(os.environ.get("RENDER_SERVICE_ID") or "").strip())
        or _is_truthy_env("RENDER", "0")
        or bool(str(os.environ.get("K_SERVICE") or "").strip())
    )

    explicit_requirement = os.environ.get("ELEVATE_REQUIRE_GEMINI_KEY")
    if explicit_requirement is None:
        require_key = is_production_runtime
    else:
        require_key = _is_truthy_env("ELEVATE_REQUIRE_GEMINI_KEY", "0")

    if not require_key:
        return

    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )
    if str(gemini_key).strip():
        return

    raise RuntimeError(
        "Gemini API key is required but missing. "
        "Set GEMINI_API_KEY (or GOOGLE_API_KEY), or set ELEVATE_REQUIRE_GEMINI_KEY=0 "
        "for non-production bootstrap runs."
    )



def _read_counts() -> tuple[int, int]:
    global _APP_CACHE

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from backend.app import create_app
    from backend.models import Question, User

    if _APP_CACHE is None:
        _APP_CACHE = create_app("development")

    app = _APP_CACHE
    with app.app_context():
        return User.query.count(), Question.query.count()


def _ensure_users(user_count: int) -> None:
    if user_count > 0:
        success(f"Users present: {user_count}. Skipping user seed.")
        return

    _run([str(PYTHON_EXE), SEED_USERS_SCRIPT], "Seeding default users")


def _ensure_questions(question_count: int) -> None:
    strict_mode = _is_truthy_env("ELEVATE_STRICT_MODE", "0")
    strict_rebuild = os.environ.get("ELEVATE_STRICT_REBUILD", "0") == "1"
    per_subtopic = int(os.environ.get("ELEVATE_PER_SUBTOPIC", "80"))
    min_questions = int(os.environ.get("ELEVATE_MIN_QUESTIONS", "300"))

    if strict_rebuild or strict_mode:
        _run(
            [
                str(PYTHON_EXE),
                "backend/seed_questions.py",
                "--reset-questions",
                "--strict-stem-rebuild",
                "--per-subtopic",
                str(max(20, min(per_subtopic, 200))),
            ],
            "Strict question bank rebuild",
        )
        return

    if question_count >= min_questions:
        success(f"Question bank healthy: {question_count} questions (target >= {min_questions}).")
        return

    # Non-destructive top-up path for normal startup.
    missing = max(0, min_questions - question_count)
    groups = 63
    per_topic = max(8, min(50, (missing + groups - 1) // groups))

    _run(
        [
            str(PYTHON_EXE),
            "backend/seed_questions.py",
            "--augment-large",
            "--per-topic",
            str(per_topic),
        ],
        f"Top-up question bank with synthetic coverage (per-topic={per_topic})",
    )


def _ensure_interaction_dataset() -> None:
    max_age_hours = max(1, int(os.environ.get("ELEVATE_DATASET_MAX_AGE_HOURS", "24")))
    min_events = max(1000, int(os.environ.get("ELEVATE_DATASET_MIN_EVENTS", "20000")))
    min_users = max(10, int(os.environ.get("ELEVATE_DATASET_MIN_USERS", "60")))
    dataset_seed = int(os.environ.get("ELEVATE_DATASET_SEED", "42"))

    command = [
        str(PYTHON_EXE),
        BUILD_INTERACTION_DATASET_SCRIPT,
        "--if-stale",
        "--max-age-hours",
        str(max_age_hours),
        "--min-events",
        str(min_events),
        "--min-users",
        str(min_users),
        "--seed",
        str(dataset_seed),
    ]

    if _is_truthy_env("ELEVATE_STRICT_MODE", "0"):
        command = [
            str(PYTHON_EXE),
            BUILD_INTERACTION_DATASET_SCRIPT,
            "--min-events",
            str(min_events),
            "--min-users",
            str(min_users),
            "--seed",
            str(dataset_seed),
        ]

    _run(
        command,
        "Ensuring interaction dataset snapshot for model training",
    )


def _run_local_strict_pipeline() -> None:
    min_emotion_acc = str(os.environ.get("HF_ML_MIN_EMOTION_ACCURACY", "0.90")).strip() or "0.90"
    processes = str(os.environ.get("HF_ML_TRAINING_PROCESSES", "4")).strip() or "4"

    _run(
        [
            str(PYTHON_EXE),
            TRAIN_STRICT_PIPELINE_SCRIPT,
            "--min-emotion-accuracy",
            min_emotion_acc,
            "--min-emotion-macro-f1",
            min_emotion_acc,
            "--processes",
            processes,
        ],
        "Running strict fail-fast local ML pipeline",
    )


def _run_remote_hf_strict_pipeline() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.hf_training_service import trigger_and_wait_hf_strict_training

    info("Triggering strict HF training via backend service client ...")
    result = trigger_and_wait_hf_strict_training()
    if not result.get("ok"):
        raise RuntimeError(
            "HF strict training failed: "
            f"status_code={result.get('status_code')} "
            f"endpoint={result.get('endpoint')} "
            f"error={result.get('error')}"
        )

    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    state = str(payload.get("status") or "").strip().lower()
    if state != "succeeded":
        raise RuntimeError(
            "HF strict training did not succeed: "
            f"state={state} payload={payload}"
        )

    summary = payload.get("summary")
    success(f"HF strict training succeeded. summary={summary}")


def _run_strict_pipeline_if_enabled() -> bool:
    strict_mode = _is_truthy_env("ELEVATE_STRICT_MODE", "0")
    if not strict_mode:
        return False

    use_remote_hf = _is_truthy_env("ELEVATE_STRICT_REMOTE_HF", "0")
    if use_remote_hf:
        _run_remote_hf_strict_pipeline()
    else:
        _run_local_strict_pipeline()
    return True


def _ensure_models_non_strict() -> None:
    _ensure_bkt_model()
    _ensure_dkt_model()
    _ensure_emotion_model()
    _ensure_at_risk_model()


def _ensure_bkt_model() -> None:
    """Train BKT model if it doesn't exist or dataset is fresh."""
    model_dir = ROOT / "backend" / "models" / "bkt"
    model_dir.mkdir(parents=True, exist_ok=True)

    latest_model = model_dir / "bkt_model_latest.pkl"
    if latest_model.exists():
        success(f"BKT model exists: {latest_model}. Skipping retraining.")
        return

    _run(
        [str(PYTHON_EXE), FIT_BKT_SCRIPT],
        "Training BKT EM parameters on interaction dataset",
    )


def _ensure_dkt_model() -> None:
    """Train Deep Knowledge Tracing model if it doesn't exist."""
    model_dir = ROOT / "backend" / "models" / "dkt"
    model_dir.mkdir(parents=True, exist_ok=True)

    latest_model = model_dir / "dkt_model_latest.pt"
    if latest_model.exists():
        success(f"DKT model exists: {latest_model}. Skipping retraining.")
        return

    torch_probe = subprocess.run(
        [str(PYTHON_EXE), "-c", "import torch"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if torch_probe.returncode != 0:
        warning(
            "torch is not installed in the main virtual environment. "
            "Skipping DKT training so backend startup can continue. "
            "Install torch in .venv to enable DKT startup training."
        )
        return

    _run(
        [str(PYTHON_EXE), TRAIN_DKT_SCRIPT],
        "Training DKT model",
    )


def _ensure_emotion_model() -> None:
    """Ensure the complete Keras + TF.js emotion artifact set exists."""
    keras_model = ROOT / "backend" / "ai_models" / "emotion_model.h5"
    tfjs_dir = ROOT / "frontend" / "js" / "emotion_tfjs"
    tfjs_model = tfjs_dir / "model.json"
    metrics_info = ROOT / "backend" / "ai_models" / "emotion_model_info.json"

    required_artifacts = [keras_model, metrics_info, tfjs_model]
    if all(path.is_file() for path in required_artifacts):
        try:
            manifest = json.loads(tfjs_model.read_text(encoding="utf-8")).get("weightsManifest", [])
            shard_paths = [
                path
                for group in manifest
                if isinstance(group, dict)
                for path in group.get("paths", [])
            ]
            shards_ok = bool(shard_paths) and all((tfjs_dir / path).is_file() for path in shard_paths)
        except Exception:
            shards_ok = False

        if shards_ok:
            success(f"Emotion model artifacts exist: {tfjs_model}. Skipping retraining.")
            return

    info("Emotion model artifacts are incomplete. Retraining and rebuilding TF.js artifacts.")

    # The local trainer intentionally uses Windows + WSL for TF.js conversion.
    # Render has no WSL runtime; production should deploy the committed artifacts
    # or use the existing remote HF strict-training path instead.
    if _is_truthy_env("RENDER", "0") or os.environ.get("RENDER_SERVICE_ID"):
        raise RuntimeError(
            "Emotion model artifacts are missing on Render. "
            "Deploy the generated backend/ai_models and frontend/js/emotion_tfjs artifacts "
            "or run the remote HF strict-training workflow; local WSL conversion is not available on Render."
        )

    if not EMOTION_DATASET_DIR.exists():
        raise RuntimeError(
            f"Emotion dataset folder not found at {EMOTION_DATASET_DIR}."
        )

    missing_folders = [
        folder
        for folder in EMOTION_CLASS_FOLDERS
        if not (EMOTION_DATASET_DIR / folder).exists()
    ]
    if missing_folders:
        raise RuntimeError(
            "Emotion dataset is incomplete. Missing class folders: "
            + ", ".join(missing_folders)
        )

    _run(
        [str(PYTHON_EXE), TRAIN_EMOTION_SCRIPT],
        "Training CNN emotion model (TF.js export)",
    )


def _ensure_at_risk_model() -> None:
    """Train at-risk predictor artifact if missing."""
    model_dir = ROOT / "backend" / "models" / "at_risk_predictor"
    latest_manifest = model_dir / "latest_manifest.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    if latest_manifest.exists():
        success(f"At-risk model exists: {latest_manifest}. Skipping retraining.")
        return

    _run(
        [str(PYTHON_EXE), TRAIN_AT_RISK_SCRIPT],
        "Training at-risk predictor (Task 7)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    init_logger()

    if not PYTHON_EXE.exists():
        error("Virtual environment python not found.")
        return 1

    try:
        with stage("Database Migrations"):
            # Idempotent — Alembic no-ops when schema is already at head.
            # MUST run before any ORM query so all columns (is_disabled, etc.) exist.
            _run_migrations()

        _assert_gemini_key_if_required()

        with stage("Dependencies & Data"):
            users_before, questions_before = _read_counts()
            info(f"Current data status -> users: {users_before}, questions: {questions_before}")

            _ensure_users(users_before)
            _ensure_questions(questions_before)
            _ensure_interaction_dataset()

        with stage("Model Artifacts"):
            strict_ran = _run_strict_pipeline_if_enabled()
            if not strict_ran:
                _ensure_models_non_strict()

        users_after, questions_after = _read_counts()
        info(f"Final data status   -> users: {users_after}, questions: {questions_after}")
        return 0
    except Exception as exc:
        error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())