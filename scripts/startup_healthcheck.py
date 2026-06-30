"""One-command startup bootstrap for Elevate.

This script is intentionally idempotent:
- Runs Alembic migrations before any DB access (auto-applies new columns).
- Seeds users only when missing.
- Ensures question bank size is healthy.
- Supports optional strict rebuild mode via env var.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
_APP_CACHE = None


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
        print(f"[BOOTSTRAP] Warning: could not load .env: {exc}")


def _run_migrations() -> None:
    """Apply all pending Alembic migrations against the configured database.

    Uses run_migrations.py which calls Alembic Python API directly — no
    Flask CLI needed. Is idempotent when schema is already at head.
    Must run BEFORE any SQLAlchemy model query so columns exist in Postgres.
    """
    _load_dotenv()

    migration_script = ROOT / "run_migrations.py"
    if not migration_script.exists():
        print("[BOOTSTRAP] Warning: run_migrations.py not found — skipping auto-migration.")
        return

    print("[BOOTSTRAP] Running database migrations (alembic upgrade head)...")
    result = subprocess.run(
        [str(PYTHON_EXE), str(migration_script)],
        cwd=ROOT,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Database migration failed. "
            "Check your DATABASE_URL in .env and that the PostgreSQL server is reachable."
        )
    print("[BOOTSTRAP] Migrations applied successfully.")


def _run(command: list[str], description: str) -> None:
    print(f"[BOOTSTRAP] {description}")
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


def _bootstrap_pip() -> None:
    subprocess.run([str(PYTHON_EXE), "-m", "ensurepip", "--upgrade"], cwd=ROOT)
    probe = subprocess.run([str(PYTHON_EXE), "-m", "pip", "--version"], cwd=ROOT)
    if probe.returncode != 0:
        raise RuntimeError("pip bootstrap failed in virtual environment")


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
        print(f"[BOOTSTRAP] Users present: {user_count}. Skipping user seed.")
        return

    _run([str(PYTHON_EXE), "seed_users.py"], "Seeding default users")


def _ensure_dependencies() -> None:
    """Install required packages if backend imports fail due missing modules."""
    try:
        global _APP_CACHE
        _APP_CACHE = None
        _read_counts()
        return
    except Exception as exc:
        message = str(exc)
        recoverable = (
            isinstance(exc, ModuleNotFoundError)
            or isinstance(exc, ImportError)
            or "No module named" in message
            or "cannot import name 'Flask' from 'flask'" in message
        )

        if not recoverable:
            raise

        print(f"[BOOTSTRAP] Dependency issue detected: {message}")
        _bootstrap_pip()

        _run(
            [
                str(PYTHON_EXE),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            "Upgrading pip tooling for dependency recovery",
        )

        runtime_packages = [
            "flask",
            "flask_sqlalchemy",
            "flask_cors",
            "python-dotenv",
            "PyJWT",
            "werkzeug",
            "bleach",
            "alembic",
            "psycopg2-binary",
            "numpy",
            "scikit-learn",
            "scipy",
            "shap",
            "joblib",
        ]
        _run(
            [
                str(PYTHON_EXE),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                *runtime_packages,
            ],
            "Installing runtime backend dependencies",
        )

        # Verify that imports now resolve correctly.
        _read_counts()


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
        print(
            f"[BOOTSTRAP] Question bank healthy: {question_count} questions (target >= {min_questions})."
        )
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
        "scripts/build_interaction_dataset.py",
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
            "scripts/build_interaction_dataset.py",
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
            "scripts/train_strict_pipeline.py",
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

    print("[BOOTSTRAP] Triggering strict HF training via backend service client ...")
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
    print(f"[BOOTSTRAP] HF strict training succeeded. summary={summary}")


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
        print(f"[BOOTSTRAP] BKT model exists: {latest_model}. Skipping retraining.")
        return

    _run(
        [str(PYTHON_EXE), "scripts/fit_bkt_model.py"],
        "Training BKT EM parameters on interaction dataset",
    )


def _ensure_dkt_model() -> None:
    """Train Deep Knowledge Tracing model if it doesn't exist."""
    model_dir = ROOT / "backend" / "models" / "dkt"
    model_dir.mkdir(parents=True, exist_ok=True)

    latest_model = model_dir / "dkt_model_latest.pt"
    if latest_model.exists():
        print(f"[BOOTSTRAP] DKT model exists: {latest_model}. Skipping retraining.")
        return

    _run(
        [str(PYTHON_EXE), "scripts/train_dkt_model.py"],
        "Training Deep Knowledge Tracing model on interaction dataset",
    )


def _ensure_emotion_model() -> None:
    """Build TFJS emotion model artifact when missing."""
    tfjs_model = ROOT / "frontend" / "js" / "emotion_tfjs" / "model.json"
    metrics_info = ROOT / "backend" / "ai_models" / "emotion_model_info.json"
    if tfjs_model.exists() and metrics_info.exists():
        print(f"[BOOTSTRAP] Emotion TFJS model exists: {tfjs_model}. Skipping retraining.")
        return

    _run(
        [str(PYTHON_EXE), "scripts/train_emotion_fast.py"],
        "Training HOG+MLP emotion model (TF.js export)",
    )


def _ensure_at_risk_model() -> None:
    """Train at-risk predictor artifact if missing."""
    model_dir = ROOT / "backend" / "models" / "at_risk_predictor"
    latest_manifest = model_dir / "latest_manifest.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    if latest_manifest.exists():
        print(f"[BOOTSTRAP] At-risk model exists: {latest_manifest}. Skipping retraining.")
        return

    _run(
        [str(PYTHON_EXE), "scripts/train_at_risk_predictor.py"],
        "Training at-risk predictor (Task 7)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if not PYTHON_EXE.exists():
        print("[BOOTSTRAP] Virtual environment python not found.")
        return 1

    try:
        # ── Step 1: Always migrate DB schema FIRST ────────────────────────────
        # Idempotent — Alembic no-ops when schema is already at head.
        # MUST run before any ORM query so all columns (is_disabled, etc.) exist.
        _run_migrations()

        _assert_gemini_key_if_required()
        _ensure_dependencies()
        users_before, questions_before = _read_counts()
        print(
            f"[BOOTSTRAP] Current data status -> users: {users_before}, questions: {questions_before}"
        )

        _ensure_users(users_before)
        _ensure_questions(questions_before)
        _ensure_interaction_dataset()
        strict_ran = _run_strict_pipeline_if_enabled()
        if not strict_ran:
            _ensure_models_non_strict()

        users_after, questions_after = _read_counts()
        print(
            f"[BOOTSTRAP] Final data status   -> users: {users_after}, questions: {questions_after}"
        )
        return 0
    except Exception as exc:
        print(f"[BOOTSTRAP] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
