"""
run_migrations.py — standalone migration runner for Elevate.

Handles all DB states automatically:
  1. Brand-new DB: run full migration chain
  2. Existing DB fully populated but no alembic tracking: stamp baseline + upgrade
  3. Corrupt state (alembic_version stamped but core tables missing): reset + full run
  4. Healthy tracked DB: normal upgrade head

Usage:  .venv\\Scripts\\python.exe run_migrations.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# These tables MUST exist for the baseline stamp to be valid.
# If alembic_version says we're at a revision but these don't exist,
# the stamp is corrupt and we reset.
CORE_TABLES = {"users", "schools", "questions"}

# The last migration that matches the original schema built by db.create_all()
BASELINE_STAMP_REVISION = "f4d91c2e7b11"


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        print(f"[migrate] No .env at {env_file} — using system env.")
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
        print("[migrate] Loaded environment from .env")
    except Exception as exc:
        print(f"[migrate] Warning: could not load .env: {exc}")


def _make_alembic_cfg(db_url: str):
    from alembic.config import Config
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "backend" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
    return cfg


def _get_table_names(engine) -> set[str]:
    from sqlalchemy import inspect as sa_inspect
    try:
        return set(sa_inspect(engine).get_table_names())
    except Exception:
        return set()


def _get_current_revision(engine) -> str | None:
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
            row = result.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _clear_alembic_version(engine) -> None:
    """Delete all rows from alembic_version so we can re-run from scratch."""
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM alembic_version"))
        print("[migrate] Cleared alembic_version table.")
    except Exception as exc:
        print(f"[migrate] Warning: could not clear alembic_version: {exc}")


def _drop_orphan_tables(engine, tables: set[str]) -> None:
    """Drop tables that were created by partial migrations but are incomplete."""
    from sqlalchemy import text
    # Only drop tables we know are safe to recreate via migrations
    safe_to_drop = {"rag_retrieval_events", "teacher_document_chunks", "teacher_documents",
                    "teacher_interventions"}
    to_drop = tables & safe_to_drop
    if not to_drop:
        return
    try:
        with engine.begin() as conn:
            for t in to_drop:
                conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))
                print(f"[migrate] Dropped orphan table: {t}")
    except Exception as exc:
        print(f"[migrate] Warning: could not drop orphan tables: {exc}")


def main() -> int:
    _load_dotenv()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    db_url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
        or ""
    ).strip()

    if not db_url:
        print("[migrate] ERROR: DATABASE_URL not set.")
        return 1

    try:
        from backend.config import normalize_database_url
        db_url = normalize_database_url(db_url)
    except Exception:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)

    masked = db_url[:40] + "..." if len(db_url) > 40 else db_url
    print(f"[migrate] Database URL: {masked}")

    try:
        from alembic import command as alembic_command
        from sqlalchemy import create_engine
    except ImportError as exc:
        print(f"[migrate] ERROR: missing package — {exc}")
        return 1

    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.INFO)

    # ── Connect ───────────────────────────────────────────────────────────────
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect():
            pass
    except Exception as exc:
        print(f"[migrate] ERROR: Cannot connect to database: {exc}")
        return 1

    alembic_cfg = _make_alembic_cfg(db_url)

    # ── Inspect DB state ──────────────────────────────────────────────────────
    tables = _get_table_names(engine)
    has_alembic = "alembic_version" in tables
    current_rev = _get_current_revision(engine) if has_alembic else None
    has_core = CORE_TABLES.issubset(tables)

    print(f"[migrate] Tables found: {len(tables)}")
    print(f"[migrate] Core tables present: {has_core}")
    print(f"[migrate] Alembic revision: {current_rev or '(none)'}")

    # ── Decision logic ────────────────────────────────────────────────────────

    if current_rev and has_core:
        # HAPPY PATH: DB is tracked and has core tables — normal upgrade
        print(f"[migrate] Database is healthy at revision {current_rev}.")

    elif current_rev and not has_core:
        # CORRUPT STATE: stamped but core tables are missing
        # (This is exactly what happened — we stamped f4d91c2e7b11 but users doesn't exist)
        print(
            f"[migrate] CORRUPT STATE detected: alembic says {current_rev} "
            f"but core tables (users, schools, questions) are MISSING."
        )
        print("[migrate] Resetting alembic_version and dropping orphan tables...")
        _clear_alembic_version(engine)
        _drop_orphan_tables(engine, tables - {"alembic_version"})
        print("[migrate] Will run full migration chain from scratch.")

    elif not current_rev and has_core:
        # UNTRACKED DB: all tables exist but alembic never managed them
        # Stamp to baseline so only NEW migrations run
        print(
            f"[migrate] Untracked database: core tables exist but no alembic revision.\n"
            f"[migrate] Stamping baseline: {BASELINE_STAMP_REVISION}"
        )
        try:
            alembic_command.stamp(alembic_cfg, BASELINE_STAMP_REVISION)
            print("[migrate] Stamped. Now applying only new migrations...")
        except Exception as exc:
            print(f"[migrate] ERROR stamping: {exc}")
            return 1

    else:
        # EMPTY DB: nothing exists — run everything from scratch
        if has_alembic:
            _clear_alembic_version(engine)
        orphans = tables - {"alembic_version"}
        if orphans:
            _drop_orphan_tables(engine, orphans)
        print("[migrate] Empty database. Running full migration chain...")

    # ── Apply migrations ──────────────────────────────────────────────────────
    print("[migrate] Running: alembic upgrade head ...")
    try:
        alembic_command.upgrade(alembic_cfg, "head")
        print("[migrate] SUCCESS — database schema is up to date.")
        return 0
    except Exception as exc:
        err = str(exc)
        print(f"[migrate] FAILED: {err[:500]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
