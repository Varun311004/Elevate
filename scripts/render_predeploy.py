"""Render pre-deploy hook: normalize DB URL and apply Alembic migrations."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = ROOT / "alembic.ini"


def _get_raw_database_url() -> str:
    return (
        os.environ.get("DATABASE_URL")
        or os.environ.get("SUPABASE_DIRECT_CONNECTION_STRING")
        or ""
    ).strip()


def _normalize_and_export_database_url() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from backend.config import normalize_database_url  # Local import keeps script standalone.

    raw = _get_raw_database_url()
    if not raw:
        raise RuntimeError("DATABASE_URL is required for pre-deploy migrations.")

    normalized = normalize_database_url(raw)
    os.environ["DATABASE_URL"] = normalized
    return normalized


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}\n{output}"
        )


def _looks_like_legacy_missing_table_failure(error_text: str) -> bool:
    normalized = str(error_text or "").lower()
    needles = [
        "no such table",
        "undefinedtable",
        "relation",
        "does not exist",
        "already exists",
        "duplicate table",
    ]
    return any(needle in normalized for needle in needles)


def _bootstrap_schema_with_models() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from backend.app import create_app  # Local import to keep script startup light.

    print("[predeploy] Bootstrapping schema via Flask models (db.create_all).")
    create_app("production")


def main() -> int:
    try:
        normalized_url = _normalize_and_export_database_url()
        host = urlsplit(normalized_url).hostname or "sqlite"
        print(f"[predeploy] Running Alembic migrations against host={host}")

        upgrade_cmd = [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(ALEMBIC_INI),
            "upgrade",
            "head",
        ]

        try:
            _run(upgrade_cmd)
        except RuntimeError as exc:
            if not _looks_like_legacy_missing_table_failure(str(exc)):
                raise

            # Some historical migration chains in this project assume pre-existing tables.
            # For fresh deployments, bootstrap tables first, then stamp migration head.
            print("[predeploy] Alembic upgrade hit legacy missing-table path. Applying bootstrap fallback.")
            _bootstrap_schema_with_models()
            _run([
                sys.executable,
                "-m",
                "alembic",
                "-c",
                str(ALEMBIC_INI),
                "stamp",
                "head",
            ])

        print("[predeploy] Migrations applied successfully.")

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from backend.hf_training_service import trigger_and_wait_hf_strict_training

        print("[predeploy] Triggering strict HF training pipeline...")
        training_result = trigger_and_wait_hf_strict_training()
        if not training_result.get("ok"):
            raise RuntimeError(
                "Strict HF training failed: "
                f"status={training_result.get('status_code')} "
                f"error={training_result.get('error')}"
            )

        payload = training_result.get("payload") if isinstance(training_result.get("payload"), dict) else {}
        state = str(payload.get("status") or "").strip().lower()
        if state != "succeeded":
            raise RuntimeError(f"Strict HF training did not succeed (status={state or 'unknown'}).")

        print("[predeploy] Strict HF training completed successfully.")
        return 0
    except Exception as exc:
        print(f"[predeploy] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
