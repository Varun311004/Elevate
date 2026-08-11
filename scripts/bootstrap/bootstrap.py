from __future__ import annotations

import sys
import time

from .config import (
    AI_DIR,
    AI_START_SCRIPT,
    PROJECT_ROOT,
    STARTUP_HEALTHCHECK,
    VENV_PYTHON,
    BACKEND_DIR,
    BACKEND_START_COMMAND,
    FRONTEND_DIR,
    FRONTEND_START_COMMAND,
    FRONTEND_URL,
)
from .logger import info, error, banner, stage, format_elapsed, track_printer, init as init_logger
from .process import run_parallel, start, clone_environment
from .readiness import wait_for_all_services
from .browser import open_application


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1 — Preflight
# Fail fast with a clear, actionable message if the environment itself isn't
# set up, before spending any time launching subprocesses that would only
# fail anyway.
# ─────────────────────────────────────────────────────────────────────────

def preflight() -> None:
    with stage("Preflight"):
        problems: list[str] = []

        if not VENV_PYTHON.exists():
            problems.append(
                f"Main virtual environment not found at {VENV_PYTHON}. "
                "Create it with: py -3.11 -m venv .venv"
            )
        if not AI_START_SCRIPT.exists():
            problems.append(f"AI start script not found at {AI_START_SCRIPT}.")
        if not STARTUP_HEALTHCHECK.exists():
            problems.append(f"Health check script not found at {STARTUP_HEALTHCHECK}.")
        if not BACKEND_DIR.exists():
            problems.append(f"Backend directory not found at {BACKEND_DIR}.")
        if not FRONTEND_DIR.exists():
            problems.append(f"Frontend directory not found at {FRONTEND_DIR}.")

        if problems:
            for problem in problems:
                error(problem)
            raise SystemExit(1)

        info("All required paths verified.")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — Parallel prep
# Health Check (DB migrations, seed data, ML artifacts) and AI Environment
# provisioning touch entirely different venvs and have no dependency on each
# other, so they run concurrently instead of back-to-back. Output from each
# is tagged so interleaved lines stay attributable to their source.
# ─────────────────────────────────────────────────────────────────────────

def run_parallel_prep() -> None:
    with stage("Health Check + AI Environment (parallel)"):
        shared_env = clone_environment(PYTHONUNBUFFERED="1", ELEVATE_FORCE_COLOR="1")

        jobs = {
            "HEALTH": {
                "command": [str(VENV_PYTHON), str(STARTUP_HEALTHCHECK)],
                "cwd": PROJECT_ROOT,
                "env": shared_env,
                "on_line": track_printer("HEALTH", 0),
            },
            "AI-ENV": {
                "command": [str(AI_START_SCRIPT), "--ensure-env"],
                "cwd": AI_DIR,
                "env": shared_env,
                "on_line": track_printer("AI-ENV", 1),
            },
        }

        results = run_parallel(jobs)
        failed = [name for name, code in results.items() if code != 0]
        if failed:
            raise RuntimeError(f"Failed track(s): {', '.join(failed)}")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — Launch services
# Frontend has no dependencies at all, so it's started before this section
# even begins (see main()). AI and backend only depend on the prep above,
# not on each other, so both are started here and all three are health
# checked together in one combined poll loop instead of one at a time.
# ─────────────────────────────────────────────────────────────────────────

def start_ai() -> None:
    info("Starting AI service...")
    env = clone_environment(PORT="7860")
    return start([str(AI_START_SCRIPT)], cwd=AI_DIR, env=env)


def start_backend() -> None:
    info("Starting backend...")
    env = clone_environment(PORT="5000")
    return start(BACKEND_START_COMMAND, cwd=BACKEND_DIR, env=env)


def start_frontend() -> None:
    info("Starting frontend...")
    return start(FRONTEND_START_COMMAND, cwd=FRONTEND_DIR)


def launch_services() -> None:
    with stage("Launch Services"):
        start_ai()
        start_backend()
        start_frontend()
        wait_for_all_services()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — Entry point
# Ties the sections together in order and reports total elapsed time.
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    init_logger()
    banner("ELEVATE", "Starting local development stack")
    overall_start = time.monotonic()

    try:
        preflight()

        # Zero dependencies — start now so it's warm well before anything
        # else needs it, instead of waiting until the very end.

        run_parallel_prep()
        launch_services()
        open_application()

    except SystemExit as exc:
        error("Startup aborted — see the failed stage above.")
        return exc.code if isinstance(exc.code, int) else 1
    except (TimeoutError, RuntimeError) as exc:
        error(f"Startup failed: {exc}")
        return 1

    total = format_elapsed(time.monotonic() - overall_start)
    banner("Elevate is ready", f"Total startup time: {total} — {FRONTEND_URL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())