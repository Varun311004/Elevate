from __future__ import annotations

import sys

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
)
from .logger import info, error, success
from .process import run, start, clone_environment
from .readiness import wait_for_ai, wait_for_backend, wait_for_frontend
from .browser import open_application


def run_healthcheck() -> None:
    info("Running startup health check...")

    exit_code = run(
        [str(VENV_PYTHON), str(STARTUP_HEALTHCHECK)],
        cwd=PROJECT_ROOT,
    )

    if exit_code != 0:
        error("Startup health check failed.")
        raise SystemExit(exit_code)

    success("Startup health check completed.")


def ensure_ai_environment() -> None:
    info("Ensuring AI environment...")

    exit_code = run(
        [str(AI_START_SCRIPT), "--ensure-env"],
        cwd=AI_DIR,
    )

    if exit_code != 0:
        error("Unable to prepare AI environment.")
        raise SystemExit(exit_code)

    success("AI environment ready.")


def start_ai() -> None:
    info("Starting AI service...")

    env = clone_environment(PORT="7860")
    return start(
        [str(AI_START_SCRIPT)],
        cwd=AI_DIR,
        env=env,
    )


def start_backend() -> None:
    info("Starting backend...")

    env = clone_environment(PORT="5000")
    return start(
        BACKEND_START_COMMAND,
        cwd=BACKEND_DIR,
        env=env,
    )


def start_frontend() -> None:
    info("Starting frontend...")

    return start(
        FRONTEND_START_COMMAND,
        cwd=FRONTEND_DIR,
    )


def main() -> int:
    run_healthcheck()
    ensure_ai_environment()

    ai = start_ai()
    wait_for_ai()

    backend = start_backend()
    wait_for_backend()

    frontend = start_frontend()
    wait_for_frontend()

    open_application()
    
    success("Elevate started successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())