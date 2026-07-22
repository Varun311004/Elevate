from __future__ import annotations

import time
import requests

from .config import (
    AI_HEALTH,
    BACKEND_HEALTH,
    FRONTEND_URL,
    BOOTSTRAP_TIMEOUT_SECONDS,
    HEALTH_POLL_INTERVAL,
)

from .logger import info, success, error, Spinner


def wait_many(targets: list[tuple[str, str]]) -> None:
    """
    Poll several health endpoints in a single loop, reporting each as it
    becomes ready, until all are ready or the shared timeout is reached.

    targets: list of (url, name) pairs.
    """

    pending = {name: url for url, name in targets}

    for name in pending:
        info(f"Waiting for {name}...")

    started = time.time()
    deadline = started + BOOTSTRAP_TIMEOUT_SECONDS
    spinner = Spinner("  Waiting")

    while pending and time.time() < deadline:

        for name, url in list(pending.items()):
            try:
                response = requests.get(url, timeout=2)
                if response.ok:
                    spinner.clear()
                    success(f"{name} is ready ({time.time() - started:.1f}s).")
                    del pending[name]
            except requests.RequestException:
                pass

        if pending:
            remaining = ", ".join(pending.keys())
            spinner.tick(f"({int(time.time() - started)}s elapsed) remaining: {remaining}")
            time.sleep(HEALTH_POLL_INTERVAL)

    if pending:
        spinner.clear()
        for name in pending:
            error(f"{name} failed to start within {BOOTSTRAP_TIMEOUT_SECONDS}s.")
        raise TimeoutError(", ".join(sorted(pending.keys())))


def wait(url: str, name: str) -> None:
    """Wait for a single endpoint. Thin wrapper around wait_many."""
    wait_many([(url, name)])


def wait_for_ai() -> None:
    wait(AI_HEALTH, "AI Service")


def wait_for_backend() -> None:
    wait(BACKEND_HEALTH, "Backend")


def wait_for_frontend() -> None:
    wait(FRONTEND_URL, "Frontend")


def wait_for_all_services() -> None:
    wait_many([
        (AI_HEALTH, "AI Service"),
        (BACKEND_HEALTH, "Backend"),
        (FRONTEND_URL, "Frontend"),
    ])