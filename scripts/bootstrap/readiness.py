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

from .logger import (
    info,
    success,
    error,
)


def wait(url: str, name: str) -> None:

    info(f"Waiting for {name}...")

    deadline = time.time() + BOOTSTRAP_TIMEOUT_SECONDS

    while time.time() < deadline:

        try:

            response = requests.get(
                url,
                timeout=2,
            )

            if response.ok:

                success(f"{name} is ready.")

                return

        except requests.RequestException:
            pass

        time.sleep(HEALTH_POLL_INTERVAL)

    error(f"{name} failed to start.")

    raise TimeoutError(name)


def wait_for_ai():

    wait(
        AI_HEALTH,
        "AI Service",
    )


def wait_for_backend():

    wait(
        BACKEND_HEALTH,
        "Backend",
    )


def wait_for_frontend():

    wait(
        FRONTEND_URL,
        "Frontend",
    )