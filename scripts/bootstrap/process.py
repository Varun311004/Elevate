from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence


def run(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """
    Run a foreground process.
    """

    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=False,
    )

    return completed.returncode


def start(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """
    Start a background process.
    """

    return subprocess.Popen(
    list(command),
    cwd=cwd,
    env=env,
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)


def windows_start(
    command: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """
    Launch through cmd/start while keeping a new console window.
    """

    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        shell=True,
    )


def clone_environment(**updates: str) -> dict[str, str]:
    """
    Create an isolated environment for a child process.
    """

    environment = os.environ.copy()

    for key, value in updates.items():
        environment[key] = value

    return environment