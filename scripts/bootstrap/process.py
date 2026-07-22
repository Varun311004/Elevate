from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Sequence


def run(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """
    Run a foreground process, inheriting stdout/stderr directly.
    Use when output doesn't need to be captured or tagged.
    """

    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=False,
    )

    return completed.returncode


def run_streaming(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """
    Run a foreground process, streaming its output line-by-line to on_line.

    Used when output needs to be tagged/prefixed — e.g. multiple processes
    running concurrently and sharing one console (see run_parallel below).
    """

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if on_line:
            on_line(line)
        else:
            print(line)

    return process.wait()


def run_parallel(jobs: dict[str, dict]) -> dict[str, int]:
    """
    Run several blocking jobs concurrently, one thread per job, each streamed
    through its own on_line callback so concurrent output stays attributable.

    jobs: {name: {"command": [...], "cwd": Path|None, "env": dict|None, "on_line": callable|None}}
    Returns {name: exit_code}
    """

    results: dict[str, int] = {}
    threads: list[threading.Thread] = []

    def _worker(name: str, spec: dict) -> None:
        results[name] = run_streaming(
            spec["command"],
            cwd=spec.get("cwd"),
            env=spec.get("env"),
            on_line=spec.get("on_line"),
        )

    for name, spec in jobs.items():
        thread = threading.Thread(target=_worker, args=(name, spec), daemon=True)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results


def start(
    command: Sequence[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """
    Start a background process in its own console window.
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