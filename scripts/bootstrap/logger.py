from __future__ import annotations

import contextlib
import os
import sys
import time
from datetime import datetime
from enum import Enum


class _Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


class Level(Enum):
    INFO = ("INFO", _Color.CYAN, "i")
    SUCCESS = ("SUCCESS", _Color.GREEN, "+")
    WARNING = ("WARNING", _Color.YELLOW, "!")
    ERROR = ("ERROR", _Color.RED, "x")


_COLOR_ENABLED = False


def _enable_windows_ansi() -> bool:
    """Turn on VT100 escape processing for classic cmd.exe consoles.

    No-op (returns True) on non-Windows platforms, which already support ANSI.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_virtual_terminal_processing = 0x0004
        new_mode = mode.value | enable_virtual_terminal_processing
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except Exception:
        return False


def init() -> None:
    """Call once at process start (in every entry-point script) to enable color.

    Safe to call multiple times. Each Python process (this script, plus every
    subprocess it launches) has its own module state, so each entry point that
    wants colored output needs to call this itself.

    Honors ELEVATE_FORCE_COLOR=1 for subprocesses whose stdout is piped back
    to a parent process for tagging (see process.run_streaming) — their
    output isn't a real tty, but it ultimately lands on one, so color is safe.
    """
    global _COLOR_ENABLED
    forced = str(os.environ.get("ELEVATE_FORCE_COLOR", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    _COLOR_ENABLED = bool(forced or (is_tty and _enable_windows_ansi()))


def _c(code: str, text: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return f"{code}{text}{_Color.RESET}"


def _log(level: Level, message: str) -> None:
    name, color, symbol = level.value
    timestamp = datetime.now().strftime("%H:%M:%S")
    ts = _c(_Color.DIM, timestamp)
    tag = _c(color, f"[{symbol}]")
    print(f"{ts} {tag} {message}")


def info(message: str) -> None:
    _log(Level.INFO, message)


def success(message: str) -> None:
    _log(Level.SUCCESS, message)


def warning(message: str) -> None:
    _log(Level.WARNING, message)


def error(message: str) -> None:
    _log(Level.ERROR, message)


def banner(title: str, subtitle: str | None = None, width: int = 60) -> None:
    print()
    print(_c(_Color.MAGENTA, "=" * width))
    print(_c(_Color.BOLD, title.center(width)))
    if subtitle:
        print(_c(_Color.DIM, subtitle.center(width)))
    print(_c(_Color.MAGENTA, "=" * width))
    print()


def divider(width: int = 60) -> None:
    print(_c(_Color.DIM, "-" * width))


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


@contextlib.contextmanager
def stage(name: str):
    """Wrap one phase of startup with a header, timing, and a pass/fail footer.

    Usage:
        with stage("Health Check"):
            ...do the work...

    On success prints "<name> completed in Xs". On any exception prints
    "<name> failed after Xs" and re-raises, so callers keep normal control flow.
    """
    divider()
    print(_c(_Color.BOLD, f"STAGE: {name}"))
    divider()
    start = time.monotonic()
    try:
        yield
    except BaseException:
        elapsed = format_elapsed(time.monotonic() - start)
        error(f"{name} failed after {elapsed}")
        raise
    else:
        elapsed = format_elapsed(time.monotonic() - start)
        success(f"{name} completed in {elapsed}")


_TRACK_COLORS = [_Color.CYAN, _Color.YELLOW, _Color.GREEN, _Color.MAGENTA]


def track_printer(tag: str, color_index: int = 0):
    """Return a callable(line) that prints a line prefixed with a colored tag.

    Used when multiple subprocesses stream output concurrently (see
    process.run_parallel) so interleaved lines stay attributable to their source.
    """
    color = _TRACK_COLORS[color_index % len(_TRACK_COLORS)]
    prefix = _c(color, f"[{tag}]")

    def _print_line(line: str) -> None:
        if line.strip():
            print(f"{prefix} {line}")

    return _print_line


class Spinner:
    """In-place spinner for long waits (used by readiness.py)."""

    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, message: str):
        self.message = message
        self._i = 0
        self._active = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def tick(self, suffix: str = "") -> None:
        if not self._active:
            return
        frame = self.FRAMES[self._i % len(self.FRAMES)]
        self._i += 1
        text = f"\r{self.message} {frame} {suffix}".rstrip()
        sys.stdout.write(text)
        sys.stdout.flush()

    def clear(self) -> None:
        if not self._active:
            return
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()