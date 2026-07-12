from __future__ import annotations

from datetime import datetime
from enum import Enum


class Level(Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _log(level: Level, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level.value}] {message}")


def info(message: str) -> None:
    _log(Level.INFO, message)


def success(message: str) -> None:
    _log(Level.SUCCESS, message)


def warning(message: str) -> None:
    _log(Level.WARNING, message)


def error(message: str) -> None:
    _log(Level.ERROR, message)