from __future__ import annotations

import webbrowser

from .config import DEFAULT_BROWSER_URL
from .logger import info


def open_application() -> None:
    info("Opening browser...")
    webbrowser.open(DEFAULT_BROWSER_URL, new=2)