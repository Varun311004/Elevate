from __future__ import annotations

from pathlib import Path
import os

# ==========================================================
# Project Paths
# ==========================================================

BOOTSTRAP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BOOTSTRAP_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# ==========================================================
# Python Environments
# ==========================================================

VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"

AI_VENV_DIR = PROJECT_ROOT / ".venv-ai"
AI_VENV_PYTHON = AI_VENV_DIR / "Scripts" / "python.exe"

# ==========================================================
# Project Directories
# ==========================================================

BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
AI_DIR = PROJECT_ROOT / "ai"

MODELS_DIR = BACKEND_DIR / "models"

# ==========================================================
# Scripts
# ==========================================================

STARTUP_HEALTHCHECK = PROJECT_ROOT / "scripts" / "startup_healthcheck.py"

AI_START_SCRIPT = AI_DIR / "start.bat"

# ==========================================================
# Runtime
# ==========================================================

HOST = "127.0.0.1"

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8000"))
AI_PORT = int(os.getenv("AI_PORT", "7860"))

BACKEND_URL = f"http://{HOST}:{BACKEND_PORT}"
FRONTEND_URL = f"http://{HOST}:{FRONTEND_PORT}"
AI_URL = f"http://{HOST}:{AI_PORT}"

BACKEND_HEALTH = f"{BACKEND_URL}/health"
AI_HEALTH = f"{AI_URL}/health"

# ==========================================================
# Browser
# ==========================================================

DEFAULT_BROWSER_URL = FRONTEND_URL

# ==========================================================
# Bootstrap
# ==========================================================

BOOTSTRAP_TIMEOUT_SECONDS = 180

# Lowered from 1.0s so the readiness spinner feels responsive instead of
# appearing to hang for a full second between checks.
HEALTH_POLL_INTERVAL = 0.4

# ==========================================================
# Startup Commands
# ==========================================================

BACKEND_START_COMMAND = [
    str(VENV_PYTHON),
    str(BACKEND_DIR / "run.py"),
]

FRONTEND_START_COMMAND = [
    str(VENV_PYTHON),
    "-m",
    "http.server",
    "8000",
]

AI_START_COMMAND = [
    str(AI_START_SCRIPT),
]

FRONTEND_URL = f"http://{HOST}:8000"
DEFAULT_BROWSER_URL = FRONTEND_URL