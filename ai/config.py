"""Configuration for the standalone AI topic MCQ service."""
from __future__ import annotations
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Create a BRAND NEW cache directory on the persistent volume.
# This acts as a clean wipe of your previous model data.
HF_HOME_DIR = Path("/data/elevate_models_v3")
HF_HOME_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = HF_HOME_DIR
os.environ["HF_HOME"] = str(HF_HOME_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HOME_DIR / "hub")

LLM_MODEL = os.environ.get("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LORA_ADAPTER_PATH = (os.environ.get("LORA_ADAPTER_PATH") or "").strip() or None

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    or ""
).strip()

if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)

# Strict behavior defaults
LLM_MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "8"))
LLM_TOTAL_TIME_BUDGET_SECONDS = float(os.environ.get("LLM_TOTAL_TIME_BUDGET_SECONDS", "28"))

# Generation controls
MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", "1200"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "280"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
TOP_P = float(os.environ.get("TOP_P", "0.92"))
LLM_GENERATE_MAX_TIME_SECONDS = float(os.environ.get("LLM_GENERATE_MAX_TIME_SECONDS", "8"))

# Context retrieval
WEB_CONTEXT_MAX_CHARS = int(os.environ.get("WEB_CONTEXT_MAX_CHARS", "3200"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "4"))

# Runtime tuning
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "3"))
FACT_SENTENCE_MIN_CHARS = int(os.environ.get("FACT_SENTENCE_MIN_CHARS", "55"))
CPU_LLM_MAX_ATTEMPTS = int(os.environ.get("CPU_LLM_MAX_ATTEMPTS", "3"))
CPU_LLM_MAX_NEW_TOKENS = int(os.environ.get("CPU_LLM_MAX_NEW_TOKENS", "140"))