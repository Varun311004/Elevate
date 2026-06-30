"""Standalone FastAPI app for topic-based MCQ generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


AI_ROOT = Path(__file__).resolve().parent
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

load_dotenv(AI_ROOT / ".env", override=False)
load_dotenv(AI_ROOT.parent / ".env", override=False)

from mcq.generator import get_mcq_generator
from mcq.validator import MCQValidator


_MODEL_READY = False
_PRELOAD_STARTED = False
_PRELOAD_ERROR: Optional[str] = None
_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
_TRAINING_LOCK = threading.Lock()


DEFAULT_RAG_CHUNK_SIZE = 1200
DEFAULT_RAG_CHUNK_OVERLAP = 220
DEFAULT_RAG_EMBEDDING_DIM = 256
RAG_CHUNKING_STRATEGY = "section_sentence_window_v2"
RAG_EMBEDDING_MODEL = "hash-v2"


def _is_range_not_satisfiable_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "416" in message and "range" in message and "satisfiable" in message


def _snapshot_download_with_416_retry(*, repo_id: str, repo_type: str, local_dir: Path, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            token=token,
        )
        return
    except Exception as exc:
        if not _is_range_not_satisfiable_error(exc):
            raise
        print("[hf-train-runner] WARN: got HTTP 416 while prefetching dataset; retrying clean + force_download.")

    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        token=token,
        force_download=True,
    )

_EMBEDDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
    "this", "these", "those", "into", "their", "there", "than", "then", "them",
    "can", "may", "might", "should", "would", "will", "about", "after", "before",
    "during", "within", "without", "your", "you", "our", "its",
}


def _normalize_document_text(raw: str | None) -> str:
    text = str(raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"\u0000", "", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _approximate_token_count(text: str | None) -> int:
    words = re.findall(r"\S+", str(text or ""))
    if not words:
        return 0
    return max(1, int(round(len(words) * 1.2)))


def _normalize_vector(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    return [float(v / norm) for v in values]


def _tokenize_for_embedding(text: str | None, *, max_tokens: int = 900) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)?", str(text or "").lower())
    if not tokens:
        return []
    return tokens[: max(64, int(max_tokens or 900))]


def _weighted_hash_update(vector: List[float], key: str, weight: float) -> None:
    if not vector:
        return
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(vector)
    sign = -1.0 if (digest[4] % 2) else 1.0
    bucket_gain = 1.0 + ((digest[5] / 255.0) * 0.15)
    vector[idx] += sign * float(weight) * bucket_gain


def _hash_embedding(text: str, dim: int = DEFAULT_RAG_EMBEDDING_DIM) -> List[float]:
    vector = [0.0] * max(32, int(dim or DEFAULT_RAG_EMBEDDING_DIM))
    tokens = _tokenize_for_embedding(text)

    if not tokens:
        seed = hashlib.sha256(str(text or "").encode("utf-8")).digest()
        vector[seed[0] % len(vector)] = 1.0
        return _normalize_vector(vector)

    token_counts: Dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    token_total = len(tokens)
    prefix_cutoff = max(1, int(token_total * 0.22))
    for index, token in enumerate(tokens):
        tf = token_counts.get(token, 1)
        token_len = min(18, len(token))
        stopword_penalty = 0.35 if token in _EMBEDDING_STOPWORDS else 1.0
        positional_gain = 1.18 if index < prefix_cutoff else 1.0
        weight = (0.7 + (token_len / 10.0)) * (1.0 + math.log1p(tf)) * stopword_penalty * positional_gain
        _weighted_hash_update(vector, f"u:{token}", weight)

        if index + 1 < token_total:
            bigram = f"{token}|{tokens[index + 1]}"
            _weighted_hash_update(vector, f"b:{bigram}", 0.68 * stopword_penalty)

    gram_budget = 1600
    for token, tf in token_counts.items():
        if gram_budget <= 0:
            break
        if token in _EMBEDDING_STOPWORDS or len(token) < 5:
            continue

        padded = f"^{token}$"
        local_budget = 0
        for gram_size in (3, 4):
            if len(padded) < gram_size:
                continue
            for offset in range(0, len(padded) - gram_size + 1):
                gram = padded[offset: offset + gram_size]
                gram_weight = 0.25 * (1.0 + math.log1p(tf))
                _weighted_hash_update(vector, f"c:{gram}", gram_weight)
                local_budget += 1
                gram_budget -= 1
                if local_budget >= 10 or gram_budget <= 0:
                    break
            if local_budget >= 10 or gram_budget <= 0:
                break

    return _normalize_vector(vector)


def _looks_like_heading(line: str) -> bool:
    candidate = re.sub(r"\s+", " ", str(line or "")).strip()
    if not candidate:
        return False
    if len(candidate) < 3 or len(candidate) > 150:
        return False
    if candidate.endswith((".", "?", "!", ";")):
        return False
    if candidate.startswith(("-", "*", "•")):
        return False

    words = candidate.split()
    if len(words) > 20:
        return False

    letters = [char for char in candidate if char.isalpha()]
    if len(letters) < 3:
        return False

    if re.match(r"^(chapter|section|part|unit)\b", candidate, flags=re.IGNORECASE):
        return True
    if re.match(r"^\d+(?:\.\d+){0,4}[\)\].:-]?\s+\S+", candidate):
        return True

    uppercase_ratio = sum(1 for char in letters if char.isupper()) / max(1, len(letters))
    return candidate.istitle() or uppercase_ratio >= 0.72 or candidate.endswith(":")


def _split_text_spans(text: str, *, max_chars: int) -> List[tuple[int, int]]:
    content = str(text or "")
    if not content:
        return []

    limit = max(180, int(max_chars or 240))
    spans: List[tuple[int, int]] = []
    break_markers = [
        ("\n\n", 0),
        ("\n", 0),
        (". ", 1),
        ("? ", 1),
        ("! ", 1),
        ("; ", 1),
        (": ", 1),
        (", ", 1),
        (" ", 0),
    ]

    start = 0
    total = len(content)
    while start < total:
        tentative_end = min(total, start + limit)
        end = tentative_end

        if tentative_end < total:
            min_breakpoint = start + int((tentative_end - start) * 0.55)
            for marker, offset in break_markers:
                candidate = content.rfind(marker, min_breakpoint, tentative_end)
                if candidate > start:
                    end = candidate + offset
                    break

        if end <= start:
            end = tentative_end

        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1

        if end > start:
            spans.append((start, end))

        if end >= total:
            break

        start = end

    return spans


def _iter_paragraph_blocks(normalized: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for match in re.finditer(r"\S(?:[\s\S]*?\S)?(?=\n\s*\n|$)", normalized):
        raw_text = match.group(0)
        if not raw_text:
            continue
        text = raw_text.strip()
        if not text:
            continue
        blocks.append(
            {
                "text": text,
                "char_start": int(match.start()),
                "char_end": int(match.end()),
            }
        )
    return blocks


def _build_deterministic_chunks(
    text: str,
    *,
    document_fingerprint: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    normalized = _normalize_document_text(text)
    if not normalized:
        return []

    chunk_size = max(300, min(int(chunk_size or DEFAULT_RAG_CHUNK_SIZE), 3000))
    overlap = max(0, min(int(overlap or DEFAULT_RAG_CHUNK_OVERLAP), int(chunk_size / 2)))

    paragraph_blocks = _iter_paragraph_blocks(normalized)
    if not paragraph_blocks:
        return []

    units: List[Dict[str, Any]] = []
    section_title = "Document"
    section_index = 0
    paragraph_index = 0
    max_unit_chars = max(220, min(int(chunk_size * 0.72), 900))

    for block in paragraph_blocks:
        paragraph_text = str(block.get("text") or "").strip()
        if not paragraph_text:
            continue

        if _looks_like_heading(paragraph_text):
            section_index += 1
            section_title = paragraph_text[:160]
            continue

        spans = _split_text_spans(paragraph_text, max_chars=max_unit_chars)
        if not spans:
            spans = [(0, len(paragraph_text))]

        paragraph_start = int(block.get("char_start") or 0)
        for unit_idx, (local_start, local_end) in enumerate(spans):
            unit_text = paragraph_text[local_start:local_end].strip()
            if not unit_text:
                continue
            units.append(
                {
                    "text": unit_text,
                    "char_start": paragraph_start + local_start,
                    "char_end": paragraph_start + local_end,
                    "section_index": section_index,
                    "section_title": section_title,
                    "paragraph_index": paragraph_index,
                    "unit_index": unit_idx,
                }
            )

        paragraph_index += 1

    if not units:
        # If only heading-like text exists, ingest the whole document as one chunk.
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(
            f"{document_fingerprint}:0:{text_hash}".encode("utf-8")
        ).hexdigest()[:40]
        return [
            {
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "text": normalized,
                "text_hash": text_hash,
                "char_start": 0,
                "char_end": len(normalized),
                "token_count": _approximate_token_count(normalized),
                "metadata": {
                    "chunking_strategy": RAG_CHUNKING_STRATEGY,
                    "section_index": 0,
                    "section_title": "Document",
                    "paragraph_start_index": 0,
                    "paragraph_end_index": 0,
                    "paragraph_count": 1,
                    "relative_start": 0.0,
                    "relative_end": 1.0,
                    "unit_count": 1,
                },
            }
        ]

    chunks: List[Dict[str, Any]] = []
    total_chars = len(normalized)
    cursor = 0
    index = 0

    while cursor < len(units):
        start_cursor = cursor
        chunk_units: List[Dict[str, Any]] = []
        chunk_chars = 0

        while cursor < len(units):
            unit = units[cursor]
            unit_text = str(unit.get("text") or "")
            if not unit_text:
                cursor += 1
                continue

            separator = 2 if chunk_units else 0
            projected = chunk_chars + separator + len(unit_text)
            if chunk_units and projected > chunk_size:
                break

            chunk_units.append(unit)
            chunk_chars = projected
            cursor += 1

            if chunk_chars >= int(chunk_size * 0.88):
                break

        if not chunk_units:
            cursor = min(len(units), cursor + 1)
            continue

        snippet = "\n\n".join(str(unit.get("text") or "").strip() for unit in chunk_units).strip()
        if snippet:
            char_start = int(chunk_units[0].get("char_start") or 0)
            char_end = int(chunk_units[-1].get("char_end") or char_start)
            text_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{document_fingerprint}:{index}:{text_hash}".encode("utf-8")
            ).hexdigest()[:40]

            section_titles: List[str] = []
            for unit in chunk_units:
                title = str(unit.get("section_title") or "").strip()
                if title and title not in section_titles:
                    section_titles.append(title)

            paragraph_ids = {
                int(unit.get("paragraph_index") or 0)
                for unit in chunk_units
            }
            metadata = {
                "chunking_strategy": RAG_CHUNKING_STRATEGY,
                "section_index": int(chunk_units[0].get("section_index") or 0),
                "section_title": section_titles[0] if section_titles else "Document",
                "section_titles": section_titles[:4],
                "paragraph_start_index": int(chunk_units[0].get("paragraph_index") or 0),
                "paragraph_end_index": int(chunk_units[-1].get("paragraph_index") or 0),
                "paragraph_count": len(paragraph_ids),
                "relative_start": round(char_start / max(1, total_chars), 4),
                "relative_end": round(char_end / max(1, total_chars), 4),
                "unit_count": len(chunk_units),
            }

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "text": snippet,
                    "text_hash": text_hash,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_count": _approximate_token_count(snippet),
                    "metadata": metadata,
                }
            )
            index += 1

        if cursor >= len(units):
            break

        if overlap > 0:
            overlap_chars = 0
            rewind = cursor
            while rewind > start_cursor and overlap_chars < overlap:
                rewind -= 1
                overlap_chars += len(str(units[rewind].get("text") or "")) + 2
            cursor = max(start_cursor + 1, rewind)

    return chunks


def _ensure_generator_loaded():
    return get_mcq_generator()


def _preload_model_blocking() -> None:
    global _MODEL_READY, _PRELOAD_STARTED, _PRELOAD_ERROR
    _PRELOAD_STARTED = True
    started = time.perf_counter()
    try:
        generator = _ensure_generator_loaded()
        if hasattr(generator, "ensure_model_loaded"):
            generator.ensure_model_loaded()
        _MODEL_READY = bool(getattr(generator, "is_model_loaded", lambda: True)())
        if not _MODEL_READY:
            raise RuntimeError("Model failed to report ready state after preload.")

        _PRELOAD_ERROR = None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"[topic-mcq] model preload completed in {elapsed_ms}ms")
    except Exception as exc:
        _MODEL_READY = False
        _PRELOAD_ERROR = str(exc)
        print(f"[topic-mcq] preload failed: {exc}")
        raise


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Do not block HTTP readiness on MCQ LLM preload. Hugging Face Spaces can take minutes
    # to load the generator; the strict training endpoints must accept requests immediately.
    def _bg_preload() -> None:
        try:
            _preload_model_blocking()
        except Exception:
            pass

    threading.Thread(target=_bg_preload, daemon=True, name="mcq-model-preload").start()
    yield


app = FastAPI(title="Elevate Topic MCQ Service", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root() -> dict:
    if not _MODEL_READY:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    return {
        "status": "ok",
        "service": "topic-mcq",
        "message": "Service is running",
        "health_endpoint": "/health",
        "generate_endpoint": "/mcq/generate",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def _get_required_service_token() -> str:
    token = (
        os.environ.get("AI_TOPIC_SERVICE_TOKEN")
        or os.environ.get("TOPIC_AI_SERVICE_TOKEN")
        or ""
    )
    return str(token).strip()


def _get_auth_scheme() -> str:
    scheme = os.environ.get("AI_TOPIC_SERVICE_AUTH_SCHEME", "Bearer")
    return str(scheme or "Bearer").strip() or "Bearer"


def _enforce_service_auth(authorization_header: Optional[str]) -> None:
    expected_token = _get_required_service_token()
    if not expected_token:
        return

    if not authorization_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")

    provided = str(authorization_header).strip()
    expected_prefix = f"{_get_auth_scheme()} "
    if provided.lower().startswith(expected_prefix.lower()):
        provided = provided[len(expected_prefix):].strip()

    if provided != expected_token:
        raise HTTPException(status_code=401, detail="Invalid service token.")


class GenerateMCQRequest(BaseModel):
    source_type: str = Field(default="topic")
    source: str = Field(..., min_length=1)
    num_questions: int = Field(default=5, ge=1, le=50)
    difficulty: str = Field(default="medium")
    subject: str = Field(default="science")
    grade: str = Field(default="high")
    seed: Optional[int] = None
    test_title: Optional[str] = None
    test_description: Optional[str] = None


class OptimizePromptRequest(BaseModel):
    source_type: str = Field(default="topic")
    source: str = Field(..., min_length=1)
    num_questions: int = Field(default=5, ge=1, le=50)
    difficulty: str = Field(default="medium")
    subject: str = Field(default="science")
    grade: str = Field(default="high")
    generation_mode: str = Field(default="standard")
    test_title: Optional[str] = None
    test_description: Optional[str] = None
    rag_context: Optional[str] = None


class ScoreMCQRequest(BaseModel):
    mcqs: List[Dict]
    user_answers: Dict[int, str]


class StrictTrainingRequest(BaseModel):
    github_repo_url: Optional[str] = None
    github_ref: str = Field(default="main", min_length=1)
    min_emotion_accuracy: float = Field(default=0.88, ge=0.50, le=0.999)
    process_count: int = Field(default=4, ge=1, le=16)


class RagProcessRequest(BaseModel):
    document_text: str = Field(..., min_length=1)
    content_sha256: Optional[str] = None
    chunk_size: int = Field(default=DEFAULT_RAG_CHUNK_SIZE, ge=300, le=3000)
    overlap: int = Field(default=DEFAULT_RAG_CHUNK_OVERLAP, ge=0, le=1500)
    vector_store_requested: str = Field(default="python")
    embedding_dim: int = Field(default=DEFAULT_RAG_EMBEDDING_DIM, ge=32, le=4096)


def _sanitize_prompt_text(raw: str | None, *, max_length: int = 24000) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    return text[:max_length].strip()


def _build_prompt_optimizer_fallback(payload: OptimizePromptRequest) -> str:
    clean_rag = str(payload.rag_context or "").strip()
    rag_block = ""
    if clean_rag:
        rag_block = (
            "RAG CONTEXT (must be used as primary factual grounding when relevant):\n"
            f"{clean_rag[:12000]}\n\n"
        )

    return (
        "You are an expert STEM assessment designer. Return only JSON with this shape: "
        "{\"questions\":[{\"text\":string,\"options\":[string,string,string,string],\"correct_index\":0-3,\"explanation\":string,\"topic\":string}]}.\n"
        f"Generation mode: {payload.generation_mode}.\n"
        f"Subject: {payload.subject}. Grade: {payload.grade}. Difficulty: {payload.difficulty}.\n"
        f"Topic/Sub-topic: {payload.source}.\n"
        f"Question count: {payload.num_questions}.\n"
        f"Test title: {payload.test_title or 'N/A'}.\n"
        f"Test description: {payload.test_description or 'N/A'}.\n"
        f"{rag_block}"
        "Rules: exactly requested count, exactly four distinct options, one correct answer, concise instructional explanations, no markdown, no extra keys."
    )


@app.get("/health")
def health() -> dict:
    generator = _ensure_generator_loaded()
    model_ready = bool(getattr(generator, "is_model_loaded", lambda: _MODEL_READY)())

    model_device = None
    llm = getattr(generator, "llm", None)
    if model_ready and llm is not None:
        model_device = str(getattr(llm, "device", "cpu"))

    payload = {
        "status": "ok",
        "service": "topic-mcq",
        "model_ready": model_ready,
        "preload_started": _PRELOAD_STARTED,
        "preload_error": _PRELOAD_ERROR,
        "device": model_device,
    }
    if not model_ready:
        payload["status"] = "starting"
        return JSONResponse(status_code=503, content=payload)
    return payload

def _run_strict_training_job(job_id: str, request_payload: StrictTrainingRequest) -> None:
    # 1. Best-effort dataset prefetch. The training runner also prepares dataset in repo/dataset.
    try:
        print("[hf-train-runner] Downloading dataset from Hugging Face...")
        _snapshot_download_with_416_retry(
            repo_id="Sana2704/elevate-emotion-dataset",
            repo_type="dataset",
            local_dir=(AI_ROOT / "dataset"),
            token=os.environ.get("AI_TOPIC_SERVICE_TOKEN"),
        )
    except Exception as e:
        print(f"[hf-train-runner] WARNING: dataset prefetch failed in service process: {e}")

    # 2. Run training command
    command = [
        sys.executable,
        "-u", 
        str(AI_ROOT / "training_runner.py"),
        "--repo-url",
        str(request_payload.github_repo_url or os.environ.get("HF_ML_TRAINING_GITHUB_REPO_URL") or "").strip(),
        "--repo-ref",
        str(request_payload.github_ref or "main").strip(),
        "--min-emotion-accuracy",
        str(float(request_payload.min_emotion_accuracy)),
        "--processes",
        str(int(request_payload.process_count)),
    ]

    started = time.perf_counter()
    full_output = []

    with _TRAINING_LOCK:
        _TRAINING_JOBS[job_id]["status"] = "running"
        _TRAINING_JOBS[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        # Popen allows us to read line-by-line in real-time
        process = subprocess.Popen(
            command,
            cwd=str(AI_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout for easier logging
            text=True,
            bufsize=1 # Line buffered
        )

        # Read output in real-time
        for line in process.stdout:
            full_output.append(line)
            # Optional: print to HF console too
            print(line, end="") 
            # Periodically sync "tail" to memory so Render can see it while running
            with _TRAINING_LOCK:
                _TRAINING_JOBS[job_id]["stdout_tail"] = "".join(full_output[-100:])

        process.wait()
        duration_ms = max(0, int((time.perf_counter() - started) * 1000))

        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            _TRAINING_JOBS[job_id]["duration_ms"] = duration_ms
            _TRAINING_JOBS[job_id]["return_code"] = process.returncode
            _TRAINING_JOBS[job_id]["status"] = "succeeded" if process.returncode == 0 else "failed"
            _TRAINING_JOBS[job_id]["stdout_tail"] = "".join(full_output[-200:])

    except Exception as exc:
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id]["status"] = "failed"
            _TRAINING_JOBS[job_id]["error"] = str(exc)
            _TRAINING_JOBS[job_id]["stdout_tail"] = "".join(full_output[-200:])


def _find_running_training_job_id() -> Optional[str]:
    for job_id, payload in _TRAINING_JOBS.items():
        if str(payload.get("status") or "").lower() == "running":
            return job_id
    return None


@app.post("/training/strict/start")
def training_strict_start(
    payload: StrictTrainingRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict:
    _enforce_service_auth(authorization)

    with _TRAINING_LOCK:
        running_job = _find_running_training_job_id()
        if running_job:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "running",
                    "job_id": running_job,
                    "message": "A strict training job is already running.",
                },
            )

        repo_url = str(payload.github_repo_url or os.environ.get("HF_ML_TRAINING_GITHUB_REPO_URL") or "").strip()
        if not repo_url:
            raise HTTPException(
                status_code=400,
                detail="github_repo_url is required (either payload field or HF_ML_TRAINING_GITHUB_REPO_URL).",
            )

        job_id = uuid4().hex
        _TRAINING_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": payload.model_dump(),
        }

    worker = threading.Thread(target=_run_strict_training_job, args=(job_id, payload), daemon=True)
    worker.start()

    return {
        "status": "queued",
        "job_id": job_id,
        "status_endpoint": f"/training/strict/status/{job_id}",
    }


@app.get("/training/strict/status/{job_id}")
def training_strict_status(
    job_id: str,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict:
    _enforce_service_auth(authorization)
    with _TRAINING_LOCK:
        job = _TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown training job id")
        return dict(job)


@app.post("/mcq/generate")
def generate_mcqs(
    payload: GenerateMCQRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict:
    global _MODEL_READY
    _enforce_service_auth(authorization)
    if payload.source_type.lower() != "topic":
        raise HTTPException(status_code=400, detail="Only source_type='topic' is supported.")

    started = time.perf_counter()
    print(
        "[topic-mcq] generate request "
        f"topic={payload.source} subject={payload.subject} grade={payload.grade} "
        f"difficulty={payload.difficulty} count={payload.num_questions}"
    )

    generator = _ensure_generator_loaded()
    if not bool(getattr(generator, "is_model_loaded", lambda: _MODEL_READY)()):
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please retry shortly.")

    try:
        mcqs = generator.generate_from_topic(
            topic=payload.source,
            num_questions=payload.num_questions,
            difficulty=payload.difficulty,
            subject=payload.subject,
            grade=payload.grade,
            seed=payload.seed,
            test_title=payload.test_title,
            test_description=payload.test_description,
        )
        _MODEL_READY = bool(getattr(generator, "is_model_loaded", lambda: _MODEL_READY)())
    except Exception as exc:
        # Log the full error for debugging, but raise a generic one to the client
        print(f"[topic-mcq] generation failed with exception: {exc}")
        raise HTTPException(status_code=500, detail=f"MCQ generation failed internally.")

    # The new parser in the generator is now the source of truth for validation.
    # If it returns an empty list, we pass it on. The backend will handle the fallback.
    valid_mcqs = mcqs

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    meta = getattr(generator, "last_generation_meta", {}) or {}
    print(
        f"[topic-mcq] generate response "
        f"count={len(valid_mcqs)} latency_ms={elapsed_ms} "
        f"llm_count={meta.get('llm_count')} "
        f"cache_hit={meta.get('cache_hit')}"
    )

    return {
        "source_type": "topic",
        "source": payload.source,
        "count": len(valid_mcqs),
        "difficulty": payload.difficulty,
        "subject": payload.subject,
        "grade": payload.grade,
        "mcqs": valid_mcqs,
        "meta": {
            **meta,
            "latency_ms": elapsed_ms,
        },
    }


@app.post("/prompt/optimize")
def optimize_prompt_for_gemini(
    payload: OptimizePromptRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict:
    _enforce_service_auth(authorization)

    generator = _ensure_generator_loaded()
    if not bool(getattr(generator, "is_model_loaded", lambda: _MODEL_READY)()):
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please retry shortly.")

    fallback_prompt = _build_prompt_optimizer_fallback(payload)
    optimizer_prompt = (
        "Create one high-performance Gemini MCQ generation prompt for a STEM assessment request.\n"
        "Output ONLY the final prompt text (no markdown, no explanation).\n\n"
        "The prompt you produce must force these guarantees:\n"
        "1) strict JSON output with exact schema\n"
        "2) uses all user-provided fields\n"
        "3) grounded use of any RAG context\n"
        "4) exact requested count\n"
        "5) strong, plausible distractors and concise explanations\n\n"
        f"Request payload:\n{json.dumps(payload.model_dump(), ensure_ascii=False)}\n\n"
        f"Safe fallback template:\n{fallback_prompt}"
    )

    try:
        llm = getattr(generator, "llm", None)
        if llm is None:
            raise RuntimeError("LLM instance is not initialized")

        optimized = llm.generate(
            prompt=optimizer_prompt,
            max_new_tokens=900,
            temperature=0.2,
            top_p=0.95,
            max_time=20.0,
        )
        optimized_prompt = _sanitize_prompt_text(optimized)
        if not optimized_prompt:
            optimized_prompt = fallback_prompt

        return {
            "status": "ok",
            "source": "hf_local_prompt_optimizer",
            "optimized_prompt": optimized_prompt,
        }
    except Exception as exc:
        return {
            "status": "fallback",
            "source": "deterministic_fallback",
            "optimized_prompt": fallback_prompt,
            "error": str(exc),
        }


@app.post("/mcq/score")
def score_mcqs(payload: ScoreMCQRequest) -> dict:
    return MCQValidator.score_answers(payload.mcqs, payload.user_answers)


@app.post("/rag/process")
def rag_process_document(
    payload: RagProcessRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict:
    _enforce_service_auth(authorization)

    normalized_text = _normalize_document_text(payload.document_text)
    if not normalized_text:
        raise HTTPException(status_code=400, detail="document_text is empty after normalization")

    fingerprint = re.sub(r"[^a-f0-9]", "", str(payload.content_sha256 or "").lower())[:64]
    if not fingerprint:
        fingerprint = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    chunk_size = max(300, min(int(payload.chunk_size or DEFAULT_RAG_CHUNK_SIZE), 3000))
    overlap = max(0, min(int(payload.overlap or DEFAULT_RAG_CHUNK_OVERLAP), int(chunk_size / 2)))
    embedding_dim = max(32, min(int(payload.embedding_dim or DEFAULT_RAG_EMBEDDING_DIM), 4096))

    raw_chunks = _build_deterministic_chunks(
        normalized_text,
        document_fingerprint=fingerprint,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="No chunks could be produced from document_text")

    vector_store_requested = str(payload.vector_store_requested or "python").strip().lower() or "python"
    vector_store_effective = "python"
    vector_store_fallback_reason = None
    if vector_store_requested != "python":
        vector_store_fallback_reason = "ai_service_python_vector_only"

    total_tokens = 0
    chunks: List[Dict[str, Any]] = []
    for chunk in raw_chunks:
        text = str(chunk.get("text") or "")
        token_count = int(chunk.get("token_count") or _approximate_token_count(text))
        total_tokens += token_count

        raw_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        section_title = str(raw_metadata.get("section_title") or "").strip()
        embedding_input = text
        if section_title and section_title.lower() not in text.lower():
            embedding_input = f"{section_title}\n{text}"

        vector = _hash_embedding(embedding_input, dim=embedding_dim)
        chunk_metadata = {
            **raw_metadata,
            "source": "hf_rag_processor",
            "chunking_strategy": raw_metadata.get("chunking_strategy") or RAG_CHUNKING_STRATEGY,
        }

        chunks.append(
            {
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "text": text,
                "text_hash": str(chunk.get("text_hash") or hashlib.sha256(text.encode("utf-8")).hexdigest()),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "token_count": token_count,
                "embedding_vector": vector,
                "embedding_dim": len(vector),
                "embedding_provider": "hash-local",
                "embedding_model": RAG_EMBEDDING_MODEL,
                "embedding_status": "embedded",
                "metadata": chunk_metadata,
            }
        )

    return {
        "status": "ok",
        "source": "hf_rag_processor",
        "chunking_strategy": RAG_CHUNKING_STRATEGY,
        "embedding_model": RAG_EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "token_count": int(total_tokens),
        "vector_store_requested": vector_store_requested,
        "vector_store_effective": vector_store_effective,
        "vector_store_fallback_reason": vector_store_fallback_reason,
        "chunks": chunks,
    }


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
