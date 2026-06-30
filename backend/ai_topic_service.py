"""HTTP client for the standalone topic MCQ AI service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from typing import Any, Dict, List
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

from .validation import sanitize_string

import concurrent.futures


DEFAULT_TOPIC_AI_SERVICE_URL = "http://127.0.0.1:7860"
DEFAULT_TOPIC_AI_TIMEOUT_SECONDS = 180


def _to_hf_space_subdomain(owner: str, space: str) -> str:
    combined = f"{owner}-{space}".strip().lower().replace("_", "-")
    combined = re.sub(r"[^a-z0-9-]", "-", combined)
    combined = re.sub(r"-+", "-", combined).strip("-")
    return combined


def _normalize_topic_ai_service_url(raw_url: str) -> str:
    candidate = str(raw_url or "").strip()
    if not candidate:
        return DEFAULT_TOPIC_AI_SERVICE_URL

    # Allow compact "owner/space" format for convenience.
    if "://" not in candidate and candidate.count("/") == 1 and " " not in candidate:
        owner, space = [segment.strip() for segment in candidate.split("/", 1)]
        if owner and space:
            return f"https://{_to_hf_space_subdomain(owner, space)}.hf.space"

    if candidate.startswith("huggingface.co/"):
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    host = (parsed.netloc or "").strip().lower()

    if host in {"huggingface.co", "www.huggingface.co"}:
        segments = [segment for segment in (parsed.path or "").split("/") if segment]
        if len(segments) >= 3 and segments[0].lower() == "spaces":
            owner = segments[1]
            space = segments[2]
            return f"https://{_to_hf_space_subdomain(owner, space)}.hf.space"

    if host.endswith(".hf.space"):
        scheme = (parsed.scheme or "https").strip().lower()
        return f"{scheme}://{host}"

    return candidate.rstrip("/")


def get_topic_ai_service_url() -> str:
    raw_url = (
        os.environ.get("AI_TOPIC_SERVICE_URL")
        or os.environ.get("TOPIC_AI_SERVICE_URL")
        or DEFAULT_TOPIC_AI_SERVICE_URL
    )
    return _normalize_topic_ai_service_url(raw_url)


def get_topic_ai_timeout_seconds() -> int:
    raw_value = os.environ.get("AI_TOPIC_SERVICE_TIMEOUT_SECONDS", DEFAULT_TOPIC_AI_TIMEOUT_SECONDS)
    try:
        timeout = int(raw_value)
    except (TypeError, ValueError):
        timeout = DEFAULT_TOPIC_AI_TIMEOUT_SECONDS
    return max(5, min(timeout, 300))


def get_topic_ai_service_token() -> str:
    token = (
        os.environ.get("AI_TOPIC_SERVICE_TOKEN")
        or os.environ.get("TOPIC_AI_SERVICE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or ""
    )
    return str(token).strip()


def get_topic_ai_auth_scheme() -> str:
    scheme = os.environ.get("AI_TOPIC_SERVICE_AUTH_SCHEME", "Bearer")
    return str(scheme or "Bearer").strip() or "Bearer"


def is_topic_ai_service_available() -> bool:
    return bool(get_topic_ai_service_url())


def _derive_source_topic(topic: str | None, subject: str, grade: str) -> str:
    clean_topic = sanitize_string(topic or "", max_length=128) or ""
    if clean_topic:
        return clean_topic

    clean_subject = sanitize_string(subject or "", max_length=64) or "general"
    clean_grade = sanitize_string(grade or "", max_length=32) or "general"
    return f"{clean_subject} {clean_grade}".strip()


def _normalize_options(raw_options: Any) -> List[str]:
    options: List[str] = []
    if isinstance(raw_options, list):
        options = [sanitize_string(str(opt), max_length=300) for opt in raw_options]
    elif isinstance(raw_options, dict):
        ordered = []
        for key in ("A", "B", "C", "D", "a", "b", "c", "d"):
            value = raw_options.get(key)
            if value is not None:
                ordered.append(value)
        options = [sanitize_string(str(opt), max_length=300) for opt in ordered]

    clean_options: List[str] = []
    seen = set()
    for option in options:
        if not option:
            continue
        key = " ".join(option.strip().lower().split())
        if key in seen:
            continue
        seen.add(key)
        clean_options.append(option.strip())
        if len(clean_options) >= 4:
            break

    return clean_options


def _extract_correct_index(item: Dict[str, Any], options: List[str]) -> int:
    try:
        index = int(item.get("correct_index"))
        if 0 <= index < len(options):
            return index
    except (TypeError, ValueError):
        pass

    answer = sanitize_string(item.get("correct_answer") or item.get("answer") or "", max_length=16)
    if answer:
        normalized = answer.strip().upper()
        if normalized in {"A", "B", "C", "D"}:
            candidate = ["A", "B", "C", "D"].index(normalized)
            if candidate < len(options):
                return candidate

    return 0


def _normalize_service_question(item: Any, default_topic: str) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    text = sanitize_string(item.get("question") or item.get("text") or "", max_length=4000)
    if not text:
        return None

    options = _normalize_options(item.get("options"))
    if len(options) < 2:
        return None

    correct_index = _extract_correct_index(item, options)
    if correct_index < 0 or correct_index >= len(options):
        correct_index = 0

    return {
        "text": text,
        "options": options,
        "correct_index": correct_index,
        "hint": sanitize_string(item.get("hint") or "", max_length=500) or None,
        "explanation": sanitize_string(item.get("explanation") or "", max_length=1500) or None,
        "topic": sanitize_string(item.get("topic") or default_topic or "", max_length=128) or None,
    }


def _extract_error_message(raw_body: str) -> str:
    body = (raw_body or "").strip()
    if not body:
        return "Topic AI service returned an empty error response."

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body

    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, dict):
            nested_message = detail.get("message") or detail.get("error")
            if nested_message:
                return str(nested_message)
        if parsed.get("message"):
            return str(parsed.get("message"))
        if parsed.get("error"):
            return str(parsed.get("error"))

    return body


def _build_service_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = get_topic_ai_service_token()
    if token:
        headers["Authorization"] = f"{get_topic_ai_auth_scheme()} {token}"
    return headers


def get_topic_ai_ingestion_timeout_seconds() -> int:
    raw_value = os.environ.get(
        "AI_TOPIC_INGESTION_TIMEOUT_SECONDS",
        os.environ.get("AI_TOPIC_SERVICE_TIMEOUT_SECONDS", "180"),
    )
    try:
        timeout = int(raw_value)
    except (TypeError, ValueError):
        timeout = 180
    return max(10, min(timeout, 600))


def _normalize_numeric_vector(raw: Any, *, max_len: int = 4096) -> List[float]:
    if not isinstance(raw, list):
        return []

    vector: List[float] = []
    for item in raw:
        if len(vector) >= max_len:
            break
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            continue
    return vector


def _normalize_ai_ingestion_chunk(
    row: Any,
    *,
    default_fingerprint: str,
    fallback_index: int,
) -> Dict[str, Any] | None:
    if not isinstance(row, dict):
        return None

    text = sanitize_string(row.get("text") or "", max_length=60000)
    if not text:
        return None

    try:
        chunk_index = int(row.get("chunk_index"))
    except (TypeError, ValueError):
        chunk_index = int(fallback_index)

    text_hash = sanitize_string(row.get("text_hash") or "", max_length=64)
    if not text_hash:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    chunk_id = sanitize_string(row.get("chunk_id") or "", max_length=64)
    if not chunk_id:
        chunk_id = hashlib.sha256(
            f"{default_fingerprint}:{chunk_index}:{text_hash}".encode("utf-8")
        ).hexdigest()[:40]

    try:
        token_count = int(row.get("token_count") or 0)
    except (TypeError, ValueError):
        token_count = 0
    if token_count <= 0:
        token_count = max(1, int(round(len(re.findall(r"\\S+", text)) * 1.2)))

    vector = _normalize_numeric_vector(row.get("embedding_vector"), max_len=4096)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "text": text,
        "text_hash": text_hash,
        "char_start": row.get("char_start"),
        "char_end": row.get("char_end"),
        "token_count": token_count,
        "embedding_vector": vector,
        "embedding_dim": int(row.get("embedding_dim") or (len(vector) if vector else 0)) or None,
        "embedding_provider": sanitize_string(row.get("embedding_provider") or "", max_length=64) or None,
        "embedding_model": sanitize_string(row.get("embedding_model") or "", max_length=128) or None,
        "embedding_status": sanitize_string(row.get("embedding_status") or "embedded", max_length=32) or "embedded",
        "metadata": metadata,
    }


def process_document_with_ai_service(
    *,
    document_text: str,
    content_sha256: str | None,
    chunk_size: int,
    overlap: int,
    vector_store_requested: str | None = None,
) -> Dict[str, Any]:
    text = sanitize_string(document_text or "", max_length=600000)
    if not text:
        return {
            "ok": False,
            "status_code": 400,
            "error": "Document text is empty; cannot process with AI service.",
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "empty_document_text",
            "service_endpoint": None,
            "service_latency_ms": 0,
        }

    fingerprint = sanitize_string(content_sha256 or "", max_length=64)
    if not fingerprint:
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()

    endpoint = f"{get_topic_ai_service_url().rstrip('/')}/rag/process"
    payload = {
        "document_text": text,
        "content_sha256": fingerprint,
        "chunk_size": max(300, min(int(chunk_size or 1200), 3000)),
        "overlap": max(0, min(int(overlap or 220), int(max(300, min(int(chunk_size or 1200), 3000)) / 2))),
        "vector_store_requested": sanitize_string(vector_store_requested or "python", max_length=32) or "python",
    }

    started = time.perf_counter()
    timeout_seconds = get_topic_ai_ingestion_timeout_seconds()
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=_build_service_headers(),
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
            status_code = int(getattr(response, "status", 200) or 200)
            raw_body = response.read().decode("utf-8", errors="ignore")
    except urlerror.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        return {
            "ok": False,
            "status_code": int(getattr(exc, "code", 502) or 502),
            "error": _extract_error_message(raw_error),
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "http_error",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except urlerror.URLError as exc:
        return {
            "ok": False,
            "status_code": 503,
            "error": f"AI ingestion service unavailable: {exc.reason}",
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "network_error",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 503,
            "error": f"AI ingestion request failed: {exc}",
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "request_error",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }

    try:
        parsed = json.loads(raw_body or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status_code": 502,
            "error": "AI ingestion response was not valid JSON.",
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "invalid_json",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "status_code": 502,
            "error": "AI ingestion response had invalid payload shape.",
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "invalid_payload",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }

    if status_code >= 400:
        detail = _extract_error_message(raw_body)
        return {
            "ok": False,
            "status_code": status_code,
            "error": detail,
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "service_error",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }

    raw_chunks = parsed.get("chunks") if isinstance(parsed.get("chunks"), list) else []
    chunks: List[Dict[str, Any]] = []
    for index, row in enumerate(raw_chunks):
        normalized = _normalize_ai_ingestion_chunk(
            row,
            default_fingerprint=fingerprint,
            fallback_index=index,
        )
        if normalized:
            chunks.append(normalized)

    if not chunks:
        return {
            "ok": False,
            "status_code": 502,
            "error": "AI ingestion returned no valid chunks.",
            "chunks": [],
            "chunk_count": 0,
            "token_count": 0,
            "vector_store_effective": "python",
            "vector_store_fallback_reason": "empty_chunks",
            "service_endpoint": endpoint,
            "service_latency_ms": int((time.perf_counter() - started) * 1000),
        }

    try:
        token_count = int(parsed.get("token_count") or 0)
    except (TypeError, ValueError):
        token_count = 0
    if token_count <= 0:
        token_count = sum(int(item.get("token_count") or 0) for item in chunks)

    vector_store_effective = sanitize_string(parsed.get("vector_store_effective") or "python", max_length=32) or "python"
    vector_store_fallback_reason = sanitize_string(parsed.get("vector_store_fallback_reason") or "", max_length=160) or None

    return {
        "ok": True,
        "status_code": status_code,
        "error": None,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "token_count": int(token_count),
        "vector_store_effective": vector_store_effective,
        "vector_store_fallback_reason": vector_store_fallback_reason,
        "service_endpoint": endpoint,
        "service_latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _get_gemini_api_key() -> str:
    return str(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()


def _get_gemini_model_name() -> str:
    raw = str(os.environ.get("AI_TOPIC_GEMINI_MODEL") or "gemini-2.5-flash").strip()
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "", raw)
    return cleaned or "gemini-2.5-flash"


def _parse_question_candidates(raw_text: str) -> List[Any]:
    text = str(raw_text or "").strip()
    if not text:
        return []

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = text.strip()

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        rows = parsed.get("questions") or parsed.get("mcqs") or []
        return rows if isinstance(rows, list) else []

    if isinstance(parsed, list):
        return parsed

    return []


def _extract_gemini_text(response_data: Dict[str, Any]) -> str:
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list):
        return ""

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


def _build_generation_context(
    *,
    subject: str,
    grade: str,
    difficulty: str,
    topic: str | None,
    requested_count: int,
    test_title: str | None,
    test_description: str | None,
    rag_context: str | None,
    generation_mode: str | None,
) -> Dict[str, Any]:
    clean_topic = _derive_source_topic(topic, subject, grade)
    clean_subject = sanitize_string(subject or "", max_length=64) or "general"
    clean_grade = sanitize_string(grade or "", max_length=32) or "high"
    clean_difficulty = sanitize_string(difficulty or "", max_length=16) or "medium"
    clean_title = sanitize_string(test_title or "", max_length=255) or ""
    clean_description = sanitize_string(test_description or "", max_length=1200) or ""
    clean_mode = sanitize_string(generation_mode or "standard", max_length=24) or "standard"
    clean_rag_context = sanitize_string(rag_context or "", max_length=12000) or ""

    return {
        "topic": clean_topic,
        "subject": clean_subject,
        "grade": clean_grade,
        "difficulty": clean_difficulty,
        "requested_count": int(max(1, min(requested_count, 50))),
        "test_title": clean_title,
        "test_description": clean_description,
        "generation_mode": clean_mode,
        "rag_context": clean_rag_context,
    }


def _build_default_gemini_prompt(context: Dict[str, Any]) -> str:
    rag_context = context.get("rag_context") or ""
    rag_block = ""
    if rag_context:
        rag_block = (
            "RAG CONTEXT (must be treated as the primary factual source when relevant):\n"
            f"{rag_context}\n\n"
        )

    title = context.get("test_title") or ""
    description = context.get("test_description") or ""
    title_block = ""
    if title or description:
        title_block = (
            "TEST CONFIGURATION:\n"
            f"- Test Title: {title or 'N/A'}\n"
            f"- Test Description: {description or 'N/A'}\n"
        )

    return (
        "You are an elite assessment designer for STEM education.\n"
        "Generate multiple-choice questions that align exactly with the user configuration.\n\n"
        "USER CONFIGURATION:\n"
        f"- Generation Mode: {context.get('generation_mode')}\n"
        f"- Subject: {context.get('subject')}\n"
        f"- Grade: {context.get('grade')}\n"
        f"- Difficulty: {context.get('difficulty')}\n"
        f"- Topic/Sub-topic: {context.get('topic')}\n"
        f"- Required Question Count: {context.get('requested_count')}\n"
        f"{title_block}\n"
        f"{rag_block}"
        "RESPONSE RULES:\n"
        "- Return ONLY valid JSON. No markdown. No prose outside JSON.\n"
        "- JSON schema: {\"questions\":[{\"text\":string,\"options\":[string,string,string,string],\"correct_index\":0-3,\"explanation\":string,\"topic\":string}]}\n"
        "- Each question must have exactly 4 distinct options and exactly one correct answer.\n"
        "- Distractors must be plausible for the chosen grade and difficulty.\n"
        "- Explanations must be concise and instructional.\n"
        "- Ensure all generated questions reflect all fields provided in USER CONFIGURATION.\n"
        "- Generate exactly the requested question count; do not under-generate.\n"
    )


def _optimize_prompt_with_hf_service(context: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = f"{get_topic_ai_service_url().rstrip('/')}/prompt/optimize"
    timeout_seconds = max(5, min(get_topic_ai_timeout_seconds(), 30))
    payload = {
        "source_type": "topic",
        "source": context.get("topic"),
        "subject": context.get("subject"),
        "grade": context.get("grade"),
        "difficulty": context.get("difficulty"),
        "num_questions": context.get("requested_count"),
        "generation_mode": context.get("generation_mode"),
        "test_title": context.get("test_title"),
        "test_description": context.get("test_description"),
        "rag_context": context.get("rag_context"),
    }

    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=_build_service_headers(),
    )

    with urlrequest.urlopen(req, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 200) or 200)
        body = response.read().decode("utf-8", errors="ignore")

    if status >= 400:
        raise ValueError(_extract_error_message(body))

    parsed = json.loads(body or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Prompt optimizer returned an invalid response.")

    prompt = sanitize_string(parsed.get("optimized_prompt") or parsed.get("prompt") or "", max_length=24000)
    if not prompt:
        raise ValueError("Prompt optimizer returned an empty prompt.")

    return {
        "prompt": prompt,
        "endpoint": endpoint,
        "source": sanitize_string(parsed.get("source") or "hf_prompt_optimizer", max_length=64) or "hf_prompt_optimizer",
    }


def generate_topic_mcqs(
    *,
    subject: str,
    grade: str,
    difficulty: str,
    topic: str | None,
    count: int,
    seed: int | None = None,
    test_title: str | None = None,
    test_description: str | None = None,
    rag_context: str | None = None,
    generation_mode: str | None = None,
) -> Dict[str, Any]:
    requested_count = max(1, min(int(count), 50))
    source_topic = _derive_source_topic(topic, subject, grade)
    request_started = time.perf_counter()

    context = _build_generation_context(
        subject=subject,
        grade=grade,
        difficulty=difficulty,
        topic=source_topic,
        requested_count=requested_count,
        test_title=test_title,
        test_description=test_description,
        rag_context=rag_context,
        generation_mode=generation_mode,
    )

    prompt_text = _build_default_gemini_prompt(context)
    prompt_source = "deterministic"
    prompt_optimizer_endpoint = None
    prompt_optimizer_error = None

    if str(os.environ.get("AI_TOPIC_PROMPT_OPTIMIZER_ENABLED") or "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            optimized = _optimize_prompt_with_hf_service(context)
            prompt_text = optimized.get("prompt") or prompt_text
            prompt_source = optimized.get("source") or "hf_prompt_optimizer"
            prompt_optimizer_endpoint = optimized.get("endpoint")
        except Exception as exc:
            prompt_optimizer_error = str(exc)

    gemini_api_key = _get_gemini_api_key()
    if not gemini_api_key:
        return {
            "ok": False,
            "status_code": 503,
            "error": "Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY.",
            "questions": [],
            "meta": {
                "provider": "gemini",
                "model": _get_gemini_model_name(),
                "prompt_source": prompt_source,
                "prompt_optimizer_endpoint": prompt_optimizer_endpoint,
                "prompt_optimizer_error": prompt_optimizer_error,
            },
            "requested_count": requested_count,
            "generated_count": 0,
            "service_latency_ms": int((time.perf_counter() - request_started) * 1000),
        }

    model_name = _get_gemini_model_name()
    gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"

    request_payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": "You generate high-quality STEM MCQs and return only valid JSON."
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}],
            }
        ],
        "generationConfig": {
            "temperature": 0.35,
            "topP": 0.9,
            "responseMimeType": "application/json",
        },
    }

    request_obj = urlrequest.Request(
        gemini_endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlrequest.urlopen(request_obj, timeout=get_topic_ai_timeout_seconds()) as response:
            status_code = int(getattr(response, "status", 200) or 200)
            raw_response = response.read().decode("utf-8", errors="ignore")
    except urlerror.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        return {
            "ok": False,
            "status_code": int(getattr(exc, "code", 502) or 502),
            "error": _extract_error_message(raw_error),
            "questions": [],
            "meta": {
                "provider": "gemini",
                "model": model_name,
                "prompt_source": prompt_source,
                "prompt_optimizer_endpoint": prompt_optimizer_endpoint,
                "prompt_optimizer_error": prompt_optimizer_error,
            },
            "service_endpoint": gemini_endpoint,
            "requested_count": requested_count,
            "generated_count": 0,
            "service_latency_ms": int((time.perf_counter() - request_started) * 1000),
        }
    except urlerror.URLError as exc:
        return {
            "ok": False,
            "status_code": 503,
            "error": f"Gemini API unavailable: {exc.reason}",
            "questions": [],
            "meta": {
                "provider": "gemini",
                "model": model_name,
                "prompt_source": prompt_source,
                "prompt_optimizer_endpoint": prompt_optimizer_endpoint,
                "prompt_optimizer_error": prompt_optimizer_error,
            },
            "service_endpoint": gemini_endpoint,
            "requested_count": requested_count,
            "generated_count": 0,
            "service_latency_ms": int((time.perf_counter() - request_started) * 1000),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 503,
            "error": f"Gemini request failed: {exc}",
            "questions": [],
            "meta": {
                "provider": "gemini",
                "model": model_name,
                "prompt_source": prompt_source,
                "prompt_optimizer_endpoint": prompt_optimizer_endpoint,
                "prompt_optimizer_error": prompt_optimizer_error,
            },
            "service_endpoint": gemini_endpoint,
            "requested_count": requested_count,
            "generated_count": 0,
            "service_latency_ms": int((time.perf_counter() - request_started) * 1000),
        }

    try:
        response_data = json.loads(raw_response)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status_code": 502,
            "error": "Gemini response was not valid JSON.",
            "questions": [],
            "meta": {
                "provider": "gemini",
                "model": model_name,
                "prompt_source": prompt_source,
                "prompt_optimizer_endpoint": prompt_optimizer_endpoint,
                "prompt_optimizer_error": prompt_optimizer_error,
            },
            "service_endpoint": gemini_endpoint,
            "requested_count": requested_count,
            "generated_count": 0,
            "service_latency_ms": int((time.perf_counter() - request_started) * 1000),
        }

    ai_content_text = _extract_gemini_text(response_data)
    candidates = _parse_question_candidates(ai_content_text)

    rows = []
    if isinstance(candidates, list):
        for candidate in candidates:
            normalized = _normalize_service_question(candidate, source_topic)
            if normalized:
                rows.append(normalized)

    rows = rows[:requested_count]

    return {
        "ok": len(rows) > 0,
        "status_code": status_code,
        "error": None if rows else "Gemini did not return parseable questions.",
        "questions": rows,
        "meta": {
            "provider": "gemini",
            "model": model_name,
            "prompt_source": prompt_source,
            "prompt_optimizer_endpoint": prompt_optimizer_endpoint,
            "prompt_optimizer_error": prompt_optimizer_error,
        },
        "service_endpoint": gemini_endpoint,
        "requested_count": requested_count,
        "generated_count": len(rows),
        "service_latency_ms": int((time.perf_counter() - request_started) * 1000),
    }
