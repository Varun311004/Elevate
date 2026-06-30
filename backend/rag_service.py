"""Utilities for teacher-side RAG document ingestion and retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from urllib.parse import quote
from urllib import error as urllib_error
from urllib import request as urllib_request

from sqlalchemy import text as sql_text

from .models import db


ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}
DEFAULT_RAG_MAX_UPLOAD_MB = 10
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 220
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_PGVECTOR_DIM = 1536
DEFAULT_R2_OBJECT_PREFIX = "teacher-documents"
DEFAULT_RAG_RETENTION_DAYS = 120


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.environ.get(name) or "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _coerce_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def normalize_document_extension(filename: str | None) -> str:
    value = str(filename or "").strip().lower()
    if "." not in value:
        return ""
    return f".{value.rsplit('.', 1)[-1]}"


def is_allowed_document_extension(filename: str | None) -> bool:
    return normalize_document_extension(filename) in ALLOWED_DOCUMENT_EXTENSIONS


def get_rag_max_upload_bytes() -> int:
    raw = os.environ.get("RAG_MAX_UPLOAD_MB", DEFAULT_RAG_MAX_UPLOAD_MB)
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = DEFAULT_RAG_MAX_UPLOAD_MB
    mb = max(1, min(50, mb))
    return mb * 1024 * 1024


def get_rag_async_min_bytes() -> int:
    mb = _coerce_float(os.environ.get("RAG_ASYNC_MIN_MB"), 4.0, minimum=0.5, maximum=20.0)
    return int(mb * 1024 * 1024)


def get_rag_max_documents_per_teacher() -> int:
    return _coerce_int(os.environ.get("RAG_MAX_DOCS_PER_TEACHER"), 120, minimum=1, maximum=1000)


def get_rag_max_storage_bytes_per_teacher() -> int:
    gb = _coerce_float(os.environ.get("RAG_MAX_STORAGE_GB_PER_TEACHER"), 1.0, minimum=0.1, maximum=20.0)
    return int(gb * 1024 * 1024 * 1024)


def get_rag_retention_days() -> int:
    return _coerce_int(os.environ.get("RAG_RETENTION_DAYS"), DEFAULT_RAG_RETENTION_DAYS, minimum=1, maximum=720)


def get_rag_cleanup_batch_size() -> int:
    return _coerce_int(os.environ.get("RAG_CLEANUP_BATCH_SIZE"), 10, minimum=1, maximum=100)


def get_rag_max_selected_docs() -> int:
    return _coerce_int(os.environ.get("RAG_MAX_SELECTED_DOCS"), 20, minimum=1, maximum=120)


def get_rag_max_chunk_candidates() -> int:
    return _coerce_int(os.environ.get("RAG_MAX_CHUNK_CANDIDATES"), 2200, minimum=100, maximum=20000)


def is_r2_strict_mode_enabled() -> bool:
    return _env_bool("RAG_R2_STRICT", default=False)


def get_rag_min_confidence() -> float:
    raw = os.environ.get("RAG_MIN_CONFIDENCE", "0.45")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.45
    return max(0.0, min(1.0, value))


def _sanitize_storage_filename(filename: str | None) -> str:
    raw = os.path.basename(str(filename or "document.txt")).strip() or "document.txt"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip(".-")
    return safe or "document.txt"


def _sanitize_teacher_folder_name(teacher_name: str | None, teacher_id: int) -> str:
    raw = str(teacher_name or "").strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "-", raw).strip(".-")
    if not safe:
        safe = f"teacher-{max(1, int(teacher_id or 1))}"
    return safe


def is_r2_configured() -> bool:
    required_keys = (
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
    )
    return all(str(os.environ.get(key) or "").strip() for key in required_keys)


def resolve_document_storage_backend(requested: str | None = None) -> Dict[str, Any]:
    normalized = str(requested or os.environ.get("RAG_DOCUMENT_STORAGE", "local")).strip().lower() or "local"
    if normalized in {"cloudflare-r2", "cloudflare_r2"}:
        normalized = "r2"

    strict_r2 = is_r2_strict_mode_enabled()

    if normalized == "r2":
        if not is_r2_configured():
            if strict_r2:
                return {
                    "requested": "r2",
                    "effective": "invalid",
                    "fallback_reason": "r2_not_configured",
                    "strict": True,
                }
            return {
                "requested": "r2",
                "effective": "local",
                "fallback_reason": "r2_not_configured",
                "strict": False,
            }
        try:
            import boto3  # noqa: F401
        except Exception:
            if strict_r2:
                return {
                    "requested": "r2",
                    "effective": "invalid",
                    "fallback_reason": "boto3_missing",
                    "strict": True,
                }
            return {
                "requested": "r2",
                "effective": "local",
                "fallback_reason": "boto3_missing",
                "strict": False,
            }
        return {
            "requested": "r2",
            "effective": "r2",
            "fallback_reason": None,
            "strict": strict_r2,
        }

    if normalized not in {"local", "filesystem"}:
        return {
            "requested": normalized,
            "effective": "local",
            "fallback_reason": "unsupported_document_storage",
            "strict": False,
        }

    return {
        "requested": normalized,
        "effective": "local",
        "fallback_reason": None,
        "strict": False,
    }


def _build_r2_client():
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime package set
        raise ValueError("R2 storage requires boto3. Install boto3 in the backend environment.") from exc

    endpoint = str(os.environ.get("R2_ENDPOINT_URL") or "").strip()
    access_key = str(os.environ.get("R2_ACCESS_KEY_ID") or "").strip()
    secret_key = str(os.environ.get("R2_SECRET_ACCESS_KEY") or "").strip()
    region = str(os.environ.get("R2_REGION") or "auto").strip() or "auto"

    if not endpoint or not access_key or not secret_key:
        raise ValueError("R2 credentials are incomplete. Set R2 endpoint, access key, and secret key.")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def build_r2_object_key(
    *,
    teacher_id: int,
    teacher_name: str | None = None,
    original_filename: str,
    content_sha256: str,
) -> str:
    prefix = str(os.environ.get("R2_OBJECT_PREFIX") or DEFAULT_R2_OBJECT_PREFIX).strip().strip("/")
    safe_name = _sanitize_storage_filename(original_filename)
    teacher_folder = _sanitize_teacher_folder_name(teacher_name, teacher_id)
    fingerprint = str(content_sha256 or "")[:12] or "nohash"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    parts = [teacher_folder, f"{timestamp}-{fingerprint}-{safe_name}"]
    if prefix:
        parts.insert(0, prefix)
    return "/".join(parts)


def build_r2_public_url(object_key: str) -> str:
    base = str(os.environ.get("R2_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/{quote(str(object_key or '').strip('/'), safe='/')}"


def upload_document_to_r2(
    raw_bytes: bytes,
    *,
    teacher_id: int,
    teacher_name: str | None = None,
    original_filename: str,
    content_sha256: str,
    content_type: str | None = None,
) -> Dict[str, Any]:
    if not isinstance(raw_bytes, (bytes, bytearray)) or not raw_bytes:
        raise ValueError("Cannot upload empty document bytes to R2.")

    bucket = str(os.environ.get("R2_BUCKET_NAME") or "").strip()
    endpoint = str(os.environ.get("R2_ENDPOINT_URL") or "").strip()
    if not bucket:
        raise ValueError("R2 bucket is missing. Set R2_BUCKET_NAME.")

    key = build_r2_object_key(
        teacher_id=teacher_id,
        teacher_name=teacher_name,
        original_filename=original_filename,
        content_sha256=content_sha256,
    )

    client = _build_r2_client()
    put_kwargs = {
        "Bucket": bucket,
        "Key": key,
        "Body": bytes(raw_bytes),
    }
    normalized_content_type = str(content_type or "").strip()
    if normalized_content_type:
        put_kwargs["ContentType"] = normalized_content_type

    client.put_object(**put_kwargs)

    return {
        "backend": "r2",
        "storage_path": f"r2://{bucket}/{key}",
        "bucket": bucket,
        "key": key,
        "endpoint": endpoint,
        "public_url": build_r2_public_url(key) or None,
    }


def _parse_r2_storage_path(storage_path: str | None) -> Dict[str, str] | None:
    value = str(storage_path or "").strip()
    if not value.startswith("r2://"):
        return None

    without_scheme = value[len("r2://"):]
    if "/" not in without_scheme:
        return None

    bucket, key = without_scheme.split("/", 1)
    bucket = bucket.strip()
    key = key.strip().lstrip("/")
    if not bucket or not key:
        return None
    return {"bucket": bucket, "key": key}


def delete_document_from_r2(storage_path: str | None = None, metadata: Dict[str, Any] | None = None) -> bool:
    metadata = metadata if isinstance(metadata, dict) else {}
    bucket = str(metadata.get("r2_bucket") or metadata.get("bucket") or "").strip()
    key = str(metadata.get("r2_key") or metadata.get("key") or "").strip().lstrip("/")

    if not bucket or not key:
        parsed = _parse_r2_storage_path(storage_path)
        if parsed:
            bucket = parsed.get("bucket", "")
            key = parsed.get("key", "")

    if not bucket or not key:
        return False

    client = _build_r2_client()
    client.delete_object(Bucket=bucket, Key=key)
    return True


def normalize_document_text(text: str | None) -> str:
    value = str(text or "").replace("\r", "\n")
    value = re.sub(r"[\t\f\v]+", " ", value)
    value = re.sub(r"\u0000", "", value)
    value = re.sub(r"[ ]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _decode_text_payload(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded = raw.decode(encoding)
            if decoded:
                return decoded
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def extract_document_text(file_path: str, extension: str) -> str:
    ext = str(extension or "").strip().lower()

    if ext == ".txt":
        with open(file_path, "rb") as handle:
            return normalize_document_text(_decode_text_payload(handle.read()))

    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime package set
            raise ValueError("PDF parsing dependency missing. Install pypdf.") from exc

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return normalize_document_text("\n\n".join(pages))

    if ext in {".docx", ".doc"}:
        try:
            from docx import Document as DocxDocument  # type: ignore
            doc = DocxDocument(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
            return normalize_document_text("\n".join(paragraphs))
        except Exception as exc:
            if ext == ".doc":
                # Legacy .doc parsing support is limited without extra binaries.
                with open(file_path, "rb") as handle:
                    decoded = normalize_document_text(_decode_text_payload(handle.read()))
                if len(decoded) >= 120:
                    return decoded
            raise ValueError("Unable to parse Word document. Prefer .docx, .pdf, or .txt files.") from exc

    raise ValueError(f"Unsupported document extension: {ext or 'unknown'}")


def approximate_token_count(text: str | None) -> int:
    words = re.findall(r"\S+", str(text or ""))
    if not words:
        return 0
    # Rough heuristic used for storage and chunk diagnostics.
    return max(1, int(round(len(words) * 1.2)))


def build_deterministic_chunks(
    text: str,
    *,
    document_fingerprint: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    normalized = normalize_document_text(text)
    if not normalized:
        return []

    chunk_size = max(300, min(int(chunk_size or DEFAULT_CHUNK_SIZE), 3000))
    overlap = max(0, min(int(overlap or DEFAULT_CHUNK_OVERLAP), int(chunk_size / 2)))

    chunks: List[Dict[str, Any]] = []
    start = 0
    index = 0
    total = len(normalized)
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

    while start < total:
        tentative_end = min(total, start + chunk_size)
        end = tentative_end

        if tentative_end < total:
            min_breakpoint = start + int((tentative_end - start) * 0.58)
            for marker, offset in break_markers:
                candidate = normalized.rfind(marker, min_breakpoint, tentative_end)
                if candidate > start:
                    end = candidate + offset
                    break

        if end <= start:
            end = tentative_end

        snippet = normalized[start:end].strip()
        if snippet:
            text_hash = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{document_fingerprint}:{index}:{text_hash}".encode("utf-8")
            ).hexdigest()[:40]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "text": snippet,
                    "text_hash": text_hash,
                    "char_start": start,
                    "char_end": end,
                    "token_count": approximate_token_count(snippet),
                }
            )
            index += 1

        if end >= total:
            break

        start = min(max(end - overlap, start + 1), total)

    return chunks


def _normalize_vector(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    return [float(v / norm) for v in values]


def hash_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> List[float]:
    vector = [0.0] * max(32, int(dim or DEFAULT_EMBEDDING_DIM))
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())

    if not tokens:
        seed = hashlib.sha256(str(text or "").encode("utf-8")).digest()
        vector[seed[0] % len(vector)] = 1.0
        return _normalize_vector(vector)

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % len(vector)
        sign = -1.0 if (digest[4] % 2) else 1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[idx] += sign * weight

    return _normalize_vector(vector)


def is_postgres_database_url() -> bool:
    configured = str(os.environ.get("DATABASE_URL") or "").strip().lower()
    return configured.startswith("postgresql://") or configured.startswith("postgres://")


def get_pgvector_dim() -> int:
    return _coerce_int(os.environ.get("RAG_PGVECTOR_DIM"), DEFAULT_PGVECTOR_DIM, minimum=256, maximum=4096)


def is_pgvector_enabled() -> bool:
    # pgvector is enabled only when explicitly allowed and a PostgreSQL URL is active.
    return _env_bool("RAG_ENABLE_PGVECTOR", default=True) and is_postgres_database_url()


def resolve_vector_store_choice(requested: str | None) -> Dict[str, Any]:
    normalized = str(requested or os.environ.get("RAG_VECTOR_STORE", "python")).strip().lower() or "python"
    if normalized in {"pgvector", "supabase_pgvector", "supabase"}:
        if is_pgvector_enabled():
            return {
                "requested": normalized,
                "effective": "pgvector",
                "fallback_reason": None,
            }
        return {
            "requested": normalized,
            "effective": "python",
            "fallback_reason": "pgvector_not_available",
        }

    if normalized not in {"python", "memory", "local"}:
        return {
            "requested": normalized,
            "effective": "python",
            "fallback_reason": "unsupported_vector_store",
        }

    return {
        "requested": normalized,
        "effective": "python",
        "fallback_reason": None,
    }


def _normalize_vector_dim(vector: List[float], target_dim: int) -> List[float]:
    dim = max(16, int(target_dim or DEFAULT_EMBEDDING_DIM))
    source = [float(v) for v in (vector or [])]
    if len(source) == dim:
        return source
    if len(source) > dim:
        return source[:dim]
    return source + ([0.0] * (dim - len(source)))


def _embed_with_openai(text: str, *, model: str, timeout_seconds: float) -> List[float]:
    api_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    payload = json.dumps({
        "model": model,
        "input": [str(text or " ")],
    }).encode("utf-8")

    request = urllib_request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=max(2.0, float(timeout_seconds))) as response:  # nosec B310
            body = response.read().decode("utf-8", errors="ignore")
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise ValueError(f"openai_http_{exc.code}: {raw[:280]}") from exc
    except urllib_error.URLError as exc:
        raise ValueError(f"openai_network_error: {exc.reason}") from exc

    parsed = json.loads(body or "{}")
    data = parsed.get("data") or []
    if not data:
        raise ValueError("openai_response_missing_data")

    vector = data[0].get("embedding") or []
    if not isinstance(vector, list) or not vector:
        raise ValueError("openai_response_missing_embedding")

    return [float(v) for v in vector]


def build_embedding_payload(text: str) -> Dict[str, Any]:
    requested_provider = str(os.environ.get("RAG_EMBEDDING_PROVIDER", "auto")).strip().lower() or "auto"
    model = str(os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small")).strip() or "text-embedding-3-small"
    timeout_seconds = _coerce_float(os.environ.get("RAG_EMBEDDING_TIMEOUT_SECONDS"), 8.0, minimum=2.0, maximum=60.0)
    target_dim = get_pgvector_dim() if requested_provider in {"auto", "openai"} else DEFAULT_EMBEDDING_DIM

    should_try_openai = requested_provider in {"openai", "auto"}
    if should_try_openai and str(os.environ.get("OPENAI_API_KEY") or "").strip():
        try:
            vector = _embed_with_openai(text, model=model, timeout_seconds=timeout_seconds)
            vector = _normalize_vector_dim(vector, target_dim)
            return {
                "embedding_vector": vector,
                "embedding_dim": len(vector),
                "embedding_provider": "openai",
                "embedding_model": model,
                "embedding_status": "embedded",
            }
        except Exception as exc:
            if requested_provider == "openai" and _env_bool("RAG_EMBEDDING_STRICT", default=False):
                raise
            fallback_vector = _normalize_vector_dim(hash_embedding(text), DEFAULT_EMBEDDING_DIM)
            return {
                "embedding_vector": fallback_vector,
                "embedding_dim": len(fallback_vector),
                "embedding_provider": "hash-local",
                "embedding_model": "hash-v1",
                "embedding_status": "embedded_fallback",
                "embedding_error": str(exc),
            }

    vector = _normalize_vector_dim(hash_embedding(text), DEFAULT_EMBEDDING_DIM)
    return {
        "embedding_vector": vector,
        "embedding_dim": len(vector),
        "embedding_provider": "hash-local",
        "embedding_model": "hash-v1",
        "embedding_status": "embedded",
    }


def vector_to_pg_literal(vector: List[float], *, dim: int | None = None) -> str:
    target_dim = int(dim or get_pgvector_dim())
    normalized = _normalize_vector_dim(vector, target_dim)
    return "[" + ",".join(f"{float(v):.8f}" for v in normalized) + "]"


def parse_pgvector_text(raw: str | None) -> List[float]:
    value = str(raw or "").strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return []
    parts = [segment.strip() for segment in value.split(",") if segment.strip()]
    result: List[float] = []
    for segment in parts:
        try:
            result.append(float(segment))
        except (TypeError, ValueError):
            continue
    return result


def ensure_pgvector_extension() -> None:
    if not is_pgvector_enabled():
        return
    db.session.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))


def store_pgvector_embeddings(document_id: int, chunk_vectors: List[Dict[str, Any]]) -> int:
    if not document_id or not chunk_vectors or not is_pgvector_enabled():
        return 0

    ensure_pgvector_extension()
    statement = sql_text(
        """
        UPDATE teacher_document_chunks
        SET embedding_vector_pg = CAST(:vector_literal AS vector),
            vector_store = 'pgvector'
        WHERE document_id = :document_id AND chunk_id = :chunk_id
        """
    )

    affected = 0
    for item in chunk_vectors:
        chunk_id = str(item.get("chunk_id") or "").strip()
        vector = item.get("vector") or []
        if not chunk_id or not isinstance(vector, list) or not vector:
            continue
        db.session.execute(
            statement,
            {
                "vector_literal": vector_to_pg_literal(vector),
                "document_id": int(document_id),
                "chunk_id": chunk_id,
            },
        )
        affected += 1

    return affected


def score_chunks_for_query_pgvector(
    query_text: str,
    *,
    teacher_id: int,
    selected_document_ids: List[int] | None = None,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    if not query_text or not teacher_id or not is_pgvector_enabled():
        return []

    payload = build_embedding_payload(query_text)
    query_vector = payload.get("embedding_vector") or []
    if not isinstance(query_vector, list) or not query_vector:
        return []

    ensure_pgvector_extension()

    params: Dict[str, Any] = {
        "teacher_id": int(teacher_id),
        "limit": max(1, min(int(top_k or 4), 20)),
        "query_vector": vector_to_pg_literal(query_vector),
    }
    filters = [
        "c.teacher_id = :teacher_id",
        "d.teacher_id = :teacher_id",
        "d.status = 'processed'",
        "c.embedding_vector_pg IS NOT NULL",
    ]

    doc_ids = [int(doc_id) for doc_id in (selected_document_ids or []) if int(doc_id or 0) > 0]
    if doc_ids:
        placeholders: List[str] = []
        for idx, doc_id in enumerate(doc_ids):
            key = f"doc_id_{idx}"
            params[key] = doc_id
            placeholders.append(f":{key}")
        filters.append(f"c.document_id IN ({','.join(placeholders)})")

    statement = sql_text(
        """
        SELECT
            c.document_id AS document_id,
            COALESCE(d.title, d.filename, 'Document') AS document_title,
            c.chunk_id AS chunk_id,
            c.chunk_index AS chunk_index,
            c.text AS chunk_text,
            c.embedding_vector_pg AS chunk_vector_pg,
            (1 - (c.embedding_vector_pg <=> CAST(:query_vector AS vector))) AS similarity
        FROM teacher_document_chunks c
        JOIN teacher_documents d ON d.id = c.document_id
        WHERE {filters}
        ORDER BY c.embedding_vector_pg <=> CAST(:query_vector AS vector)
        LIMIT :limit
        """.format(filters=" AND ".join(filters))
    )

    rows = db.session.execute(statement, params).mappings().all()
    scored: List[Dict[str, Any]] = []
    for row in rows:
        scored.append(
            {
                "document_id": int(row.get("document_id") or 0),
                "document_title": str(row.get("document_title") or "Document"),
                "chunk_id": str(row.get("chunk_id") or ""),
                "chunk_index": int(row.get("chunk_index") or 0),
                "chunk_text": str(row.get("chunk_text") or ""),
                "chunk_vector": parse_pgvector_text(row.get("chunk_vector_pg")),
                "similarity": float(row.get("similarity") or 0.0),
                "retrieval_mode": "pgvector",
            }
        )

    return scored


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = list(vec_a or [])
    b = list(vec_b or [])
    if not a or not b:
        return 0.0

    size = min(len(a), len(b))
    if size <= 0:
        return 0.0

    dot = sum(float(a[i]) * float(b[i]) for i in range(size))
    norm_a = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(size)))
    norm_b = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(size)))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def score_chunks_for_query(
    query_text: str,
    chunk_rows: List[Any],
    top_k: int = 4,
    *,
    vector_store: str = "python",
    teacher_id: int | None = None,
    selected_document_ids: List[int] | None = None,
) -> List[Dict[str, Any]]:
    if not query_text:
        return []

    normalized_store = str(vector_store or "python").strip().lower()
    if normalized_store == "pgvector" and teacher_id:
        try:
            pgvector_rows = score_chunks_for_query_pgvector(
                query_text,
                teacher_id=int(teacher_id),
                selected_document_ids=selected_document_ids or [],
                top_k=top_k,
            )
            if pgvector_rows:
                return pgvector_rows
        except Exception:
            # Fall through to Python retrieval path for resilience.
            pass

    if not chunk_rows:
        return []

    query_vector = build_embedding_payload(query_text).get("embedding_vector") or hash_embedding(query_text)
    scored: List[Dict[str, Any]] = []

    for row in chunk_rows:
        vector = getattr(row, "embedding_vector", None) or []
        if not isinstance(vector, list) or not vector:
            vector = parse_pgvector_text(getattr(row, "embedding_vector_pg", None))
        if not isinstance(vector, list) or not vector:
            continue

        score = cosine_similarity(query_vector, vector)
        document = getattr(row, "document", None)
        document_title = getattr(document, "title", None) or getattr(document, "filename", "Document")

        scored.append(
            {
                "document_id": int(getattr(row, "document_id", 0) or 0),
                "document_title": str(document_title),
                "chunk_id": str(getattr(row, "chunk_id", "")),
                "chunk_index": int(getattr(row, "chunk_index", 0) or 0),
                "chunk_text": str(getattr(row, "text", "")),
                "chunk_vector": vector,
                "similarity": float(score),
                "retrieval_mode": "python",
            }
        )

    scored.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return scored[: max(1, min(int(top_k or 4), 12))]


def summarize_retrieval_confidence(retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    if not retrieved_chunks:
        return {
            "confidence": 0.0,
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
        }

    sims = [float(item.get("similarity", 0.0)) for item in retrieved_chunks]
    avg_similarity = sum(sims) / len(sims)
    max_similarity = max(sims)

    avg_norm = (avg_similarity + 1.0) / 2.0
    max_norm = (max_similarity + 1.0) / 2.0
    confidence = max(0.0, min(1.0, (0.65 * avg_norm) + (0.35 * max_norm)))

    return {
        "confidence": round(confidence, 4),
        "avg_similarity": round(avg_similarity, 4),
        "max_similarity": round(max_similarity, 4),
    }


def assemble_context_text(retrieved_chunks: List[Dict[str, Any]], max_chars: int = 7000) -> str:
    if not retrieved_chunks:
        return ""

    parts: List[str] = []
    current_size = 0
    limit = max(500, int(max_chars or 7000))

    for item in retrieved_chunks:
        chunk_text = str(item.get("chunk_text", "")).strip()
        if not chunk_text:
            continue

        header = (
            f"[Source: {item.get('document_title', 'Document')} | "
            f"Chunk {int(item.get('chunk_index', 0)) + 1} | "
            f"Score {float(item.get('similarity', 0.0)):.3f}]"
        )
        section = f"{header}\n{chunk_text}"

        if current_size + len(section) > limit:
            remaining = limit - current_size
            if remaining <= 0:
                break
            section = section[:remaining].rstrip()
            if not section:
                break

        parts.append(section)
        current_size += len(section) + 2
        if current_size >= limit:
            break

    return "\n\n".join(parts)


def _question_identity_key(question: Dict[str, Any]) -> str:
    text = " ".join(str(question.get("text", "")).strip().lower().split())
    options = sorted(
        " ".join(str(opt).strip().lower().split())
        for opt in (question.get("options") or [])
        if str(opt).strip()
    )
    return f"{text}|{'|'.join(options)}"


def attach_question_provenance(
    questions: List[Dict[str, Any]],
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not questions:
        return {
            "questions": [],
            "metrics": {
                "coverage": 0.0,
                "relevance": 0.0,
                "duplication": 0.0,
                "provenance_count": 0,
            },
        }

    enriched: List[Dict[str, Any]] = []
    used_chunk_ids = set()
    best_scores: List[float] = []
    identity_keys = []

    for row in questions:
        item = dict(row or {})
        identity_keys.append(_question_identity_key(item))

        if retrieved_chunks:
            question_vector = build_embedding_payload(item.get("text") or "").get("embedding_vector") or hash_embedding(item.get("text") or "")
            ranked: List[Dict[str, Any]] = []

            for chunk in retrieved_chunks:
                chunk_vector = chunk.get("chunk_vector") or []
                if not chunk_vector:
                    chunk_vector = hash_embedding(chunk.get("chunk_text") or "")
                similarity = cosine_similarity(question_vector, chunk_vector)
                ranked.append({
                    "chunk": chunk,
                    "similarity": float(similarity),
                })

            ranked.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
            best = ranked[0] if ranked else None
            if best:
                best_chunk = best["chunk"]
                best_similarity = float(best.get("similarity") or 0.0)
                used_chunk_ids.add(best_chunk.get("chunk_id"))
                best_scores.append(float(best_similarity))
                item["provenance"] = {
                    "document_id": int(best_chunk.get("document_id") or 0),
                    "document_title": str(best_chunk.get("document_title") or "Document"),
                    "chunk_id": str(best_chunk.get("chunk_id") or ""),
                    "chunk_index": int(best_chunk.get("chunk_index") or 0),
                    "similarity": round(float(best_similarity), 4),
                }

                trace_items = []
                for trace in ranked[:3]:
                    chunk = trace["chunk"]
                    trace_items.append(
                        {
                            "document_id": int(chunk.get("document_id") or 0),
                            "document_title": str(chunk.get("document_title") or "Document"),
                            "chunk_id": str(chunk.get("chunk_id") or ""),
                            "chunk_index": int(chunk.get("chunk_index") or 0),
                            "similarity": round(float(trace.get("similarity") or 0.0), 4),
                            "snippet": str(chunk.get("chunk_text") or "")[:260].strip(),
                        }
                    )
                item["retrieval_trace"] = trace_items

        enriched.append(item)

    unique_identity_count = len({key for key in identity_keys if key})
    total_count = len(identity_keys)
    duplication = 0.0
    if total_count > 0:
        duplication = max(0.0, 1.0 - (unique_identity_count / float(total_count)))

    coverage = 0.0
    if retrieved_chunks:
        coverage = len(used_chunk_ids) / float(len(retrieved_chunks))

    relevance = 0.0
    if best_scores:
        relevance = sum(best_scores) / float(len(best_scores))

    return {
        "questions": enriched,
        "metrics": {
            "coverage": round(coverage, 4),
            "relevance": round(relevance, 4),
            "duplication": round(duplication, 4),
            "provenance_count": int(len(used_chunk_ids)),
        },
    }
