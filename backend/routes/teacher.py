from ..question_generator import generate_fallback_mcqs
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import random
import concurrent.futures
import threading
import time

from flask import Blueprint, jsonify, request, g, current_app
from sqlalchemy import and_, func, or_
from werkzeug.utils import secure_filename

from ..models import (
    Test,
    TestResult,
    TeacherIntervention,
    TeacherDocument,
    TeacherDocumentChunk,
    RagRetrievalEvent,
    User,
    Question,
    TestQuestion,
    AnswerLog,
    Classroom,
    ClassroomStudent,
    TestAssignment,
    db,
)
from ..security import require_auth, role_required
from ..validation import validate_required_fields, sanitize_string
from ..ai_topic_service import (
    generate_topic_mcqs,
    get_topic_ai_service_url,
    process_document_with_ai_service,
)
from ..at_risk_predictor import get_at_risk_predictions_for_students
from ..rag_service import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    assemble_context_text,
    attach_question_provenance,
    build_deterministic_chunks,
    build_embedding_payload,
    delete_document_from_r2,
    get_rag_async_min_bytes,
    get_rag_cleanup_batch_size,
    get_rag_max_chunk_candidates,
    get_rag_max_documents_per_teacher,
    get_rag_max_selected_docs,
    get_rag_max_storage_bytes_per_teacher,
    extract_document_text,
    get_rag_max_upload_bytes,
    get_rag_min_confidence,
    get_rag_retention_days,
    is_pgvector_enabled,
    is_allowed_document_extension,
    normalize_document_extension,
    resolve_document_storage_backend,
    resolve_vector_store_choice,
    score_chunks_for_query,
    store_pgvector_embeddings,
    summarize_retrieval_confidence,
    upload_document_to_r2,
)

teacher_bp = Blueprint("teacher", __name__)

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_GRADES = {"elementary", "middle", "high", "college"}
INTERVENTION_STATUSES = {"planned", "in_progress", "monitoring", "completed", "cancelled"}
UNASSIGNED_TEACHER_SCHOOL_ERROR = "Teacher is not assigned to a school yet. Ask an admin to assign school access."


def _utcnow():
    # Keep UTC semantics while storing naive datetimes used by existing models.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clamp_int(raw, default_value, min_value, max_value):
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default_value
    return max(min_value, min(max_value, value))


RAG_INGESTION_WORKERS = _clamp_int(os.environ.get("RAG_INGESTION_WORKERS", 2), 2, 1, 8)
RAG_INGESTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=RAG_INGESTION_WORKERS)
RAG_INGESTION_LOCK = threading.Lock()


def _clamp_float(raw, default_value, min_value, max_value):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default_value)
    return max(float(min_value), min(float(max_value), value))


def _parse_bool_flag(raw_value, default=False):
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return raw_value != 0
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _get_rag_ingestion_processor() -> str:
    raw = str(os.environ.get("RAG_INGESTION_PROCESSOR") or "ai_service").strip().lower()
    if raw in {"ai_service", "hf", "huggingface", "ai"}:
        return "ai_service"
    return "local"


def _is_rag_ingestion_processor_strict() -> bool:
    return _parse_bool_flag(os.environ.get("RAG_INGESTION_PROCESSOR_STRICT"), False)


def _parse_days_arg(default_value=30):
    return _clamp_int(request.args.get("days", default_value), default_value, 1, 3650)


def _parse_iso_datetime(raw_value):
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(str(raw_value))
    except (TypeError, ValueError):
        return None


def _parse_int_list(values):
    if not isinstance(values, list):
        return []

    normalized = []
    seen = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _teacher_document_storage_root() -> str:
    root = os.path.join(current_app.instance_path, "teacher_docs")
    os.makedirs(root, exist_ok=True)
    return root


def _teacher_document_storage_path(teacher_id: int, original_filename: str) -> str:
    safe_name = secure_filename(original_filename or "document.txt") or "document.txt"
    teacher_folder = os.path.join(_teacher_document_storage_root(), str(int(teacher_id)))
    os.makedirs(teacher_folder, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_suffix = f"{random.randint(1000, 9999)}"
    stored_name = f"{timestamp}_{random_suffix}_{safe_name}"
    return os.path.join(teacher_folder, stored_name)


def _safe_remove_local_file(path: str | None) -> None:
    if not path:
        return
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        current_app.logger.warning("Failed to remove local teacher document file: %s", path)


def _delete_teacher_document_storage(storage_path: str | None, metadata: dict | None) -> None:
    details = metadata if isinstance(metadata, dict) else {}
    backend = str(details.get("storage_backend") or "").strip().lower()
    path = str(storage_path or "").strip()

    if backend == "r2" or path.startswith("r2://"):
        try:
            delete_document_from_r2(path, details)
        except Exception:
            current_app.logger.warning("Failed to delete R2 teacher document object: %s", path)
        return

    if path:
        _safe_remove_local_file(path)


def _cleanup_expired_teacher_documents(
    teacher_id: int,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    retention_days = get_rag_retention_days()
    max_batch = limit if limit is not None else get_rag_cleanup_batch_size()
    batch_size = _clamp_int(max_batch, get_rag_cleanup_batch_size(), 1, 200)
    cutoff = _utcnow() - timedelta(days=retention_days)

    stale_docs = TeacherDocument.query.filter(
        TeacherDocument.teacher_id == int(teacher_id),
        TeacherDocument.uploaded_at < cutoff,
    ).order_by(
        TeacherDocument.uploaded_at.asc(),
    ).limit(batch_size).all()

    if dry_run:
        return {
            "retention_days": retention_days,
            "cutoff": cutoff.isoformat(),
            "deleted_count": 0,
            "candidate_count": len(stale_docs),
            "document_ids": [int(doc.id) for doc in stale_docs],
        }

    deleted_ids = []
    for doc in stale_docs:
        deleted_ids.append(int(doc.id))
        _delete_teacher_document_storage(doc.storage_path, doc.metadata_json if isinstance(doc.metadata_json, dict) else {})
        db.session.delete(doc)

    if deleted_ids:
        db.session.commit()

    return {
        "retention_days": retention_days,
        "cutoff": cutoff.isoformat(),
        "deleted_count": len(deleted_ids),
        "candidate_count": len(stale_docs),
        "document_ids": deleted_ids,
    }


def _enforce_teacher_document_quota(teacher_id: int, incoming_size_bytes: int) -> str | None:
    max_docs = get_rag_max_documents_per_teacher()
    max_storage = get_rag_max_storage_bytes_per_teacher()

    current_docs = TeacherDocument.query.filter(
        TeacherDocument.teacher_id == int(teacher_id),
    ).count()

    if current_docs >= max_docs:
        return (
            f"Document quota exceeded. Maximum {max_docs} documents per teacher is allowed. "
            "Delete older documents before uploading new ones."
        )

    used_storage = db.session.query(
        func.coalesce(func.sum(TeacherDocument.file_size_bytes), 0),
    ).filter(
        TeacherDocument.teacher_id == int(teacher_id),
    ).scalar() or 0

    projected = int(used_storage) + int(max(0, incoming_size_bytes))
    if projected > max_storage:
        limit_mb = int(max_storage / (1024 * 1024))
        used_mb = int(int(used_storage) / (1024 * 1024))
        return (
            f"Storage quota exceeded for your document library ({used_mb}MB/{limit_mb}MB used). "
            "Delete documents or upload a smaller file."
        )

    return None


def _process_teacher_document_ingestion(
    *,
    teacher_id: int,
    document_id: int,
    source_file_path: str,
    extension: str,
    content_sha: str,
    chunk_size: int,
    overlap: int,
    vector_store_choice: dict,
    cleanup_source_file: bool,
) -> dict:
    doc = TeacherDocument.query.filter_by(id=document_id, teacher_id=teacher_id).first()
    if not doc:
        raise ValueError("Document record not found for ingestion")

    try:
        text = extract_document_text(source_file_path, extension)
        if not text:
            raise ValueError("No readable text found in uploaded document.")

        processor_requested = _get_rag_ingestion_processor()
        processor_effective = "local"
        processor_error = None
        ai_ingestion = None

        if processor_requested == "ai_service":
            ai_ingestion = process_document_with_ai_service(
                document_text=text,
                content_sha256=content_sha,
                chunk_size=chunk_size,
                overlap=overlap,
                vector_store_requested=vector_store_choice.get("requested"),
            )
            if ai_ingestion.get("ok"):
                processor_effective = "ai_service"
            else:
                processor_error = str(ai_ingestion.get("error") or "AI ingestion failed")
                if _is_rag_ingestion_processor_strict():
                    raise ValueError(processor_error)
                current_app.logger.warning(
                    "AI ingestion failed; falling back to local processing (doc_id=%s): %s",
                    document_id,
                    processor_error,
                )

        prepared_chunks = []
        chunk_vectors = []
        total_tokens = 0

        if processor_effective == "ai_service" and ai_ingestion:
            for chunk in ai_ingestion.get("chunks") or []:
                if not isinstance(chunk, dict):
                    continue

                chunk_text = str(chunk.get("text") or "")
                if not chunk_text.strip():
                    continue

                chunk_id = sanitize_string(chunk.get("chunk_id") or "", max_length=64)
                if not chunk_id:
                    chunk_id = hashlib.sha256(
                        f"{content_sha}:{int(chunk.get('chunk_index') or 0)}:{chunk_text}".encode("utf-8")
                    ).hexdigest()[:40]

                vector = chunk.get("embedding_vector") if isinstance(chunk.get("embedding_vector"), list) else []
                try:
                    token_count = int(chunk.get("token_count") or 0)
                except (TypeError, ValueError):
                    token_count = 0
                token_count = max(0, token_count)
                total_tokens += token_count

                chunk_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
                chunk_metadata = {
                    "source": "teacher_upload_ai_service",
                    **chunk_metadata,
                }

                prepared_chunks.append(
                    {
                        "chunk_id": str(chunk_id),
                        "chunk_index": int(chunk.get("chunk_index") or 0),
                        "text": chunk_text,
                        "text_hash": sanitize_string(chunk.get("text_hash") or "", max_length=64)
                        or hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end"),
                        "token_count": token_count,
                        "embedding_vector": vector,
                        "embedding_dim": chunk.get("embedding_dim"),
                        "embedding_provider": sanitize_string(chunk.get("embedding_provider") or "", max_length=64) or None,
                        "embedding_model": sanitize_string(chunk.get("embedding_model") or "", max_length=128) or None,
                        "embedding_status": sanitize_string(chunk.get("embedding_status") or "embedded", max_length=32) or "embedded",
                        "metadata_json": chunk_metadata,
                    }
                )

                if chunk_id and isinstance(vector, list) and vector:
                    chunk_vectors.append(
                        {
                            "chunk_id": str(chunk_id),
                            "vector": vector,
                        }
                    )

            if total_tokens <= 0:
                try:
                    total_tokens = int(ai_ingestion.get("token_count") or 0)
                except (TypeError, ValueError):
                    total_tokens = 0

        if not prepared_chunks:
            processor_effective = "local"
            chunks = build_deterministic_chunks(
                text,
                document_fingerprint=content_sha,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if not chunks:
                raise ValueError("Document text is too short to chunk.")

            for chunk in chunks:
                embedding = build_embedding_payload(chunk.get("text") or "")
                token_count = int(chunk.get("token_count") or 0)
                total_tokens += token_count

                vector = embedding.get("embedding_vector") or []
                chunk_id = chunk.get("chunk_id")
                prepared_chunks.append(
                    {
                        "chunk_id": str(chunk_id),
                        "chunk_index": int(chunk.get("chunk_index") or 0),
                        "text": chunk.get("text") or "",
                        "text_hash": chunk.get("text_hash") or hashlib.sha256((chunk.get("text") or "").encode("utf-8")).hexdigest(),
                        "char_start": chunk.get("char_start"),
                        "char_end": chunk.get("char_end"),
                        "token_count": token_count,
                        "embedding_vector": vector if isinstance(vector, list) else [],
                        "embedding_dim": embedding.get("embedding_dim"),
                        "embedding_provider": embedding.get("embedding_provider"),
                        "embedding_model": embedding.get("embedding_model"),
                        "embedding_status": embedding.get("embedding_status") or "embedded",
                        "metadata_json": {
                            "source": "teacher_upload_local",
                            "embedding_error": embedding.get("embedding_error"),
                        },
                    }
                )

                if chunk_id and isinstance(vector, list) and vector:
                    chunk_vectors.append(
                        {
                            "chunk_id": str(chunk_id),
                            "vector": vector,
                        }
                    )

        if not prepared_chunks:
            raise ValueError("Document ingestion produced zero chunks.")

        TeacherDocumentChunk.query.filter_by(document_id=doc.id).delete(synchronize_session=False)
        for item in prepared_chunks:
            db.session.add(
                TeacherDocumentChunk(
                    document_id=doc.id,
                    teacher_id=teacher_id,
                    chunk_id=item.get("chunk_id"),
                    chunk_index=int(item.get("chunk_index") or 0),
                    text=item.get("text") or "",
                    text_hash=item.get("text_hash") or hashlib.sha256((item.get("text") or "").encode("utf-8")).hexdigest(),
                    char_start=item.get("char_start"),
                    char_end=item.get("char_end"),
                    token_count=int(item.get("token_count") or 0),
                    embedding_vector=item.get("embedding_vector") if isinstance(item.get("embedding_vector"), list) else [],
                    embedding_dim=item.get("embedding_dim"),
                    embedding_provider=item.get("embedding_provider"),
                    embedding_model=item.get("embedding_model"),
                    embedding_status=item.get("embedding_status") or "embedded",
                    vector_store=vector_store_choice.get("effective") or "python",
                    metadata_json=item.get("metadata_json") if isinstance(item.get("metadata_json"), dict) else {},
                )
            )

        db.session.flush()

        vector_store_effective = vector_store_choice.get("effective") or "python"
        vector_store_fallback_reason = vector_store_choice.get("fallback_reason")

        if ai_ingestion:
            ai_vector_store_effective = sanitize_string(ai_ingestion.get("vector_store_effective") or "", max_length=32)
            ai_vector_store_fallback_reason = sanitize_string(ai_ingestion.get("vector_store_fallback_reason") or "", max_length=160)
            if ai_vector_store_effective:
                vector_store_effective = ai_vector_store_effective
            if ai_vector_store_fallback_reason and not vector_store_fallback_reason:
                vector_store_fallback_reason = ai_vector_store_fallback_reason

        if vector_store_effective == "pgvector" and not is_pgvector_enabled():
            vector_store_effective = "python"
            vector_store_fallback_reason = vector_store_fallback_reason or "pgvector_not_available"

        if vector_store_effective == "pgvector" and is_pgvector_enabled():
            try:
                stored_count = store_pgvector_embeddings(doc.id, chunk_vectors)
                if stored_count <= 0:
                    vector_store_effective = "python"
                    vector_store_fallback_reason = "pgvector_write_noop"
            except Exception as exc:
                vector_store_effective = "python"
                vector_store_fallback_reason = f"pgvector_write_failed:{str(exc)[:160]}"

        metadata = doc.metadata_json if isinstance(doc.metadata_json, dict) else {}
        metadata.update(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "vector_store_requested": vector_store_choice.get("requested"),
                "vector_store_effective": vector_store_effective,
                "vector_store_fallback_reason": vector_store_fallback_reason,
                "ingestion_processor_requested": processor_requested,
                "ingestion_processor_effective": processor_effective,
                "ingestion_processor_error": processor_error,
                "ingestion_processor_endpoint": ai_ingestion.get("service_endpoint") if ai_ingestion else None,
                "ingestion_processor_latency_ms": ai_ingestion.get("service_latency_ms") if ai_ingestion else None,
                "ingestion_mode": metadata.get("ingestion_mode") or "sync",
                "ingested_at": _utcnow().isoformat(),
            }
        )

        doc.metadata_json = metadata
        doc.status = "processed"
        doc.error_message = None
        doc.chunk_count = len(prepared_chunks)
        doc.token_count = total_tokens
        doc.processed_at = _utcnow()
        doc.updated_at = _utcnow()
        db.session.commit()

        return {
            "document": doc.as_dict(),
            "chunk_count": len(prepared_chunks),
            "vector_store_effective": vector_store_effective,
            "vector_store_fallback_reason": vector_store_fallback_reason,
            "ingestion_processor_effective": processor_effective,
            "ingestion_processor_error": processor_error,
        }
    except Exception as exc:
        db.session.rollback()
        failed = TeacherDocument.query.filter_by(id=document_id, teacher_id=teacher_id).first()
        if failed:
            failed.status = "failed"
            failed.error_message = str(exc)[:1200]
            failed.updated_at = _utcnow()
            db.session.commit()
        raise
    finally:
        if cleanup_source_file:
            _safe_remove_local_file(source_file_path)


def _run_background_ingestion(app, **kwargs):
    with app.app_context():
        try:
            _process_teacher_document_ingestion(**kwargs)
        except Exception as exc:
            current_app.logger.exception("Background teacher document ingestion failed: %s", exc)


def _record_rag_retrieval_event(
    *,
    teacher_id: int,
    test_id: int | None,
    generation_status: dict,
    requested_count: int,
    generated_count: int,
) -> None:
    status_payload = generation_status if isinstance(generation_status, dict) else {}
    rag_metrics = status_payload.get("rag_metrics") if isinstance(status_payload.get("rag_metrics"), dict) else {}
    fallback_reason = (
        status_payload.get("rag_fallback_reason")
        or status_payload.get("vector_store_fallback_reason")
        or None
    )

    has_error = bool(status_payload.get("service_error")) and int(generated_count or 0) <= 0
    event_status = "error" if has_error else ("fallback" if fallback_reason else "success")

    event = RagRetrievalEvent(
        teacher_id=int(teacher_id),
        test_id=int(test_id) if test_id else None,
        generation_mode_requested=str(status_payload.get("generation_mode_requested") or "standard"),
        generation_mode_effective=str(status_payload.get("generation_mode_effective") or "standard"),
        vector_store_requested=status_payload.get("vector_store_requested"),
        vector_store_effective=status_payload.get("vector_store_effective"),
        status=event_status,
        fallback_reason=str(fallback_reason)[:128] if fallback_reason else None,
        error_message=str(status_payload.get("service_error") or "")[:1200] or None,
        selected_doc_count=len(status_payload.get("rag_selected_document_ids") or []),
        candidate_chunk_count=int(status_payload.get("rag_candidate_chunk_count") or 0),
        retrieval_count=int(status_payload.get("rag_retrieval_count") or 0),
        confidence=float(status_payload.get("rag_confidence") or 0.0),
        avg_similarity=float(status_payload.get("rag_avg_similarity") or 0.0),
        max_similarity=float(status_payload.get("rag_max_similarity") or 0.0),
        provenance_count=int(rag_metrics.get("provenance_count") or 0),
        coverage=float(rag_metrics.get("coverage") or 0.0),
        relevance=float(rag_metrics.get("relevance") or 0.0),
        duplication=float(rag_metrics.get("duplication") or 0.0),
        requested_count=int(requested_count or 0),
        generated_count=int(generated_count or 0),
        service_latency_ms=float(status_payload.get("service_latency_ms") or 0.0) if status_payload.get("service_latency_ms") is not None else None,
        metadata_json={
            "service_status_code": status_payload.get("service_status_code"),
            "service_endpoint": status_payload.get("service_endpoint"),
            "technical_fallback_used": bool(status_payload.get("technical_fallback_used")),
        },
    )
    db.session.add(event)


def _teacher_student_filter(teacher, grade=None):
    # Only apply grade filtering when explicitly requested by caller.
    # Default behavior should expose all students in the teacher's school scope.
    effective_grade = sanitize_string(grade, max_length=32) if grade else None
    teacher_school_id = _ensure_teacher_school(teacher)
    if not teacher_school_id:
        return User.query.filter(User.id == -1), effective_grade

    query = User.query.filter(
        User.role == "student",
        User.school_id == teacher_school_id,
    )
    if effective_grade:
        query = query.filter(User.grade == effective_grade)
    return query, effective_grade


def _ensure_teacher_school(teacher):
    """Return assigned teacher school_id without creating or attaching schools."""
    bound_teacher = User.query.filter_by(id=teacher.id).first()
    if not bound_teacher:
        return None

    if bound_teacher.school_id:
        if teacher.school_id != bound_teacher.school_id:
            teacher.school_id = bound_teacher.school_id
        return bound_teacher.school_id

    return None


def _normalize_grade_for_generation(value, fallback="high"):
    normalized = sanitize_string(value or "", max_length=32).lower()
    if normalized in VALID_GRADES:
        return normalized
    fallback_normalized = sanitize_string(fallback or "", max_length=32).lower()
    if fallback_normalized in VALID_GRADES:
        return fallback_normalized
    return "high"


def _scope_teacher_students(teacher, grade=None):
    students_query, effective_grade = _teacher_student_filter(teacher, grade=grade)
    students = students_query.order_by(User.name.asc()).all()
    return students, [int(s.id) for s in students], effective_grade


def _score_pct_expression():
    return func.coalesce(
        (TestResult.correct_answers * 100.0) / func.nullif(TestResult.total_questions, 0),
        0,
    )


def _topic_accuracy_rows(student_ids, cutoff, subject=None):
    if not student_ids:
        return []

    query = db.session.query(
        AnswerLog.user_id.label("student_id"),
        Question.subject.label("subject"),
        Question.syllabus_topic.label("topic"),
        func.count(AnswerLog.id).label("attempts"),
        func.avg(func.cast(AnswerLog.is_correct, db.Integer) * 100.0).label("accuracy"),
    ).join(
        Question,
        AnswerLog.question_id == Question.id,
    ).filter(
        AnswerLog.user_id.in_(student_ids),
        AnswerLog.answered_at >= cutoff,
        Question.syllabus_topic.isnot(None),
    )

    if subject:
        query = query.filter(Question.subject == subject)

    rows = query.group_by(
        AnswerLog.user_id,
        Question.subject,
        Question.syllabus_topic,
    ).all()

    return [
        {
            "student_id": int(row.student_id),
            "subject": str(row.subject or "").strip().lower() or None,
            "topic": str(row.topic or "").strip().lower() or None,
            "attempts": int(row.attempts or 0),
            "accuracy": round(float(row.accuracy or 0), 2),
        }
        for row in rows
    ]


def _select_subject_for_students(student_ids, cutoff, preferred_subject=None):
    preferred = sanitize_string(preferred_subject or "", max_length=64) or None
    if preferred:
        return preferred

    if not student_ids:
        return "science"

    rows = db.session.query(
        TestResult.subject,
        func.avg(_score_pct_expression()).label("avg_score"),
        func.count(TestResult.id).label("attempts"),
    ).filter(
        TestResult.user_id.in_(student_ids),
        TestResult.started_at >= cutoff,
    ).group_by(
        TestResult.subject,
    ).order_by(
        func.avg(_score_pct_expression()).asc(),
        func.count(TestResult.id).desc(),
    ).all()

    if rows:
        return sanitize_string(rows[0].subject or "science", max_length=64)
    return "science"


def _build_generated_intervention_test(
    teacher,
    *,
    title,
    subject,
    grade,
    difficulty,
    topic,
    question_count,
    time_limit,
    description,
    seed=None,
):
    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        raise ValueError(UNASSIGNED_TEACHER_SCHOOL_ERROR)

    resolved_grade = _normalize_grade_for_generation(grade, teacher.grade or "high")
    resolved_subject = sanitize_string(subject or "science", max_length=64)
    resolved_difficulty = sanitize_string(difficulty or "easy", max_length=16).lower()
    if resolved_difficulty not in VALID_DIFFICULTIES:
        resolved_difficulty = "easy"

    resolved_topic = sanitize_string(topic or "", max_length=128) or None
    resolved_title = sanitize_string(title or "Intervention Test", max_length=255) or "Intervention Test"
    resolved_description = sanitize_string(description or "", max_length=1000) or None

    test = Test(
        title=resolved_title,
        description=resolved_description,
        subject=resolved_subject,
        grade=resolved_grade,
        topic=resolved_topic,
        difficulty=resolved_difficulty,
        question_count=int(max(1, question_count)),
        time_limit=int(max(5, time_limit)),
        created_by=teacher.id,
        school_id=school_id,
        total_points=int(max(1, question_count)),
        is_published=True,
        is_active=True,
    )
    db.session.add(test)
    db.session.flush()

    questions, generation_status = _pick_or_generate_questions(
        teacher_id=teacher.id,
        subject=resolved_subject,
        grade=resolved_grade,
        difficulty=resolved_difficulty,
        topic=resolved_topic,
        count=int(max(1, question_count)),
        seed=seed,
        test_title=resolved_title,
        test_description=resolved_description,
    )

    if not questions:
        raise RuntimeError(
            generation_status.get("service_error") or
            "Question generation is temporarily unavailable"
        )

    for index, question in enumerate(questions, start=1):
        db.session.add(TestQuestion(test_id=test.id, question_id=question.id, order=index, points=1))

    warning = None
    if len(questions) < int(max(1, question_count)):
        warning = (
            f"Requested {int(max(1, question_count))} questions, "
            f"but generated {len(questions)}."
        )
    test.question_count = len(questions)
    test.total_points = len(questions)

    return test, generation_status, warning


def _build_assignments_for_students(
    teacher,
    *,
    test,
    student_ids,
    due_at,
    notes,
    is_mandatory=True,
    allow_late=False,
):
    if not student_ids:
        return []

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return []

    scoped_students = User.query.filter(
        User.id.in_([int(sid) for sid in student_ids]),
        User.role == "student",
        User.school_id == school_id,
    ).all()

    created = []
    for student in scoped_students:
        assignment = TestAssignment(
            test_id=test.id,
            student_id=student.id,
            assigned_by=teacher.id,
            notes=notes,
            due_at=due_at,
            is_mandatory=bool(is_mandatory),
            allow_late=bool(allow_late),
            require_camera=True,
            require_emotion=True,
            status="assigned",
        )
        db.session.add(assignment)
        created.append(assignment)

    return created


def _create_intervention_entry(
    teacher,
    *,
    action_type,
    title,
    notes,
    status,
    subject=None,
    topic=None,
    due_at=None,
    classroom_id=None,
    related_test_id=None,
    student_ids=None,
    assignment_ids=None,
    cluster_payload=None,
    metadata_json=None,
):
    intervention = TeacherIntervention(
        teacher_id=teacher.id,
        action_type=sanitize_string(action_type or "note", max_length=64) or "note",
        title=sanitize_string(title or "Intervention", max_length=255) or "Intervention",
        notes=sanitize_string(notes or "", max_length=4000) or None,
        status=sanitize_string(status or "planned", max_length=32).lower() or "planned",
        subject=sanitize_string(subject or "", max_length=64) or None,
        topic=sanitize_string(topic or "", max_length=128) or None,
        due_at=due_at,
        classroom_id=classroom_id,
        related_test_id=related_test_id,
        student_ids=student_ids or [],
        assignment_ids=assignment_ids or [],
        cluster_payload=cluster_payload or [],
        metadata_json=metadata_json or {},
    )
    db.session.add(intervention)
    db.session.flush()
    return intervention


def _get_teacher_classroom_or_404(teacher_id, classroom_id):
    classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=teacher_id).first()
    if not classroom:
        return None, (jsonify({"error": "Classroom not found"}), 404)
    return classroom, None


def _normalize_question_payload(item, subject, grade, difficulty, topic):
    text = sanitize_string(item.get("text"), max_length=4000)
    options = item.get("options") if isinstance(item.get("options"), list) else []
    options = [sanitize_string(str(opt), max_length=300) for opt in options][:6]
    options = [opt for opt in options if opt]

    if len(options) < 2:
        return None

    try:
        correct_index = int(item.get("correct_index", 0))
    except (TypeError, ValueError):
        correct_index = 0
    if correct_index < 0 or correct_index >= len(options):
        correct_index = 0

    return {
        "subject": subject,
        "grade": grade,
        "difficulty": difficulty,
        "text": text,
        "options": options,
        "correct_index": correct_index,
        "hint": sanitize_string(item.get("hint") or "", max_length=500) or None,
        "explanation": sanitize_string(item.get("explanation") or "", max_length=1500) or None,
        "syllabus_topic": sanitize_string(topic or item.get("topic") or "", max_length=128) or None,
        "is_generated": True,
    }


def _question_identity_key(text, options):
    normalized_text = " ".join(str(text or "").strip().lower().split())
    normalized_options = "|".join(
        sorted(" ".join(str(opt or "").strip().lower().split()) for opt in (options or []))
    )
    return f"{normalized_text}|{normalized_options}"


def _normalize_preview_reuse_payload(
    preview_questions,
    subject,
    grade,
    difficulty,
    topic,
    max_count,
):
    if not isinstance(preview_questions, list) or max_count <= 0:
        return []

    rows = []
    seen = set()
    for item in preview_questions:
        if not isinstance(item, dict):
            continue

        normalized = _normalize_question_payload(item, subject, grade, difficulty, topic)
        if not normalized:
            continue

        payload_row = {
            "text": normalized["text"],
            "options": normalized["options"],
            "correct_index": normalized["correct_index"],
            "hint": normalized["hint"],
            "explanation": normalized["explanation"],
            "topic": normalized["syllabus_topic"],
            "source": sanitize_string(item.get("source") or "preview_reuse", max_length=64) or "preview_reuse",
        }

        key = _question_identity_key(payload_row["text"], payload_row["options"])
        if not key or key in seen:
            continue

        seen.add(key)
        rows.append(payload_row)

        if len(rows) >= max_count:
            break

    return rows


def _build_preview_signature(*, subject, grade, difficulty, topic, count, seed, questions):
    canonical_payload = {
        "subject": str(subject or "").strip().lower(),
        "grade": str(grade or "").strip().lower(),
        "difficulty": str(difficulty or "").strip().lower(),
        "topic": str(topic or "").strip().lower(),
        "count": int(max(0, count or 0)),
        "seed": seed,
        "questions": [
            {
                "text": str(item.get("text") or "").strip(),
                "options": [str(opt or "").strip() for opt in (item.get("options") or [])],
                "correct_index": int(item.get("correct_index") or 0),
                "topic": str(item.get("topic") or "").strip(),
                "source": str(item.get("source") or "").strip().lower(),
            }
            for item in (questions or [])
            if isinstance(item, dict)
        ],
    }
    raw = json.dumps(canonical_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _build_generated_question_records(
    generated_questions,
    subject,
    grade,
    difficulty,
    topic,
    generated_by,
    seed,
    default_source="teacher_dashboard",
    generation_meta_extras=None,
):
    records = []
    generated_at = _utcnow().isoformat()
    for item in generated_questions:
        normalized = _normalize_question_payload(item, subject, grade, difficulty, topic)
        if not normalized:
            continue

        source = sanitize_string(item.get("source") or default_source, max_length=64) or default_source
        generation_meta = {
            "generated_at": generated_at,
            "generator_version": "teacher-v3",
            "seed": seed,
            "source": source,
        }

        item_provenance = item.get("provenance") if isinstance(item, dict) else None
        if isinstance(item_provenance, dict) and item_provenance:
            generation_meta["provenance"] = item_provenance

        item_retrieval_trace = item.get("retrieval_trace") if isinstance(item, dict) else None
        if isinstance(item_retrieval_trace, list) and item_retrieval_trace:
            generation_meta["retrieval_trace"] = item_retrieval_trace

        if generation_meta_extras:
            generation_meta.update(generation_meta_extras)

        record = Question(
            **normalized,
            generated_by=generated_by,
            generation_meta=generation_meta,
        )
        records.append(record)
    return records

def generate_topic_mcqs_concurrently(subject, grade, difficulty, topic, count, seed=None, test_title=None, test_description=None, max_workers=2):
    """
    Dynamically splits a question generation request to maximize parallel processing.
    Adapts chunk sizes automatically based on total count and max_workers.
    """
    # 1. Dynamic Load Balancing Algorithm
    chunks = []
    if count > 0:
        base_chunk = count // max_workers
        remainder = count % max_workers
        
        if base_chunk == 0:
            # E.g., user asks for 3 questions, max_workers is 5. 
            # We just create 3 workers handling 1 question each.
            chunks = [1] * count
        else:
            # E.g., user asks for 30 questions -> 5 chunks of 6.
            # E.g., user asks for 12 questions -> 2 chunks of 3, and 3 chunks of 2.
            chunks = [base_chunk + 1 if i < remainder else base_chunk for i in range(max_workers)]
            
    # Filter out any zero-sized chunks for safety
    chunks = [c for c in chunks if c > 0]
        
    combined_result = {
        "ok": True,
        "questions": [],
        "error": None,
        "status_code": 200,
        "service_endpoint": get_topic_ai_service_url(),
        "service_latency_ms": 0.0,
        "meta": {
            "llm_count": 0,
            "template_count": 0,
            "cache_hit": False
        }
    }
    
    start_time = time.time()
    
    # 2. Fire dynamic requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = []
        for index, c_count in enumerate(chunks):
            # Dynamic seed injection to ensure Qwen generates totally unique batches
            chunk_seed = (seed + index) if seed is not None else random.randint(1000, 99999)
            
            futures.append(executor.submit(
                generate_topic_mcqs,
                subject=subject,
                grade=grade,
                difficulty=difficulty,
                topic=topic,
                count=c_count,
                seed=chunk_seed,
                test_title=test_title,
                test_description=test_description
            ))
            
        # 3. Gather and merge results as they finish
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res.get("ok"):
                    combined_result["questions"].extend(res.get("questions", []))
                    
                    # Aggregate the metadata for accuracy
                    meta = res.get("meta", {})
                    combined_result["meta"]["llm_count"] += int(meta.get("llm_count", 0))
                    combined_result["meta"]["template_count"] += int(meta.get("template_count", 0))
                    # If any chunk hit the cache, mark true
                    combined_result["meta"]["cache_hit"] = combined_result["meta"]["cache_hit"] or bool(meta.get("cache_hit", False))
                else:
                    # If one chunk fails, log it but don't crash the others
                    current_app.logger.warning(f"Chunk failed: {res.get('error')}")
                    combined_result["ok"] = False
                    combined_result["error"] = res.get("error", "Partial chunk generation failed")
                    combined_result["status_code"] = res.get("status_code", 500)
            except Exception as e:
                current_app.logger.error(f"Thread execution error: {str(e)}")
                combined_result["ok"] = False
                combined_result["error"] = str(e)
                combined_result["status_code"] = 500
                
    # 4. Finalize
    combined_result["service_latency_ms"] = round((time.time() - start_time) * 1000, 2)
    return combined_result

def _pick_or_generate_questions(
    teacher_id,
    subject,
    grade,
    difficulty,
    topic,
    count,
    seed=None,
    test_title=None,
    test_description=None,
    generation_mode="standard",
    rag_options=None,
):
    pool = []
    normalized_mode = sanitize_string(generation_mode or "standard", max_length=16).lower()
    if normalized_mode not in {"standard", "rag"}:
        normalized_mode = "standard"

    rag_options = rag_options if isinstance(rag_options, dict) else {}
    selected_document_ids = _parse_int_list(rag_options.get("selected_document_ids"))
    rag_top_k = _clamp_int(rag_options.get("top_k", 4), 4, 1, 12)
    rag_min_confidence = _clamp_float(
        rag_options.get("min_confidence", get_rag_min_confidence()),
        get_rag_min_confidence(),
        0.0,
        1.0,
    )
    vector_store_choice = resolve_vector_store_choice(rag_options.get("vector_store"))

    generation_status = {
        "requested_count": count,
        "generated_count": 0,
        "service_generated_count": 0,
        "service_llm_count": 0,
        "service_template_count": 0,
        "service_cache_hit": False,
        "technical_sample_count": 0,
        "service_error": None,
        "service_status_code": None,
        "service_endpoint": None,
        "service_latency_ms": None,
        "source_mode": "strict_llm",
        "technical_fallback_used": False,
        "generation_mode_requested": normalized_mode,
        "generation_mode_effective": "standard",
        "rag_selected_document_ids": selected_document_ids,
        "rag_candidate_chunk_count": 0,
        "rag_top_k": rag_top_k,
        "rag_retrieval_count": 0,
        "rag_confidence": 0.0,
        "rag_min_confidence": rag_min_confidence,
        "rag_fallback_reason": None,
        "rag_error": None,
        "rag_metrics": {
            "coverage": 0.0,
            "relevance": 0.0,
            "duplication": 0.0,
            "provenance_count": 0,
        },
        "vector_store_requested": vector_store_choice.get("requested"),
        "vector_store_effective": vector_store_choice.get("effective"),
        "vector_store_fallback_reason": vector_store_choice.get("fallback_reason"),
    }

    retrieved_chunks = []
    rag_context = None
    effective_generation_mode = "standard"

    if normalized_mode == "rag":
        try:
            max_doc_window = get_rag_max_selected_docs()
            max_chunk_candidates = get_rag_max_chunk_candidates()

            doc_query = TeacherDocument.query.filter(
                TeacherDocument.teacher_id == teacher_id,
                TeacherDocument.status == "processed",
            )
            if selected_document_ids:
                effective_document_ids = selected_document_ids[:max_doc_window]
            else:
                effective_document_ids = [
                    int(row.id)
                    for row in doc_query.with_entities(TeacherDocument.id)
                    .order_by(TeacherDocument.uploaded_at.desc())
                    .limit(max_doc_window)
                    .all()
                ]

            generation_status["rag_selected_document_ids"] = effective_document_ids

            query_text = " ".join(
                item
                for item in [
                    sanitize_string(subject or "", max_length=64),
                    sanitize_string(grade or "", max_length=32),
                    sanitize_string(difficulty or "", max_length=16),
                    sanitize_string(topic or "", max_length=128),
                    sanitize_string(test_title or "", max_length=255),
                    sanitize_string(test_description or "", max_length=1000),
                ]
                if item
            )

            if vector_store_choice.get("effective") == "pgvector":
                retrieved_chunks = score_chunks_for_query(
                    query_text,
                    [],
                    top_k=rag_top_k,
                    vector_store="pgvector",
                    teacher_id=teacher_id,
                    selected_document_ids=effective_document_ids,
                )
                if not retrieved_chunks:
                    generation_status["vector_store_effective"] = "python"
                    generation_status["vector_store_fallback_reason"] = "pgvector_retrieval_empty"

            if not retrieved_chunks:
                chunk_query = TeacherDocumentChunk.query.join(
                    TeacherDocument,
                    TeacherDocumentChunk.document_id == TeacherDocument.id,
                ).filter(
                    TeacherDocumentChunk.teacher_id == teacher_id,
                    TeacherDocument.teacher_id == teacher_id,
                    TeacherDocument.status == "processed",
                )

                if effective_document_ids:
                    chunk_query = chunk_query.filter(TeacherDocumentChunk.document_id.in_(effective_document_ids))

                chunk_rows = chunk_query.order_by(
                    TeacherDocumentChunk.document_id.asc(),
                    TeacherDocumentChunk.chunk_index.asc(),
                ).limit(max_chunk_candidates).all()
                generation_status["rag_candidate_chunk_count"] = len(chunk_rows)

                retrieved_chunks = score_chunks_for_query(
                    query_text,
                    chunk_rows,
                    top_k=rag_top_k,
                    vector_store="python",
                )
            else:
                generation_status["rag_candidate_chunk_count"] = len(retrieved_chunks)

            confidence_summary = summarize_retrieval_confidence(retrieved_chunks)
            generation_status["rag_retrieval_count"] = len(retrieved_chunks)
            generation_status["rag_confidence"] = confidence_summary.get("confidence", 0.0)
            generation_status["rag_avg_similarity"] = confidence_summary.get("avg_similarity", 0.0)
            generation_status["rag_max_similarity"] = confidence_summary.get("max_similarity", 0.0)

            if retrieved_chunks and confidence_summary.get("confidence", 0.0) >= rag_min_confidence:
                rag_context = assemble_context_text(retrieved_chunks)
                if rag_context:
                    effective_generation_mode = "rag"
                    generation_status["source_mode"] = "rag_grounded"
                    generation_status["generation_mode_effective"] = "rag"
                else:
                    generation_status["rag_fallback_reason"] = "empty_retrieval_context"
            elif not retrieved_chunks:
                generation_status["rag_fallback_reason"] = "no_relevant_documents_found"
            else:
                generation_status["rag_fallback_reason"] = "low_retrieval_confidence"
        except Exception as exc:
            generation_status["rag_error"] = str(exc)
            generation_status["rag_fallback_reason"] = "retrieval_error"

    if generation_status["generation_mode_effective"] != "rag":
        effective_generation_mode = "standard"

    service_result = generate_topic_mcqs(
        subject=subject,
        grade=grade,
        difficulty=difficulty,
        topic=topic,
        count=count,
        seed=seed,
        test_title=test_title,
        test_description=test_description,
        rag_context=rag_context,
        generation_mode=effective_generation_mode,
    )

    service_meta = service_result.get("meta") if isinstance(service_result.get("meta"), dict) else {}
    generation_status["service_status_code"] = service_result.get("status_code")
    generation_status["service_error"] = service_result.get("error")
    generation_status["service_endpoint"] = service_result.get("service_endpoint")
    generation_status["service_latency_ms"] = service_result.get("service_latency_ms")
    generation_status["service_llm_count"] = int(service_meta.get("llm_count") or 0)
    generation_status["service_template_count"] = int(service_meta.get("template_count") or 0)
    generation_status["service_cache_hit"] = bool(service_meta.get("cache_hit"))

    service_payload = []
    if service_result.get("ok"):
        for row in service_result.get("questions") or []:
            if not isinstance(row, dict):
                continue
            service_payload.append({
                **row,
                "source": "topic_ai_service_rag" if effective_generation_mode == "rag" else "topic_ai_service",
            })
    elif service_result.get("error"):
        current_app.logger.warning(
            "Topic AI service call failed during test generation: %s",
            service_result.get("error"),
        )

    if effective_generation_mode == "rag" and service_payload and retrieved_chunks:
        enriched = attach_question_provenance(service_payload, retrieved_chunks)
        service_payload = enriched.get("questions") or service_payload
        rag_metrics = enriched.get("metrics") or {}
        generation_status["rag_metrics"] = {
            "coverage": float(rag_metrics.get("coverage", 0.0) or 0.0),
            "relevance": float(rag_metrics.get("relevance", 0.0) or 0.0),
            "duplication": float(rag_metrics.get("duplication", 0.0) or 0.0),
            "provenance_count": int(rag_metrics.get("provenance_count", 0) or 0),
        }

    generated_records = _build_generated_question_records(
        service_payload,
        subject,
        grade,
        difficulty,
        topic,
        teacher_id,
        seed,
        default_source="topic_ai_service",
        generation_meta_extras={
            "service_meta": service_meta,
            "test_title": sanitize_string(test_title or "", max_length=255) or None,
            "test_description": sanitize_string(test_description or "", max_length=1000) or None,
            "generation_mode_requested": normalized_mode,
            "generation_mode_effective": effective_generation_mode,
            "rag_selected_document_ids": generation_status.get("rag_selected_document_ids") or selected_document_ids,
            "rag_candidate_chunk_count": generation_status.get("rag_candidate_chunk_count"),
            "rag_retrieval_count": generation_status.get("rag_retrieval_count"),
            "rag_confidence": generation_status.get("rag_confidence"),
            "rag_metrics": generation_status.get("rag_metrics") or {},
            "vector_store_requested": generation_status.get("vector_store_requested"),
            "vector_store_effective": generation_status.get("vector_store_effective"),
        },
    )

    for record in generated_records:
        db.session.add(record)
    if generated_records:
        db.session.flush()
        pool.extend(generated_records)
    generation_status["service_generated_count"] = len(generated_records)

    if not service_result.get("ok") and len(pool) < count:
        sample_payload = generate_fallback_mcqs(
            subject=subject,
            grade=grade,
            difficulty=difficulty,
            topic=topic,
            count=count - len(pool)
        )
        sample_records = _build_generated_question_records(
            sample_payload,
            subject,
            grade,
            difficulty,
            topic,
            teacher_id,
            seed,
            default_source="technical_sample_fallback",
            generation_meta_extras={
                "fallback_reason": "topic_ai_service_technical_failure",
                "service_error": service_result.get("error"),
                "service_status_code": service_result.get("status_code"),
                "generation_mode_requested": normalized_mode,
                "generation_mode_effective": effective_generation_mode,
            },
        )
        for record in sample_records:
            db.session.add(record)
        if sample_records:
            db.session.flush()
            pool.extend(sample_records)
            generation_status["technical_sample_count"] = len(sample_records)
            generation_status["technical_fallback_used"] = True

    final_pool = pool[:count]
    generation_status["generated_count"] = len(final_pool)
    generation_status["generation_mode_effective"] = effective_generation_mode
    return final_pool, generation_status


@teacher_bp.get("/documents")
@require_auth
@role_required("teacher")
def list_teacher_documents():
    teacher = g.current_user
    docs = TeacherDocument.query.filter_by(teacher_id=teacher.id).order_by(TeacherDocument.uploaded_at.desc()).all()
    return jsonify({
        "count": len(docs),
        "documents": [item.as_dict() for item in docs],
    }), 200


@teacher_bp.post("/documents/upload")
@require_auth
@role_required("teacher")
def upload_teacher_document():
    teacher = g.current_user
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"error": "A file is required."}), 400

    original_filename = str(upload.filename or "").strip()
    extension = normalize_document_extension(original_filename)
    if not is_allowed_document_extension(original_filename):
        allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed: {allowed}"}), 400

    raw_bytes = upload.read()
    if not raw_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    max_bytes = get_rag_max_upload_bytes()
    if len(raw_bytes) > max_bytes:
        return jsonify({"error": f"File exceeds max size of {int(max_bytes / (1024 * 1024))} MB."}), 400

    try:
        _cleanup_expired_teacher_documents(teacher.id, dry_run=False)
    except Exception as cleanup_exc:
        db.session.rollback()
        current_app.logger.warning("Teacher document cleanup skipped due to error: %s", cleanup_exc)

    content_sha = hashlib.sha256(raw_bytes).hexdigest()
    duplicate = TeacherDocument.query.filter_by(
        teacher_id=teacher.id,
        content_sha256=content_sha,
    ).order_by(TeacherDocument.uploaded_at.desc()).first()
    if duplicate and duplicate.status in {"processing", "processed"}:
        return jsonify({
            "message": "Document already indexed.",
            "deduplicated": True,
            "document": duplicate.as_dict(),
        }), 200

    quota_error = _enforce_teacher_document_quota(teacher.id, len(raw_bytes))
    if quota_error:
        return jsonify({"error": quota_error}), 409

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    temp_storage_path = _teacher_document_storage_path(teacher.id, original_filename)
    storage_resolution = resolve_document_storage_backend(request.form.get("document_storage"))
    if storage_resolution.get("effective") == "invalid":
        return jsonify({
            "error": "R2 storage was requested but is not available in strict mode.",
            "reason": storage_resolution.get("fallback_reason"),
        }), 503

    uploaded_r2 = None
    storage_path = temp_storage_path

    requested_title = request.form.get("title") or os.path.splitext(original_filename)[0]
    title = sanitize_string(requested_title, max_length=255) or "Teacher Document"
    content_type = sanitize_string(upload.mimetype or "", max_length=128) or None
    chunk_size = _clamp_int(request.form.get("chunk_size", 1200), 1200, 300, 3000)
    overlap = _clamp_int(request.form.get("overlap", 220), 220, 0, int(chunk_size / 2))
    vector_store_choice = resolve_vector_store_choice(request.form.get("vector_store"))

    force_async = _parse_bool_flag(request.form.get("async_ingestion"), False)
    should_async = force_async or len(raw_bytes) >= get_rag_async_min_bytes()

    try:
        with open(temp_storage_path, "wb") as handle:
            handle.write(raw_bytes)

        storage_metadata = {
            "storage_backend_requested": storage_resolution.get("requested"),
            "storage_backend_effective": storage_resolution.get("effective"),
            "storage_backend_fallback_reason": storage_resolution.get("fallback_reason"),
            "ingestion_mode": "async" if should_async else "sync",
        }

        if storage_resolution.get("effective") == "r2":
            uploaded_r2 = upload_document_to_r2(
                raw_bytes,
                teacher_id=teacher.id,
                teacher_name=teacher.name,
                original_filename=original_filename,
                content_sha256=content_sha,
                content_type=content_type,
            )
            storage_path = str(uploaded_r2.get("storage_path") or storage_path)
            storage_metadata.update(
                {
                    "storage_backend": "r2",
                    "r2_bucket": uploaded_r2.get("bucket"),
                    "r2_key": uploaded_r2.get("key"),
                    "r2_endpoint": uploaded_r2.get("endpoint"),
                    "r2_public_url": uploaded_r2.get("public_url"),
                }
            )
        else:
            storage_metadata["storage_backend"] = "local"

        document = TeacherDocument(
            teacher_id=teacher.id,
            school_id=school_id,
            title=title,
            filename=original_filename,
            file_ext=extension,
            content_type=content_type,
            file_size_bytes=len(raw_bytes),
            content_sha256=content_sha,
            storage_path=storage_path,
            status="processing",
            metadata_json={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "vector_store_requested": vector_store_choice.get("requested"),
                "vector_store_effective": vector_store_choice.get("effective"),
                "vector_store_fallback_reason": vector_store_choice.get("fallback_reason"),
                **storage_metadata,
            },
        )
        db.session.add(document)
        db.session.commit()

        ingestion_kwargs = {
            "teacher_id": int(teacher.id),
            "document_id": int(document.id),
            "source_file_path": temp_storage_path,
            "extension": extension,
            "content_sha": content_sha,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "vector_store_choice": vector_store_choice,
            "cleanup_source_file": storage_resolution.get("effective") == "r2",
        }

        if should_async:
            app_obj = current_app._get_current_object()
            with RAG_INGESTION_LOCK:
                RAG_INGESTION_EXECUTOR.submit(_run_background_ingestion, app_obj, **ingestion_kwargs)

            return jsonify({
                "message": "Document uploaded. Background indexing started.",
                "queued": True,
                "document": document.as_dict(),
            }), 202

        ingestion_result = _process_teacher_document_ingestion(**ingestion_kwargs)
        refreshed_document = TeacherDocument.query.filter_by(id=document.id, teacher_id=teacher.id).first()
        return jsonify({
            "message": "Document uploaded and indexed successfully.",
            "queued": False,
            "document": (refreshed_document or document).as_dict(),
            "ingestion": {
                "chunk_count": ingestion_result.get("chunk_count"),
                "vector_store_effective": ingestion_result.get("vector_store_effective"),
                "vector_store_fallback_reason": ingestion_result.get("vector_store_fallback_reason"),
            },
        }), 201
    except ValueError as exc:
        current_app.logger.warning(
            "Teacher document upload validation failed (teacher_id=%s, filename=%s): %s",
            teacher.id,
            original_filename,
            exc,
        )
        if uploaded_r2:
            try:
                delete_document_from_r2(
                    str(uploaded_r2.get("storage_path") or ""),
                    {
                        "r2_bucket": uploaded_r2.get("bucket"),
                        "r2_key": uploaded_r2.get("key"),
                    },
                )
            except Exception:
                current_app.logger.warning("Failed to rollback R2 upload after validation error.")

        _safe_remove_local_file(temp_storage_path)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception(
            "Teacher document upload failed (teacher_id=%s, filename=%s)",
            teacher.id,
            original_filename,
        )
        db.session.rollback()
        if uploaded_r2:
            try:
                delete_document_from_r2(
                    str(uploaded_r2.get("storage_path") or ""),
                    {
                        "r2_bucket": uploaded_r2.get("bucket"),
                        "r2_key": uploaded_r2.get("key"),
                    },
                )
            except Exception:
                current_app.logger.warning("Failed to rollback R2 upload after server error.")

        _safe_remove_local_file(temp_storage_path)
        return jsonify({"error": f"Failed to upload document: {str(exc)}"}), 500


@teacher_bp.delete("/documents/<int:document_id>")
@require_auth
@role_required("teacher")
def delete_teacher_document(document_id):
    teacher = g.current_user
    document = TeacherDocument.query.filter_by(id=document_id, teacher_id=teacher.id).first()
    if not document:
        return jsonify({"error": "Document not found"}), 404

    storage_path = document.storage_path
    storage_metadata = document.metadata_json if isinstance(document.metadata_json, dict) else {}
    try:
        db.session.delete(document)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete document: {str(exc)}"}), 500

    _delete_teacher_document_storage(storage_path, storage_metadata)

    return jsonify({"message": "Document deleted successfully", "document_id": document_id}), 200


@teacher_bp.post("/documents/cleanup")
@require_auth
@role_required("teacher")
def cleanup_teacher_documents():
    teacher = g.current_user
    payload = request.get_json(silent=True) if request.is_json else {}
    if not isinstance(payload, dict):
        payload = {}

    dry_run = _parse_bool_flag(payload.get("dry_run", request.args.get("dry_run")), False)
    limit = _clamp_int(payload.get("limit", request.args.get("limit", get_rag_cleanup_batch_size())), get_rag_cleanup_batch_size(), 1, 200)

    try:
        result = _cleanup_expired_teacher_documents(
            teacher.id,
            dry_run=dry_run,
            limit=limit,
        )
        return jsonify({
            "message": "Cleanup completed" if not dry_run else "Cleanup dry-run completed",
            **result,
        }), 200
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to cleanup documents: {str(exc)}"}), 500


@teacher_bp.get("/dashboard")
@require_auth
@role_required("teacher")
def teacher_dashboard():
    """Get teacher dashboard data with scoped school/grade metrics."""
    teacher = g.current_user
    days = _parse_days_arg(30)
    include_at_risk = _parse_bool_flag(request.args.get("include_at_risk"), False)
    cutoff = _utcnow() - timedelta(days=days)

    students_query, effective_grade = _teacher_student_filter(teacher)
    students = students_query.order_by(User.name.asc()).all()
    student_ids = [student.id for student in students]

    my_tests_query = Test.query.filter_by(created_by=teacher.id)
    my_tests = my_tests_query.order_by(Test.created_at.desc()).all()

    recent_results = []
    total_attempts = 0
    average_score = 0.0

    if student_ids:
        recent_results = TestResult.query.filter(
            TestResult.user_id.in_(student_ids),
            TestResult.started_at >= cutoff,
        ).order_by(TestResult.started_at.desc()).limit(12).all()

        all_period_results = TestResult.query.filter(
            TestResult.user_id.in_(student_ids),
            TestResult.started_at >= cutoff,
        ).all()

        total_attempts = len(all_period_results)
        if total_attempts > 0:
            pct_values = [
                (r.correct_answers / r.total_questions * 100)
                for r in all_period_results
                if r.total_questions and r.total_questions > 0
            ]
            average_score = round(sum(pct_values) / len(pct_values), 2) if pct_values else 0.0

    published_tests = sum(1 for test in my_tests if test.is_published)
    active_assignments = TestAssignment.query.filter(
        TestAssignment.assigned_by == teacher.id,
        TestAssignment.status.in_(["assigned", "started"]),
    ).count()
    classrooms_count = Classroom.query.filter_by(teacher_id=teacher.id).count()

    at_risk_payload = {"at_risk_students": [], "meta": {"reason": "skipped_for_dashboard_performance"}}
    if include_at_risk:
        try:
            # Compute predictions only for the small sample shown on the landing dashboard.
            # Avoid SHAP on initial dashboard load to keep first-request latency low.
            sample_student_ids = [s.id for s in students[:8]]
            at_risk_payload = get_at_risk_predictions_for_students(
                sample_student_ids,
                cutoff=cutoff,
                top_k_shap=0,
            )
        except Exception:
            at_risk_payload = {"at_risk_students": [], "meta": {"reason": "at_risk_failed"}}

    at_risk_by_id = {int(x.get("student_id")): x for x in at_risk_payload.get("at_risk_students", []) if x.get("student_id") is not None}
    return jsonify({
        "period_days": days,
        "teacher": {
            "id": teacher.id,
            "name": teacher.name,
            "grade": effective_grade,
            "school_id": teacher.school_id,
        },
        "students": {
            "total": len(students),
            "sample": [
                {
                    **student.as_dict(),
                    "at_risk": at_risk_by_id.get(int(student.id), {}).get("at_risk_probability") if at_risk_by_id.get(int(student.id)) else None,
                    "at_risk_explanation": at_risk_by_id.get(int(student.id), {}).get("explanation") if at_risk_by_id.get(int(student.id)) else None,
                }
                for student in students[:8]
            ],
        },
        "tests": {
            "total": len(my_tests),
            "published": published_tests,
            "drafts": max(0, len(my_tests) - published_tests),
            "recent": [test.as_dict() for test in my_tests[:6]],
        },
        "assignments": {
            "active": active_assignments,
            "classrooms": classrooms_count,
        },
        "performance": {
            "total_attempts": total_attempts,
            "average_score": average_score,
            "recent_results": [result.as_dict() for result in recent_results],
        },
        "at_risk_meta": at_risk_payload.get("meta") or {},
    }), 200


@teacher_bp.post("/question-bank/generate")
@require_auth
@role_required("teacher")
def teacher_generate_question_bank():
    """Generate question payload with optional persistence for teacher workflows."""
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    error = validate_required_fields(data, ["subject", "grade", "difficulty", "count"])
    if error:
        return jsonify({"error": error}), 400

    subject = sanitize_string(data.get("subject"), max_length=64)
    grade = sanitize_string(data.get("grade"), max_length=32).lower()
    difficulty = sanitize_string(data.get("difficulty"), max_length=16).lower()
    topic = sanitize_string(data.get("topic") or "", max_length=128) or None
    test_title = sanitize_string(data.get("title") or "", max_length=255) or None
    test_description = sanitize_string(data.get("description") or "", max_length=1000) or None
    count = _clamp_int(data.get("count", 10), 10, 1, 50)
    persist = _parse_bool_flag(data.get("persist"), False)

    seed_raw = data.get("seed")
    seed = None
    if seed_raw is not None and str(seed_raw).strip() != "":
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "seed must be an integer"}), 400

    if difficulty not in VALID_DIFFICULTIES:
        return jsonify({"error": "Difficulty must be one of: easy, medium, hard"}), 400
    if grade not in VALID_GRADES:
        return jsonify({"error": "Grade must be one of: elementary, middle, high, college"}), 400

    service_result = generate_topic_mcqs(
        subject=subject,
        grade=grade,
        difficulty=difficulty,
        topic=topic,
        count=count,
        seed=seed,
        test_title=test_title,
        test_description=test_description,
        generation_mode="standard",
    )

    service_meta = service_result.get("meta") if isinstance(service_result.get("meta"), dict) else {}
    generation_status = {
        "requested_count": count,
        "generated_count": 0,
        "service_generated_count": 0,
        "service_llm_count": int(service_meta.get("llm_count") or 0),
        "service_template_count": int(service_meta.get("template_count") or 0),
        "service_cache_hit": bool(service_meta.get("cache_hit")),
        "technical_sample_count": 0,
        "technical_fallback_used": False,
        "service_error": service_result.get("error"),
        "service_status_code": service_result.get("status_code"),
        "service_endpoint": service_result.get("service_endpoint"),
        "service_latency_ms": service_result.get("service_latency_ms"),
        "source_mode": "strict_llm",
    }

    generated_payload = []
    if service_result.get("ok"):
        generated_payload = [
            {
                **row,
                "source": "topic_ai_service",
            }
            for row in (service_result.get("questions") or [])
            if isinstance(row, dict)
        ]
        generation_status["service_generated_count"] = len(generated_payload)
    elif service_result.get("error"):
        current_app.logger.warning(
            "Topic AI service call failed for question bank generation: %s",
            service_result.get("error"),
        )

    if not service_result.get("ok") and len(generated_payload) < count:
        sample_payload = generate_fallback_mcqs(
            subject=subject,
            grade=grade,
            difficulty=difficulty,
            topic=topic,
            count=count - len(generated_payload)
        )
        generated_payload.extend(sample_payload)
        generation_status["technical_sample_count"] = len(sample_payload)
        generation_status["technical_fallback_used"] = len(sample_payload) > 0

    normalized_preview = []
    normalized_payload = []
    for item in generated_payload:
        normalized = _normalize_question_payload(item, subject, grade, difficulty, topic)
        if not normalized:
            continue

        normalized_preview.append({
            "subject": normalized["subject"],
            "grade": normalized["grade"],
            "difficulty": normalized["difficulty"],
            "text": normalized["text"],
            "options": normalized["options"],
            "correct_index": normalized["correct_index"],
            "hint": normalized["hint"],
            "explanation": normalized["explanation"],
            "topic": normalized["syllabus_topic"],
            "source": item.get("source") or "topic_ai_service",
        })
        normalized_payload.append({
            "text": normalized["text"],
            "options": normalized["options"],
            "correct_index": normalized["correct_index"],
            "hint": normalized["hint"],
            "explanation": normalized["explanation"],
            "topic": normalized["syllabus_topic"],
            "source": item.get("source") or "topic_ai_service",
        })

        if len(normalized_payload) >= count:
            break

    generation_status["generated_count"] = len(normalized_payload)
    preview_signature = _build_preview_signature(
        subject=subject,
        grade=grade,
        difficulty=difficulty,
        topic=topic,
        count=count,
        seed=seed,
        questions=normalized_preview,
    )

    if not persist:
        warning = None
        if generation_status["generated_count"] == 0:
            warning = generation_status["service_error"] or "No valid questions were generated for this request."
        elif generation_status.get("technical_fallback_used"):
            warning = (
                "Topic AI service was temporarily unavailable. "
                "Sample questions were generated for continuity."
            )

        return jsonify({
            "persisted": False,
            "requested_count": count,
            "generated_count": len(normalized_preview),
            "seed": seed,
            "preview_signature": preview_signature,
            "warning": warning,
            "generation_status": generation_status,
            "questions": normalized_preview,
        }), 200

    warning = None
    if len(normalized_payload) == 0:
        status_code = 503 if generation_status.get("service_status_code") in {503, 504} else 502
        return jsonify({
            "error": "Question generation is temporarily unavailable. Please retry shortly.",
            "generated_count": len(normalized_payload),
            "requested_count": count,
            "service_url": get_topic_ai_service_url(),
            "generation_status": generation_status,
        }), status_code

    if len(normalized_payload) < count:
        warning = (
            f"Requested {count} questions, but only {len(normalized_payload)} were generated and saved."
        )

    generated_records = _build_generated_question_records(
        normalized_payload,
        subject,
        grade,
        difficulty,
        topic,
        teacher.id,
        seed,
        default_source="topic_ai_service",
        generation_meta_extras={
            "service_meta": service_meta,
            "service_url": service_result.get("service_url"),
            "test_title": test_title,
            "test_description": test_description,
            "technical_fallback_used": bool(generation_status.get("technical_fallback_used")),
        },
    )

    for record in generated_records:
        db.session.add(record)

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to save generated questions: {str(exc)}"}), 500

    return jsonify({
        "persisted": True,
        "requested_count": count,
        "generated_count": len(generated_records),
        "seed": seed,
        "warning": warning,
        "generation_status": generation_status,
        "questions": [
            {
                "id": q.id,
                "text": q.text,
                "topic": q.syllabus_topic,
                "difficulty": q.difficulty,
                "subject": q.subject,
                "grade": q.grade,
            }
            for q in generated_records
        ],
    }), 201


@teacher_bp.post("/tests")
@require_auth
@role_required("teacher")
def create_test():
    """Create a new test and attach selected/generated questions."""
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    required_fields = ["title", "subject", "grade", "difficulty", "question_count"]
    error = validate_required_fields(data, required_fields)
    if error:
        return jsonify({"error": error}), 400

    title = sanitize_string(data.get("title"), max_length=255)
    subject = sanitize_string(data.get("subject"), max_length=64)
    grade = sanitize_string(data.get("grade"), max_length=32).lower()
    topic = sanitize_string(data.get("topic") or "", max_length=128) or None
    difficulty = sanitize_string(data.get("difficulty"), max_length=16).lower()
    question_count = _clamp_int(data.get("question_count", 10), 10, 1, 50)
    time_limit = _clamp_int(data.get("time_limit", 30), 30, 5, 240)
    description = sanitize_string(data.get("description") or "", max_length=1000) or None
    generation_mode = sanitize_string(data.get("generation_mode") or "standard", max_length=16).lower() or "standard"
    if generation_mode not in {"standard", "rag"}:
        return jsonify({"error": "generation_mode must be either 'standard' or 'rag'"}), 400
    if generation_mode == "standard" and not topic:
        return jsonify({"error": "Sub topic is required for topic-based generation."}), 400

    selected_document_ids = _parse_int_list(
        data.get("selected_document_ids") if isinstance(data.get("selected_document_ids"), list) else []
    )
    max_selected_docs = get_rag_max_selected_docs()
    if len(selected_document_ids) > max_selected_docs:
        return jsonify({
            "error": f"You can select up to {max_selected_docs} documents per request.",
        }), 400

    rag_top_k = _clamp_int(data.get("rag_top_k", 4), 4, 1, 12)
    rag_min_confidence_default = 0.0 if generation_mode == "rag" else get_rag_min_confidence()
    rag_min_confidence = _clamp_float(
        data.get("rag_min_confidence", rag_min_confidence_default),
        rag_min_confidence_default,
        0.0,
        1.0,
    )
    rag_vector_store = sanitize_string(data.get("rag_vector_store") or "", max_length=32) or None

    seed_raw = data.get("seed")
    seed = None
    if seed_raw is not None and str(seed_raw).strip() != "":
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "seed must be an integer"}), 400

    if difficulty not in VALID_DIFFICULTIES:
        return jsonify({"error": "Difficulty must be one of: easy, medium, hard"}), 400
    if grade not in VALID_GRADES:
        return jsonify({"error": "Grade must be one of: elementary, middle, high, college"}), 400

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    if generation_mode == "rag":
        processed_docs_query = TeacherDocument.query.filter(
            TeacherDocument.teacher_id == teacher.id,
            TeacherDocument.status == "processed",
        )

        if selected_document_ids:
            available_count = processed_docs_query.filter(
                TeacherDocument.id.in_(selected_document_ids),
            ).count()
            if available_count != len(selected_document_ids):
                return jsonify({"error": "One or more selected documents are unavailable for RAG."}), 400
        elif processed_docs_query.count() <= 0:
            return jsonify({"error": "Upload at least one processed document before using document based generation."}), 400

    try:
        test = Test(
            title=title,
            description=description,
            subject=subject,
            grade=grade,
            topic=topic,
            difficulty=difficulty,
            question_count=question_count,
            time_limit=time_limit,
            created_by=teacher.id,
            school_id=school_id,
            total_points=question_count,
        )
        db.session.add(test)
        db.session.flush()

        questions, generation_status = _pick_or_generate_questions(
            teacher_id=teacher.id,
            subject=subject,
            grade=grade,
            difficulty=difficulty,
            topic=topic,
            count=question_count,
            seed=seed,
            test_title=title,
            test_description=description,
            generation_mode=generation_mode,
            rag_options={
                "selected_document_ids": selected_document_ids,
                "top_k": rag_top_k,
                "min_confidence": rag_min_confidence,
                "vector_store": rag_vector_store,
            },
        )

        warning = None
        generated_count = len(questions)

        if generation_status.get("service_error"):
            if generation_status.get("technical_fallback_used"):
                warning = (
                    "Topic AI service was temporarily unavailable. "
                    "Sample questions were used for continuity."
                )
            else:
                warning = generation_status.get("service_error")

        if generated_count == 0:
            db.session.rollback()
            status_code = 503 if generation_status.get("service_status_code") in {503, 504} else 502
            return jsonify({
                "error": "Question generation is temporarily unavailable. Please retry shortly.",
                "generated_count": generated_count,
                "requested_count": question_count,
                "service_url": get_topic_ai_service_url(),
                "generation_status": generation_status,
            }), status_code

        if generated_count < question_count:
            partial_warning = (
                f"Requested {question_count} questions, but only {generated_count} could be generated. "
                "Test was created with available questions."
            )
            warning = f"{warning} {partial_warning}".strip() if warning else partial_warning
            test.question_count = generated_count
            test.total_points = generated_count

        for idx, question in enumerate(questions, start=1):
            db.session.add(TestQuestion(
                test_id=test.id,
                question_id=question.id,
                order=idx,
                points=1,
            ))

        _record_rag_retrieval_event(
            teacher_id=teacher.id,
            test_id=test.id,
            generation_status=generation_status,
            requested_count=question_count,
            generated_count=generated_count,
        )

        db.session.commit()

        current_app.logger.info(
            "Teacher test generation status test_id=%s requested=%s generated=%s service_status=%s latency_ms=%s endpoint=%s error=%s",
            test.id,
            question_count,
            generated_count,
            generation_status.get("service_status_code"),
            generation_status.get("service_latency_ms"),
            generation_status.get("service_endpoint"),
            generation_status.get("service_error"),
        )

        return jsonify({
            "message": "Test created successfully",
            "test": test.as_dict(),
            "question_ids": [q.id for q in questions],
            "requested_count": question_count,
            "generated_count": generated_count,
            "warning": warning,
            "generation_status": generation_status,
        }), 201
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create test: {str(exc)}"}), 500


@teacher_bp.get("/tests")
@require_auth
@role_required("teacher")
def get_tests():
    """Get all tests created by the teacher with performance metadata."""
    teacher = g.current_user

    tests = Test.query.filter_by(created_by=teacher.id).order_by(Test.created_at.desc()).all()

    test_ids = [t.id for t in tests]
    result_map = {}
    if test_ids:
        aggregates = db.session.query(
            TestResult.test_id,
            func.count(TestResult.id).label("attempts"),
            func.avg(
                func.coalesce((TestResult.correct_answers * 100.0) / func.nullif(TestResult.total_questions, 0), 0)
            ).label("avg_score"),
        ).filter(
            TestResult.test_id.in_(test_ids),
        ).group_by(TestResult.test_id).all()
        for row in aggregates:
            result_map[row.test_id] = {
                "attempts": int(row.attempts or 0),
                "average_score": round(float(row.avg_score or 0), 2),
            }

    payload = []
    for test in tests:
        item = test.as_dict()
        item.update(result_map.get(test.id, {"attempts": 0, "average_score": 0.0}))
        payload.append(item)

    return jsonify({"tests": payload}), 200


@teacher_bp.get("/tests/<int:test_id>")
@require_auth
@role_required("teacher")
def get_test(test_id):
    """Get a specific test with attached questions."""
    teacher = g.current_user

    test = Test.query.filter_by(id=test_id, created_by=teacher.id).first()
    if not test:
        return jsonify({"error": "Test not found"}), 404

    test_questions = TestQuestion.query.filter_by(test_id=test_id).order_by(TestQuestion.order.asc()).all()
    questions_data = []
    for tq in test_questions:
        question = db.session.get(Question, tq.question_id)
        if not question:
            continue
        generation_meta = question.generation_meta if isinstance(question.generation_meta, dict) else {}
        questions_data.append({
            "id": question.id,
            "text": question.text,
            "options": question.options,
            "correct_index": question.correct_index,
            "order": tq.order,
            "points": tq.points,
            "difficulty": question.difficulty,
            "topic": question.syllabus_topic,
            "hint": question.hint,
            "explanation": question.explanation,
            "source": generation_meta.get("source"),
            "provenance": generation_meta.get("provenance") if isinstance(generation_meta.get("provenance"), dict) else None,
            "retrieval_trace": generation_meta.get("retrieval_trace") if isinstance(generation_meta.get("retrieval_trace"), list) else [],
        })

    test_data = test.as_dict()
    test_data["questions"] = questions_data
    return jsonify(test_data), 200


@teacher_bp.put("/tests/<int:test_id>")
@require_auth
@role_required("teacher")
def update_test(test_id):
    """Update an existing teacher-owned test."""
    teacher = g.current_user

    test = Test.query.filter_by(id=test_id, created_by=teacher.id).first()
    if not test:
        return jsonify({"error": "Test not found"}), 404

    data = request.get_json(silent=True) or {}

    if "title" in data:
        test.title = sanitize_string(data.get("title"), max_length=255)
    if "description" in data:
        test.description = sanitize_string(data.get("description") or "", max_length=1000) or None
    if "time_limit" in data:
        test.time_limit = _clamp_int(data.get("time_limit"), test.time_limit, 5, 240)
    if "is_published" in data:
        test.is_published = bool(data.get("is_published"))
    if "is_active" in data:
        test.is_active = bool(data.get("is_active"))

    if "scheduled_at" in data:
        parsed = _parse_iso_datetime(data.get("scheduled_at"))
        if data.get("scheduled_at") and parsed is None:
            return jsonify({"error": "scheduled_at must be ISO datetime"}), 400
        test.scheduled_at = parsed

    if "expires_at" in data:
        parsed = _parse_iso_datetime(data.get("expires_at"))
        if data.get("expires_at") and parsed is None:
            return jsonify({"error": "expires_at must be ISO datetime"}), 400
        test.expires_at = parsed

    if "questions" in data:
        questions_payload = data.get("questions")
        if not isinstance(questions_payload, list) or len(questions_payload) == 0:
            return jsonify({"error": "questions must be a non-empty array"}), 400

        tqs = TestQuestion.query.filter_by(test_id=test.id).order_by(TestQuestion.order.asc()).all()
        tq_map = {tq.question_id: tq for tq in tqs}
        question_ids = [tq.question_id for tq in tqs]
        question_map = {q.id: q for q in Question.query.filter(Question.id.in_(question_ids)).all()} if question_ids else {}

        seen_orders = set()
        for idx, item in enumerate(questions_payload, start=1):
            if not isinstance(item, dict):
                return jsonify({"error": f"questions[{idx - 1}] must be an object"}), 400

            question_id = item.get("id")
            if not question_id or int(question_id) not in question_map:
                return jsonify({"error": f"questions[{idx - 1}] references unknown question id"}), 400
            question_id = int(question_id)

            text = sanitize_string(item.get("text") or "", max_length=2000)
            if not text:
                return jsonify({"error": f"questions[{idx - 1}].text is required"}), 400

            options = item.get("options")
            if not isinstance(options, list) or len(options) < 2:
                return jsonify({"error": f"questions[{idx - 1}].options must have at least 2 choices"}), 400
            normalized_options = [sanitize_string(opt or "", max_length=300) for opt in options]
            if any(not opt for opt in normalized_options):
                return jsonify({"error": f"questions[{idx - 1}].options cannot be empty"}), 400

            try:
                correct_index = int(item.get("correct_index"))
            except (TypeError, ValueError):
                return jsonify({"error": f"questions[{idx - 1}].correct_index must be an integer"}), 400

            if correct_index < 0 or correct_index >= len(normalized_options):
                return jsonify({"error": f"questions[{idx - 1}].correct_index out of range"}), 400

            order = item.get("order", idx)
            try:
                order = int(order)
            except (TypeError, ValueError):
                return jsonify({"error": f"questions[{idx - 1}].order must be an integer"}), 400

            if order < 1:
                return jsonify({"error": f"questions[{idx - 1}].order must be >= 1"}), 400
            if order in seen_orders:
                return jsonify({"error": "Question order values must be unique"}), 400
            seen_orders.add(order)

            try:
                points = int(item.get("points", 1))
            except (TypeError, ValueError):
                return jsonify({"error": f"questions[{idx - 1}].points must be an integer"}), 400
            points = max(1, points)
            
            explanation = sanitize_string(item.get("explanation") or "", max_length=1500)
            
            question = question_map[question_id]
            question.text = text
            question.options = normalized_options
            question.correct_index = correct_index
            
            if explanation is not None:
                question.explanation = explanation

            tq = tq_map.get(question_id)
            if tq:
                tq.order = order
                tq.points = points

        test.question_count = len(questions_payload)
        test.total_points = sum(max(1, int(item.get("points", 1))) for item in questions_payload)

    try:
        db.session.commit()
        return jsonify({"message": "Test updated successfully", "test": test.as_dict()}), 200
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to update test: {str(exc)}"}), 500


@teacher_bp.delete("/tests/<int:test_id>")
@require_auth
@role_required("teacher")
def delete_test(test_id):
    """Delete teacher-owned test."""
    teacher = g.current_user

    test = Test.query.filter_by(id=test_id, created_by=teacher.id).first()
    if not test:
        return jsonify({"error": "Test not found"}), 404

    try:
        db.session.delete(test)
        db.session.commit()
        return jsonify({"message": "Test deleted successfully"}), 200
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete test: {str(exc)}"}), 500


@teacher_bp.get("/students")
@require_auth
@role_required("teacher")
def get_students():
    """Get students in teacher scope with summary stats."""
    teacher = g.current_user
    grade = request.args.get("grade")

    students_query, effective_grade = _teacher_student_filter(teacher, grade=grade)
    students = students_query.order_by(User.name.asc()).all()
    student_ids = [s.id for s in students]

    stats_map = {sid: {"attempts": 0, "avg_score": 0.0} for sid in student_ids}
    if student_ids:
        rows = db.session.query(
            TestResult.user_id,
            func.count(TestResult.id).label("attempts"),
            func.avg(
                func.coalesce((TestResult.correct_answers * 100.0) / func.nullif(TestResult.total_questions, 0), 0)
            ).label("avg_score"),
        ).filter(
            TestResult.user_id.in_(student_ids),
        ).group_by(TestResult.user_id).all()

        for row in rows:
            stats_map[row.user_id] = {
                "attempts": int(row.attempts or 0),
                "avg_score": round(float(row.avg_score or 0), 2),
            }

    payload = []
    for student in students:
        item = student.as_dict()
        item.update(stats_map.get(student.id, {"attempts": 0, "avg_score": 0.0}))
        payload.append(item)

    return jsonify({
        "grade": effective_grade,
        "students": payload,
    }), 200


@teacher_bp.get("/classrooms")
@require_auth
@role_required("teacher")
def list_classrooms():
    teacher = g.current_user
    classrooms = Classroom.query.filter_by(teacher_id=teacher.id).order_by(Classroom.created_at.desc()).all()

    payload = []
    for classroom in classrooms:
        memberships = ClassroomStudent.query.filter_by(classroom_id=classroom.id, is_active=True).all()
        student_ids = [m.student_id for m in memberships]
        students = User.query.filter(User.id.in_(student_ids)).order_by(User.name.asc()).all() if student_ids else []

        item = classroom.as_dict()
        item["student_count"] = len(students)
        item["students"] = [student.as_dict() for student in students]
        payload.append(item)

    return jsonify({"classrooms": payload}), 200


@teacher_bp.post("/classrooms")
@require_auth
@role_required("teacher")
def create_classroom():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}
    error = validate_required_fields(data, ["name"])
    if error:
        return jsonify({"error": error}), 400

    name = sanitize_string(data.get("name"), max_length=128)
    grade = sanitize_string(data.get("grade") or teacher.grade or "", max_length=32) or None
    auto_enroll_students = bool(data.get("auto_enroll_students", False))

    if grade and grade not in VALID_GRADES:
        return jsonify({"error": "Invalid grade"}), 400

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    classroom = Classroom(
        name=name,
        grade=grade,
        school_id=school_id,
        teacher_id=teacher.id,
    )
    db.session.add(classroom)

    enrolled_count = 0
    try:
        db.session.flush()

        # Enroll only students already assigned to the teacher's school.
        if auto_enroll_students and grade:
            candidates = User.query.filter(
                User.role == "student",
                User.grade == grade,
                User.school_id == school_id,
            ).all()

            for student in candidates:
                membership = ClassroomStudent.query.filter_by(classroom_id=classroom.id, student_id=student.id).first()
                if membership:
                    if not membership.is_active:
                        membership.is_active = True
                    enrolled_count += 1
                    continue

                db.session.add(ClassroomStudent(classroom_id=classroom.id, student_id=student.id, is_active=True))
                enrolled_count += 1

        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create classroom: {str(exc)}"}), 500

    return jsonify({
        "message": "Classroom created",
        "classroom": classroom.as_dict(),
        "enrolled_count": enrolled_count,
        "auto_enroll_students": auto_enroll_students,
    }), 201


@teacher_bp.post("/classrooms/<int:classroom_id>/enroll-grade")
@require_auth
@role_required("teacher")
def enroll_classroom_by_grade(classroom_id):
    teacher = g.current_user
    classroom, error_response = _get_teacher_classroom_or_404(teacher.id, classroom_id)
    if error_response:
        return error_response

    if not classroom.grade:
        return jsonify({"error": "Classroom grade is required for bulk enroll"}), 400

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    enrolled_count = 0
    candidates = User.query.filter(
        User.role == "student",
        User.grade == classroom.grade,
        User.school_id == school_id,
    ).all()

    for student in candidates:
        membership = ClassroomStudent.query.filter_by(classroom_id=classroom.id, student_id=student.id).first()
        if membership:
            if not membership.is_active:
                membership.is_active = True
            enrolled_count += 1
            continue

        db.session.add(ClassroomStudent(classroom_id=classroom.id, student_id=student.id, is_active=True))
        enrolled_count += 1

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to enroll students: {str(exc)}"}), 500

    return jsonify({
        "message": "Students enrolled by grade",
        "classroom_id": classroom.id,
        "grade": classroom.grade,
        "enrolled_count": enrolled_count,
    }), 200


@teacher_bp.post("/classrooms/<int:classroom_id>/students")
@require_auth
@role_required("teacher")
def add_student_to_classroom(classroom_id):
    teacher = g.current_user
    classroom, error_response = _get_teacher_classroom_or_404(teacher.id, classroom_id)
    if error_response:
        return error_response

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id is required"}), 400

    student = User.query.filter_by(id=student_id, role="student", school_id=school_id).first()
    if not student:
        return jsonify({"error": "Student not found in your school"}), 404

    if classroom.grade and student.grade and classroom.grade != student.grade:
        return jsonify({"error": "Student grade does not match classroom grade"}), 400

    membership = ClassroomStudent.query.filter_by(classroom_id=classroom_id, student_id=student.id).first()
    if membership:
        membership.is_active = True
    else:
        membership = ClassroomStudent(classroom_id=classroom_id, student_id=student.id, is_active=True)
        db.session.add(membership)

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to add student: {str(exc)}"}), 500

    return jsonify({"message": "Student added", "membership": membership.as_dict()}), 200


@teacher_bp.delete("/classrooms/<int:classroom_id>/students/<int:student_id>")
@require_auth
@role_required("teacher")
def remove_student_from_classroom(classroom_id, student_id):
    teacher = g.current_user
    classroom, error_response = _get_teacher_classroom_or_404(teacher.id, classroom_id)
    if error_response:
        return error_response

    _ = classroom
    membership = ClassroomStudent.query.filter_by(classroom_id=classroom_id, student_id=student_id).first()
    if not membership:
        return jsonify({"error": "Membership not found"}), 404

    membership.is_active = False
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to remove student: {str(exc)}"}), 500

    return jsonify({"message": "Student removed"}), 200


@teacher_bp.get("/assignments")
@require_auth
@role_required("teacher")
def list_assignments():
    teacher = g.current_user
    limit = _clamp_int(request.args.get("limit", 200), 200, 1, 1000)

    rows = TestAssignment.query.filter_by(assigned_by=teacher.id).order_by(TestAssignment.created_at.desc()).limit(limit).all()

    test_ids = {row.test_id for row in rows}
    classroom_ids = {row.classroom_id for row in rows if row.classroom_id}
    student_ids = {row.student_id for row in rows if row.student_id}

    tests = {item.id: item for item in Test.query.filter(Test.id.in_(test_ids)).all()} if test_ids else {}
    classrooms = {item.id: item for item in Classroom.query.filter(Classroom.id.in_(classroom_ids)).all()} if classroom_ids else {}
    students = {item.id: item for item in User.query.filter(User.id.in_(student_ids)).all()} if student_ids else {}

    payload = []
    for row in rows:
        item = row.as_dict()
        item["test"] = tests.get(row.test_id).as_dict() if tests.get(row.test_id) else None
        item["classroom"] = classrooms.get(row.classroom_id).as_dict() if row.classroom_id and classrooms.get(row.classroom_id) else None
        item["student"] = students.get(row.student_id).as_dict() if row.student_id and students.get(row.student_id) else None
        payload.append(item)

    return jsonify({"assignments": payload}), 200


@teacher_bp.post("/assignments")
@require_auth
@role_required("teacher")
def create_assignment():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    error = validate_required_fields(data, ["test_id"])
    if error:
        return jsonify({"error": error}), 400

    test_id = data.get("test_id")
    classroom_id = data.get("classroom_id")
    student_id = data.get("student_id")
    notes = sanitize_string(data.get("notes") or "", max_length=1000) or None
    due_at = _parse_iso_datetime(data.get("due_at"))
    is_mandatory = bool(data.get("is_mandatory", True))
    allow_late = bool(data.get("allow_late", False))
    requested_require_camera = data.get("require_camera")
    requested_require_emotion = data.get("require_emotion")

    if requested_require_camera is False or requested_require_emotion is False:
        return jsonify({"error": "Assignments must enforce camera and emotion tracking"}), 400

    require_camera = True
    require_emotion = True

    if not classroom_id and not student_id:
        return jsonify({"error": "classroom_id or student_id is required"}), 400

    test = Test.query.filter_by(id=test_id, created_by=teacher.id).first()
    if not test:
        return jsonify({"error": "Test not found"}), 404

    school_id = _ensure_teacher_school(teacher)
    if not school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    if not test.is_published:
        return jsonify({"error": "Publish the test before assigning"}), 400

    created = []

    if classroom_id:
        classroom, error_response = _get_teacher_classroom_or_404(teacher.id, classroom_id)
        if error_response:
            return error_response

        assignment = TestAssignment(
            test_id=test.id,
            classroom_id=classroom.id,
            assigned_by=teacher.id,
            notes=notes,
            due_at=due_at,
            is_mandatory=is_mandatory,
            allow_late=allow_late,
            require_camera=require_camera,
            require_emotion=require_emotion,
            status='assigned',
        )
        db.session.add(assignment)
        created.append(assignment)

    if student_id:
        student = User.query.filter_by(id=student_id, role='student', school_id=school_id).first()
        if not student:
            return jsonify({"error": "Student not found in your school"}), 404

        assignment = TestAssignment(
            test_id=test.id,
            student_id=student.id,
            assigned_by=teacher.id,
            notes=notes,
            due_at=due_at,
            is_mandatory=is_mandatory,
            allow_late=allow_late,
            require_camera=require_camera,
            require_emotion=require_emotion,
            status='assigned',
        )
        db.session.add(assignment)
        created.append(assignment)

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create assignment: {str(exc)}"}), 500

    return jsonify({
        "message": "Assignment created",
        "assignments": [row.as_dict() for row in created],
    }), 201


@teacher_bp.patch('/assignments/<int:assignment_id>')
@require_auth
@role_required('teacher')
def update_assignment(assignment_id):
    teacher = g.current_user
    assignment = TestAssignment.query.filter_by(id=assignment_id, assigned_by=teacher.id).first()
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404

    data = request.get_json(silent=True) or {}
    if 'status' in data:
        status = sanitize_string(data.get('status'), max_length=32).lower()
        if status not in {'assigned', 'started', 'submitted', 'reviewed', 'expired', 'cancelled'}:
            return jsonify({"error": "Invalid status"}), 400
        assignment.status = status
        if status == 'reviewed':
            assignment.reviewed_at = _utcnow()

    if 'due_at' in data:
        parsed_due = _parse_iso_datetime(data.get('due_at'))
        if data.get('due_at') and parsed_due is None:
            return jsonify({"error": "due_at must be ISO datetime"}), 400
        assignment.due_at = parsed_due

    if 'notes' in data:
        assignment.notes = sanitize_string(data.get('notes') or '', max_length=1000) or None

    if 'require_camera' in data or 'require_emotion' in data:
        require_camera = bool(data.get('require_camera', True))
        require_emotion = bool(data.get('require_emotion', True))
        if not require_camera or not require_emotion:
            return jsonify({"error": "Assignments must enforce camera and emotion tracking"}), 400
        assignment.require_camera = True
        assignment.require_emotion = True

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to update assignment: {str(exc)}"}), 500

    return jsonify({"message": "Assignment updated", "assignment": assignment.as_dict()}), 200


@teacher_bp.delete("/assignments/<int:assignment_id>")
@require_auth
@role_required("teacher")
def delete_assignment(assignment_id: int):
    teacher = g.current_user
    assignment = TestAssignment.query.filter_by(id=int(assignment_id), assigned_by=teacher.id).first()
    if not assignment:
        return jsonify({"error": "Assignment not found"}), 404

    try:
        db.session.delete(assignment)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete assignment: {str(exc)}"}), 500

    return jsonify({"message": "Assignment deleted"}), 200


@teacher_bp.get("/reports")
@require_auth
@role_required("teacher")
def teacher_reports():
    """Detailed test reports scoped to teacher's school + grade."""
    teacher = g.current_user

    subject = sanitize_string(request.args.get("subject") or "", max_length=64) or None
    grade = request.args.get("grade", teacher.grade)
    limit = _clamp_int(request.args.get("limit", 100), 100, 1, 500)
    days = _parse_days_arg(90)
    cutoff = _utcnow() - timedelta(days=days)

    students_query, effective_grade = _teacher_student_filter(teacher, grade=grade)
    student_ids = [s.id for s in students_query.all()]
    if not student_ids:
        return jsonify({"items": [], "grade": effective_grade, "period_days": days}), 200

    q = TestResult.query.filter(
        TestResult.user_id.in_(student_ids),
        TestResult.started_at >= cutoff,
    )

    if subject:
        q = q.filter(TestResult.subject == subject)

    results = q.order_by(TestResult.started_at.desc()).limit(limit).all()

    user_map = {
        user.id: user
        for user in User.query.filter(User.id.in_({r.user_id for r in results})).all()
    }

    items = []
    for tr in results:
        user = user_map.get(tr.user_id)
        score_pct = round((tr.correct_answers / tr.total_questions * 100) if tr.total_questions > 0 else 0, 2)
        items.append({
            "id": tr.id,
            "student_name": user.name if user else "Unknown",
            "student_email": user.email if user else "Unknown",
            "subject": tr.subject,
            "score_pct": score_pct,
            "correct_answers": tr.correct_answers,
            "total_questions": tr.total_questions,
            "earned_points": tr.earned_points,
            "total_points": tr.total_points,
            "status": tr.status,
            "test_date": tr.started_at.isoformat() if tr.started_at else None,
            "finished_at": tr.finished_at.isoformat() if tr.finished_at else None,
        })

    return jsonify({
        "period_days": days,
        "grade": effective_grade,
        "items": items,
    }), 200


@teacher_bp.get("/rag/observability")
@require_auth
@role_required("teacher")
def teacher_rag_observability():
    teacher = g.current_user
    days = _parse_days_arg(30)
    limit = _clamp_int(request.args.get("limit", 500), 500, 20, 3000)
    cutoff = _utcnow() - timedelta(days=days)

    events = RagRetrievalEvent.query.filter(
        RagRetrievalEvent.teacher_id == teacher.id,
        RagRetrievalEvent.created_at >= cutoff,
    ).order_by(
        RagRetrievalEvent.created_at.desc(),
    ).limit(limit).all()

    total = len(events)
    rag_events = [event for event in events if str(event.generation_mode_requested or "").lower() == "rag"]
    success = sum(1 for event in events if event.status == "success")
    fallback = sum(1 for event in events if event.status == "fallback")
    errors = sum(1 for event in events if event.status == "error")

    rag_confidences = [float(event.confidence or 0.0) for event in rag_events]
    rag_relevance = [float(event.relevance or 0.0) for event in rag_events]
    rag_coverage = [float(event.coverage or 0.0) for event in rag_events]
    rag_duplication = [float(event.duplication or 0.0) for event in rag_events]
    rag_latency = [float(event.service_latency_ms or 0.0) for event in rag_events if event.service_latency_ms is not None]

    fallback_reasons = {}
    for event in events:
        reason = str(event.fallback_reason or "").strip()
        if not reason:
            continue
        fallback_reasons[reason] = int(fallback_reasons.get(reason, 0)) + 1

    daily = {}
    for event in events:
        if not event.created_at:
            continue
        key = event.created_at.date().isoformat()
        if key not in daily:
            daily[key] = {
                "date": key,
                "total": 0,
                "success": 0,
                "fallback": 0,
                "error": 0,
            }
        daily[key]["total"] += 1
        daily[key][str(event.status or "success")] = daily[key].get(str(event.status or "success"), 0) + 1

    status_rows = db.session.query(
        TeacherDocument.status,
        func.count(TeacherDocument.id),
    ).filter(
        TeacherDocument.teacher_id == teacher.id,
    ).group_by(
        TeacherDocument.status,
    ).all()
    status_counts = {str(row[0] or "unknown"): int(row[1] or 0) for row in status_rows}

    doc_storage_bytes = db.session.query(
        func.coalesce(func.sum(TeacherDocument.file_size_bytes), 0),
    ).filter(
        TeacherDocument.teacher_id == teacher.id,
    ).scalar() or 0

    avg_confidence = (sum(rag_confidences) / len(rag_confidences)) if rag_confidences else 0.0
    avg_relevance = (sum(rag_relevance) / len(rag_relevance)) if rag_relevance else 0.0
    avg_coverage = (sum(rag_coverage) / len(rag_coverage)) if rag_coverage else 0.0
    avg_duplication = (sum(rag_duplication) / len(rag_duplication)) if rag_duplication else 0.0
    avg_latency = (sum(rag_latency) / len(rag_latency)) if rag_latency else 0.0

    return jsonify({
        "period_days": days,
        "summary": {
            "total_events": total,
            "rag_events": len(rag_events),
            "success_events": success,
            "fallback_events": fallback,
            "error_events": errors,
            "fallback_rate": round((fallback / total), 4) if total > 0 else 0.0,
            "error_rate": round((errors / total), 4) if total > 0 else 0.0,
            "avg_confidence": round(avg_confidence, 4),
            "avg_relevance": round(avg_relevance, 4),
            "avg_coverage": round(avg_coverage, 4),
            "avg_duplication": round(avg_duplication, 4),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "fallback_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(fallback_reasons.items(), key=lambda item: item[1], reverse=True)
        ],
        "daily": [daily[key] for key in sorted(daily.keys())],
        "documents": {
            "total": int(sum(status_counts.values())),
            "storage_bytes": int(doc_storage_bytes),
            "storage_mb": round(float(doc_storage_bytes) / (1024 * 1024), 2),
            "status_counts": status_counts,
            "retention_days": get_rag_retention_days(),
            "max_docs": get_rag_max_documents_per_teacher(),
            "max_storage_mb": round(get_rag_max_storage_bytes_per_teacher() / (1024 * 1024), 2),
        },
        "recent_events": [event.as_dict() for event in events[:20]],
    }), 200


@teacher_bp.get("/analytics")
@require_auth
@role_required("teacher")
def teacher_analytics():
    """Get chart-ready analytics for teacher's students."""
    teacher = g.current_user
    grade = request.args.get("grade")
    days = _parse_days_arg(30)
    cutoff = _utcnow() - timedelta(days=days)

    students_query, effective_grade = _teacher_student_filter(teacher, grade=grade)
    student_ids = [s.id for s in students_query.all()]

    if not student_ids:
        return jsonify({
            "period_days": days,
            "grade": effective_grade,
            "summary": {
                "total_students": 0,
                "active_students": 0,
                "total_attempts": 0,
                "average_score": 0.0,
                "top_student": None,
            },
            "student_performance": [],
            "subject_performance": [],
            "grade_performance": [],
            "difficulty_performance": [],
            "weak_topics": [],
        }), 200

    score_expr = func.coalesce(
        (TestResult.correct_answers * 100.0) / func.nullif(TestResult.total_questions, 0),
        0,
    )

    student_performance_rows = db.session.query(
        User.id.label("student_id"),
        User.name.label("student_name"),
        User.grade.label("grade"),
        func.count(TestResult.id).label("total_attempts"),
        func.avg(score_expr).label("avg_score"),
        func.sum(func.coalesce(TestResult.correct_answers, 0)).label("correct_answers"),
        func.sum(func.coalesce(TestResult.total_questions, 0)).label("total_questions"),
    ).outerjoin(
        TestResult,
        and_(
            TestResult.user_id == User.id,
            TestResult.started_at >= cutoff,
        ),
    ).filter(
        User.id.in_(student_ids),
    ).group_by(
        User.id,
        User.name,
        User.grade,
    ).order_by(
        func.avg(score_expr).desc(),
        func.count(TestResult.id).desc(),
    ).all()

    subject_performance = db.session.query(
        TestResult.subject,
        func.count(TestResult.id).label("total_attempts"),
        func.avg(score_expr).label("avg_score"),
    ).filter(
        TestResult.user_id.in_(student_ids),
        TestResult.started_at >= cutoff,
    ).group_by(TestResult.subject).all()

    student_subject_rows = db.session.query(
        TestResult.user_id,
        TestResult.subject,
        func.count(TestResult.id).label("total_attempts"),
        func.avg(score_expr).label("avg_score"),
    ).filter(
        TestResult.user_id.in_(student_ids),
        TestResult.started_at >= cutoff,
    ).group_by(
        TestResult.user_id,
        TestResult.subject,
    ).all()

    user_name_by_id = {
        row.student_id: row.student_name
        for row in student_performance_rows
    }

    best_subject_by_student = {}
    top_student_by_subject = {}
    for row in student_subject_rows:
        student_id = int(row.user_id)
        subject = row.subject or "Unknown"
        avg_score = float(row.avg_score or 0)
        total_attempts = int(row.total_attempts or 0)

        candidate_subject = {
            "subject": subject,
            "avg_score": round(avg_score, 2),
            "total_attempts": total_attempts,
        }
        current_subject = best_subject_by_student.get(student_id)
        if not current_subject or candidate_subject["avg_score"] > current_subject["avg_score"]:
            best_subject_by_student[student_id] = candidate_subject

        candidate_student = {
            "student_id": student_id,
            "student_name": user_name_by_id.get(student_id, "Unknown"),
            "avg_score": round(avg_score, 2),
            "total_attempts": total_attempts,
        }
        current_student = top_student_by_subject.get(subject)
        if not current_student or candidate_student["avg_score"] > current_student["avg_score"]:
            top_student_by_subject[subject] = candidate_student

    difficulty_performance = db.session.query(
        Question.difficulty,
        func.count(AnswerLog.id).label("total_attempts"),
        func.avg(func.cast(AnswerLog.is_correct, db.Integer) * 100.0).label("accuracy"),
    ).join(
        Question,
        AnswerLog.question_id == Question.id,
    ).filter(
        AnswerLog.user_id.in_(student_ids),
        AnswerLog.answered_at >= cutoff,
    ).group_by(Question.difficulty).all()

    topic_performance = db.session.query(
        Question.syllabus_topic,
        func.count(AnswerLog.id).label("total_attempts"),
        func.avg(func.cast(AnswerLog.is_correct, db.Integer) * 100.0).label("accuracy"),
    ).join(
        Question,
        AnswerLog.question_id == Question.id,
    ).filter(
        AnswerLog.user_id.in_(student_ids),
        AnswerLog.answered_at >= cutoff,
        Question.syllabus_topic.isnot(None),
    ).group_by(Question.syllabus_topic).order_by(func.avg(func.cast(AnswerLog.is_correct, db.Integer) * 100.0).asc()).limit(10).all()

    grade_rollup = {}
    for row in student_performance_rows:
        grade_key = row.grade or "unknown"
        bucket = grade_rollup.setdefault(grade_key, {
            "grade": grade_key,
            "total_students": 0,
            "active_students": 0,
            "total_attempts": 0,
            "weighted_score_sum": 0.0,
        })
        attempts = int(row.total_attempts or 0)
        avg_score = float(row.avg_score or 0)

        bucket["total_students"] += 1
        bucket["total_attempts"] += attempts
        bucket["weighted_score_sum"] += avg_score * attempts
        if attempts > 0:
            bucket["active_students"] += 1

    grade_order = {"elementary": 0, "middle": 1, "high": 2, "college": 3, "unknown": 9}
    grade_performance = []
    for bucket in grade_rollup.values():
        attempts = bucket["total_attempts"]
        avg_score = (bucket["weighted_score_sum"] / attempts) if attempts > 0 else 0.0
        grade_performance.append({
            "grade": bucket["grade"],
            "total_students": int(bucket["total_students"]),
            "active_students": int(bucket["active_students"]),
            "total_attempts": int(attempts),
            "avg_score": round(float(avg_score), 2),
        })

    grade_performance.sort(key=lambda item: grade_order.get(item["grade"], 99))

    total_attempts = sum(int(row.total_attempts or 0) for row in student_performance_rows)
    active_students = sum(1 for row in student_performance_rows if int(row.total_attempts or 0) > 0)
    average_score = round(
        sum(float(row.avg_score or 0) for row in student_performance_rows if int(row.total_attempts or 0) > 0) / active_students,
        2,
    ) if active_students > 0 else 0.0
    top_student = None
    ranked_active_students = [row for row in student_performance_rows if int(row.total_attempts or 0) > 0]
    if ranked_active_students:
        top_row = ranked_active_students[0]
        top_student = {
            "student_id": int(top_row.student_id),
            "student_name": top_row.student_name,
            "avg_score": round(float(top_row.avg_score or 0), 2),
            "total_attempts": int(top_row.total_attempts or 0),
        }

    at_risk_payload = {}
    try:
        at_risk_payload = get_at_risk_predictions_for_students(
            [int(sid) for sid in student_ids],
            cutoff=cutoff,
            top_k_shap=0,
        )
    except Exception:
        at_risk_payload = {"at_risk_students": [], "meta": {"reason": "at_risk_failed"}}

    return jsonify({
        "period_days": days,
        "grade": effective_grade,
        "summary": {
            "total_students": len(student_ids),
            "active_students": active_students,
            "total_attempts": total_attempts,
            "average_score": average_score,
            "top_student": top_student,
        },
        "student_performance": [
            {
                "student_id": int(row.student_id),
                "student_name": row.student_name,
                "grade": row.grade,
                "total_attempts": int(row.total_attempts or 0),
                "avg_score": round(float(row.avg_score or 0), 2),
                "correct_answers": int(row.correct_answers or 0),
                "total_questions": int(row.total_questions or 0),
                "best_subject": best_subject_by_student.get(int(row.student_id), {}).get("subject"),
            }
            for row in student_performance_rows
        ],
        "subject_performance": [
            {
                "subject": row.subject,
                "total_attempts": int(row.total_attempts or 0),
                "avg_score": round(float(row.avg_score or 0), 2),
                "top_student": top_student_by_subject.get(row.subject or "Unknown"),
            }
            for row in subject_performance
        ],
        "grade_performance": grade_performance,
        "difficulty_performance": [
            {
                "difficulty": row.difficulty or "unknown",
                "total_attempts": int(row.total_attempts or 0),
                "accuracy": round(float(row.accuracy or 0), 2),
            }
            for row in difficulty_performance
        ],
        "weak_topics": [
            {
                "topic": row.syllabus_topic,
                "total_attempts": int(row.total_attempts or 0),
                "accuracy": round(float(row.accuracy or 0), 2),
            }
            for row in topic_performance
        ],
        "at_risk_students": [
            {
                **item,
                "student_name": user_name_by_id.get(int(item.get("student_id")), None),
            }
            for item in (at_risk_payload.get("at_risk_students") or [])
        ],
        "at_risk_meta": at_risk_payload.get("meta") or {},
    }), 200


@teacher_bp.get("/interventions")
@require_auth
@role_required("teacher")
def list_interventions():
    teacher = g.current_user
    status = sanitize_string(request.args.get("status") or "", max_length=32).lower() or None
    limit = _clamp_int(request.args.get("limit", 100), 100, 1, 500)

    query = TeacherIntervention.query.filter_by(teacher_id=teacher.id)
    if status:
        if status not in INTERVENTION_STATUSES:
            return jsonify({"error": "Invalid intervention status"}), 400
        query = query.filter(TeacherIntervention.status == status)

    rows = query.order_by(TeacherIntervention.updated_at.desc(), TeacherIntervention.created_at.desc()).limit(limit).all()
    return jsonify({
        "count": len(rows),
        "items": [row.as_dict() for row in rows],
    }), 200


@teacher_bp.post("/interventions")
@require_auth
@role_required("teacher")
def create_intervention():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    teacher_school_id = _ensure_teacher_school(teacher)
    if not teacher_school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    title = sanitize_string(data.get("title") or "", max_length=255)
    if not title:
        return jsonify({"error": "title is required"}), 400

    status = sanitize_string(data.get("status") or "planned", max_length=32).lower() or "planned"
    if status not in INTERVENTION_STATUSES:
        return jsonify({"error": "Invalid intervention status"}), 400

    due_at = _parse_iso_datetime(data.get("due_at"))
    if data.get("due_at") and due_at is None:
        return jsonify({"error": "due_at must be ISO datetime"}), 400

    classroom_id = data.get("classroom_id")
    if classroom_id:
        classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=teacher.id).first()
        if not classroom:
            return jsonify({"error": "Classroom not found"}), 404

    related_test_id = data.get("related_test_id")
    if related_test_id:
        test = Test.query.filter_by(id=related_test_id, created_by=teacher.id).first()
        if not test:
            return jsonify({"error": "Related test not found"}), 404

    allowed_student_ids = {
        int(row.id)
        for row in User.query.filter_by(role="student", school_id=teacher_school_id).all()
    }
    student_ids = []
    if isinstance(data.get("student_ids"), list):
        for raw_id in data.get("student_ids"):
            try:
                sid = int(raw_id)
            except (TypeError, ValueError):
                continue
            if sid in allowed_student_ids:
                student_ids.append(sid)
    student_ids = sorted(set(student_ids))

    assignment_ids = []
    if isinstance(data.get("assignment_ids"), list):
        allowed_assignment_ids = {
            int(row.id)
            for row in TestAssignment.query.filter(
                TestAssignment.assigned_by == teacher.id,
                TestAssignment.id.in_([int(x) for x in data.get("assignment_ids") if str(x).isdigit()]),
            ).all()
        }
        assignment_ids = sorted(allowed_assignment_ids)

    try:
        intervention = _create_intervention_entry(
            teacher,
            action_type=sanitize_string(data.get("action_type") or "note", max_length=64) or "note",
            title=title,
            notes=data.get("notes"),
            status=status,
            subject=sanitize_string(data.get("subject") or "", max_length=64) or None,
            topic=sanitize_string(data.get("topic") or "", max_length=128) or None,
            due_at=due_at,
            classroom_id=classroom_id,
            related_test_id=related_test_id,
            student_ids=student_ids,
            assignment_ids=assignment_ids,
            cluster_payload=data.get("cluster_payload") if isinstance(data.get("cluster_payload"), list) else [],
            metadata_json=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create intervention: {str(exc)}"}), 500

    return jsonify({"message": "Intervention created", "intervention": intervention.as_dict()}), 201


@teacher_bp.patch("/interventions/<int:intervention_id>")
@require_auth
@role_required("teacher")
def update_intervention(intervention_id):
    teacher = g.current_user
    intervention = TeacherIntervention.query.filter_by(id=intervention_id, teacher_id=teacher.id).first()
    if not intervention:
        return jsonify({"error": "Intervention not found"}), 404

    teacher_school_id = _ensure_teacher_school(teacher)
    if not teacher_school_id:
        return jsonify({"error": UNASSIGNED_TEACHER_SCHOOL_ERROR}), 403

    data = request.get_json(silent=True) or {}

    if "title" in data:
        title = sanitize_string(data.get("title") or "", max_length=255)
        if not title:
            return jsonify({"error": "title cannot be empty"}), 400
        intervention.title = title

    if "notes" in data:
        intervention.notes = sanitize_string(data.get("notes") or "", max_length=4000) or None

    if "status" in data:
        status = sanitize_string(data.get("status") or "", max_length=32).lower()
        if status not in INTERVENTION_STATUSES:
            return jsonify({"error": "Invalid intervention status"}), 400
        intervention.status = status

    if "due_at" in data:
        parsed_due = _parse_iso_datetime(data.get("due_at"))
        if data.get("due_at") and parsed_due is None:
            return jsonify({"error": "due_at must be ISO datetime"}), 400
        intervention.due_at = parsed_due

    if "subject" in data:
        intervention.subject = sanitize_string(data.get("subject") or "", max_length=64) or None

    if "topic" in data:
        intervention.topic = sanitize_string(data.get("topic") or "", max_length=128) or None

    if "student_ids" in data and isinstance(data.get("student_ids"), list):
        allowed_student_ids = {
            int(row.id)
            for row in User.query.filter_by(role="student", school_id=teacher_school_id).all()
        }
        filtered_ids = []
        for raw_id in data.get("student_ids"):
            try:
                sid = int(raw_id)
            except (TypeError, ValueError):
                continue
            if sid in allowed_student_ids:
                filtered_ids.append(sid)
        intervention.student_ids = sorted(set(filtered_ids))

    if "metadata" in data and isinstance(data.get("metadata"), dict):
        intervention.metadata_json = data.get("metadata")

    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to update intervention: {str(exc)}"}), 500

    return jsonify({"message": "Intervention updated", "intervention": intervention.as_dict()}), 200


@teacher_bp.post("/interventions/actions/remedial-assignment")
@require_auth
@role_required("teacher")
def action_assign_remedial_test():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    days = _clamp_int(data.get("days", 30), 30, 7, 365)
    due_days = _clamp_int(data.get("due_days", 7), 7, 1, 120)
    max_students = _clamp_int(data.get("max_students", 12), 12, 1, 200)
    min_topic_attempts = _clamp_int(data.get("min_topic_attempts", 2), 2, 1, 30)
    question_count = _clamp_int(data.get("question_count", 8), 8, 3, 30)
    time_limit = _clamp_int(data.get("time_limit", 20), 20, 5, 180)
    accuracy_threshold = _clamp_float(data.get("accuracy_threshold", 65), 65, 1, 99)

    subject = sanitize_string(data.get("subject") or "", max_length=64) or None
    selected_topic = sanitize_string(data.get("topic") or "", max_length=128)
    selected_topic = selected_topic.lower() if selected_topic else None

    _, student_ids, effective_grade = _scope_teacher_students(teacher, grade=data.get("grade"))
    if not student_ids:
        return jsonify({"error": "No students available in teacher scope"}), 400

    cutoff = _utcnow() - timedelta(days=days)
    topic_rows = _topic_accuracy_rows(student_ids, cutoff, subject=subject)

    weakest_topic = selected_topic
    weakest_subject = subject
    if not weakest_topic:
        rollup = {}
        for row in topic_rows:
            if row["attempts"] < min_topic_attempts or not row["topic"]:
                continue
            key = (row["topic"], row["subject"])
            bucket = rollup.setdefault(key, {"attempts": 0, "weighted_accuracy": 0.0})
            bucket["attempts"] += row["attempts"]
            bucket["weighted_accuracy"] += row["accuracy"] * row["attempts"]

        ranked_topics = sorted(
            [
                {
                    "topic": key[0],
                    "subject": key[1],
                    "attempts": value["attempts"],
                    "accuracy": (value["weighted_accuracy"] / value["attempts"]) if value["attempts"] else 100.0,
                }
                for key, value in rollup.items()
            ],
            key=lambda item: (item["accuracy"], -item["attempts"]),
        )
        if ranked_topics:
            weakest_topic = ranked_topics[0]["topic"]
            weakest_subject = ranked_topics[0]["subject"]

    candidate_rows = [
        row for row in topic_rows
        if row.get("topic") == weakest_topic
        and (not weakest_subject or row.get("subject") == weakest_subject)
        and row.get("attempts", 0) >= min_topic_attempts
        and row.get("accuracy", 100.0) <= accuracy_threshold
    ]
    candidate_rows.sort(key=lambda row: (row.get("accuracy", 100.0), -row.get("attempts", 0)))
    target_student_ids = [int(row["student_id"]) for row in candidate_rows[:max_students]]

    if not target_student_ids:
        score_expr = _score_pct_expression()
        low_rows = db.session.query(
            TestResult.user_id,
            func.avg(score_expr).label("avg_score"),
        ).filter(
            TestResult.user_id.in_(student_ids),
            TestResult.started_at >= cutoff,
        ).group_by(
            TestResult.user_id,
        ).having(
            func.avg(score_expr) <= accuracy_threshold,
        ).order_by(
            func.avg(score_expr).asc(),
        ).limit(max_students).all()
        target_student_ids = [int(row.user_id) for row in low_rows]

    if not target_student_ids:
        target_student_ids = student_ids[:max_students]

    if not target_student_ids:
        return jsonify({"error": "No target students found for remedial assignment"}), 400

    selected_subject = _select_subject_for_students(target_student_ids, cutoff, preferred_subject=weakest_subject or subject)

    students_by_id = {
        int(s.id): s
        for s in User.query.filter(User.id.in_(target_student_ids)).all()
    }
    grade_votes = {}
    for sid in target_student_ids:
        grade_key = _normalize_grade_for_generation(getattr(students_by_id.get(sid), "grade", None), effective_grade or teacher.grade)
        grade_votes[grade_key] = grade_votes.get(grade_key, 0) + 1
    selected_grade = sorted(grade_votes.items(), key=lambda item: item[1], reverse=True)[0][0] if grade_votes else _normalize_grade_for_generation(effective_grade, teacher.grade)

    due_at = _utcnow() + timedelta(days=due_days)
    action_notes = sanitize_string(data.get("notes") or "", max_length=1000) or (
        f"Remedial support for weak topic: {weakest_topic or 'general revision'}"
    )

    try:
        test, generation_status, warning = _build_generated_intervention_test(
            teacher,
            title=f"Remedial: {to_title_case(weakest_topic) if weakest_topic else to_title_case(selected_subject)}",
            subject=selected_subject,
            grade=selected_grade,
            difficulty="easy",
            topic=weakest_topic,
            question_count=question_count,
            time_limit=time_limit,
            description="Auto-generated remedial intervention from teacher reports.",
            seed=data.get("seed"),
        )

        created_assignments = _build_assignments_for_students(
            teacher,
            test=test,
            student_ids=target_student_ids,
            due_at=due_at,
            notes=action_notes,
            is_mandatory=True,
            allow_late=False,
        )
        db.session.flush()

        intervention = _create_intervention_entry(
            teacher,
            action_type="assign_remedial_test",
            title=f"Assign remedial test for {weakest_topic or selected_subject}",
            notes=action_notes,
            status="planned",
            subject=selected_subject,
            topic=weakest_topic,
            due_at=due_at,
            related_test_id=test.id,
            student_ids=target_student_ids,
            assignment_ids=[int(row.id) for row in created_assignments],
            metadata_json={
                "days": days,
                "accuracy_threshold": accuracy_threshold,
                "generation_status": generation_status,
                "warning": warning,
            },
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create remedial action: {str(exc)}"}), 500

    return jsonify({
        "message": "Remedial action created",
        "action": "assign_remedial_test",
        "topic": weakest_topic,
        "subject": selected_subject,
        "target_student_count": len(target_student_ids),
        "created_assignments": len(created_assignments),
        "test": test.as_dict(),
        "intervention": intervention.as_dict(),
        "warning": warning,
    }), 201


@teacher_bp.post("/interventions/actions/focused-practice")
@require_auth
@role_required("teacher")
def action_create_focused_practice():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    days = _clamp_int(data.get("days", 30), 30, 7, 365)
    due_days = _clamp_int(data.get("due_days", 5), 5, 1, 120)
    max_students = _clamp_int(data.get("max_students", 15), 15, 1, 200)
    min_attempts = _clamp_int(data.get("min_attempts", 1), 1, 1, 50)
    question_count = _clamp_int(data.get("question_count", 10), 10, 3, 40)
    time_limit = _clamp_int(data.get("time_limit", 20), 20, 5, 180)
    low_accuracy_threshold = _clamp_float(data.get("low_accuracy_threshold", 60), 60, 1, 99)

    grade = data.get("grade")
    subject = sanitize_string(data.get("subject") or "", max_length=64) or None
    topic = sanitize_string(data.get("topic") or "", max_length=128) or None

    _, student_ids, effective_grade = _scope_teacher_students(teacher, grade=grade)
    if not student_ids:
        return jsonify({"error": "No students available in teacher scope"}), 400

    cutoff = _utcnow() - timedelta(days=days)
    score_expr = _score_pct_expression()

    perf_query = db.session.query(
        TestResult.user_id,
        func.avg(score_expr).label("avg_score"),
        func.count(TestResult.id).label("attempts"),
    ).filter(
        TestResult.user_id.in_(student_ids),
        TestResult.started_at >= cutoff,
    )
    if subject:
        perf_query = perf_query.filter(TestResult.subject == subject)

    low_rows = perf_query.group_by(
        TestResult.user_id,
    ).having(
        and_(
            func.count(TestResult.id) >= min_attempts,
            func.avg(score_expr) <= low_accuracy_threshold,
        )
    ).order_by(
        func.avg(score_expr).asc(),
        func.count(TestResult.id).desc(),
    ).limit(max_students).all()

    target_student_ids = [int(row.user_id) for row in low_rows]
    if not target_student_ids:
        return jsonify({"error": "No low-accuracy students found for focused practice"}), 400

    selected_subject = _select_subject_for_students(target_student_ids, cutoff, preferred_subject=subject)

    selected_topic = topic
    if not selected_topic:
        topic_rows = _topic_accuracy_rows(target_student_ids, cutoff, subject=selected_subject)
        if topic_rows:
            topic_rows.sort(key=lambda row: (row.get("accuracy", 100.0), -row.get("attempts", 0)))
            selected_topic = topic_rows[0].get("topic")

    due_at = _utcnow() + timedelta(days=due_days)
    notes = sanitize_string(data.get("notes") or "", max_length=1000) or "Focused practice set for low-accuracy students"

    try:
        test, generation_status, warning = _build_generated_intervention_test(
            teacher,
            title=f"Focused Practice: {to_title_case(selected_subject)}",
            subject=selected_subject,
            grade=_normalize_grade_for_generation(effective_grade, teacher.grade),
            difficulty="easy",
            topic=selected_topic,
            question_count=question_count,
            time_limit=time_limit,
            description="Auto-created focused practice set based on low accuracy insights.",
            seed=data.get("seed"),
        )

        created_assignments = _build_assignments_for_students(
            teacher,
            test=test,
            student_ids=target_student_ids,
            due_at=due_at,
            notes=notes,
            is_mandatory=False,
            allow_late=True,
        )
        db.session.flush()

        intervention = _create_intervention_entry(
            teacher,
            action_type="create_focused_practice_set",
            title=f"Focused practice for low accuracy ({selected_subject})",
            notes=notes,
            status="planned",
            subject=selected_subject,
            topic=selected_topic,
            due_at=due_at,
            related_test_id=test.id,
            student_ids=target_student_ids,
            assignment_ids=[int(row.id) for row in created_assignments],
            metadata_json={
                "days": days,
                "low_accuracy_threshold": low_accuracy_threshold,
                "generation_status": generation_status,
                "warning": warning,
            },
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create focused practice action: {str(exc)}"}), 500

    return jsonify({
        "message": "Focused practice action created",
        "action": "create_focused_practice_set",
        "subject": selected_subject,
        "topic": selected_topic,
        "target_student_count": len(target_student_ids),
        "created_assignments": len(created_assignments),
        "test": test.as_dict(),
        "intervention": intervention.as_dict(),
        "warning": warning,
    }), 201


@teacher_bp.post("/interventions/actions/follow-up-assignment")
@require_auth
@role_required("teacher")
def action_schedule_follow_up_assignment():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    days = _clamp_int(data.get("days", 30), 30, 7, 365)
    due_days = _clamp_int(data.get("due_days", 10), 10, 1, 180)
    max_students = _clamp_int(data.get("max_students", 12), 12, 1, 200)
    question_count = _clamp_int(data.get("question_count", 8), 8, 3, 30)
    time_limit = _clamp_int(data.get("time_limit", 25), 25, 5, 240)
    at_risk_threshold = _clamp_float(data.get("at_risk_threshold", 0.55), 0.55, 0.01, 0.99)

    _, student_ids, effective_grade = _scope_teacher_students(teacher, grade=data.get("grade"))
    if not student_ids:
        return jsonify({"error": "No students available in teacher scope"}), 400

    cutoff = _utcnow() - timedelta(days=days)
    selected_subject = _select_subject_for_students(
        student_ids,
        cutoff,
        preferred_subject=sanitize_string(data.get("subject") or "", max_length=64) or None,
    )
    selected_topic = sanitize_string(data.get("topic") or "", max_length=128) or None

    at_risk_payload = {}
    try:
        at_risk_payload = get_at_risk_predictions_for_students(student_ids, cutoff=cutoff, top_k_shap=0)
    except Exception:
        at_risk_payload = {"at_risk_students": [], "meta": {"reason": "at_risk_failed"}}

    def _probability(item):
        return _clamp_float(item.get("at_risk_probability", 0), 0, 0, 1)

    at_risk_rows = sorted(
        [item for item in (at_risk_payload.get("at_risk_students") or []) if item.get("student_id") is not None],
        key=_probability,
        reverse=True,
    )
    target_student_ids = [
        int(item.get("student_id"))
        for item in at_risk_rows
        if _probability(item) >= at_risk_threshold
    ][:max_students]

    if not target_student_ids:
        target_student_ids = [int(item.get("student_id")) for item in at_risk_rows[:max_students] if item.get("student_id") is not None]

    if not target_student_ids:
        return jsonify({"error": "No at-risk students found for follow-up assignment"}), 400

    due_at = _utcnow() + timedelta(days=due_days)
    notes = sanitize_string(data.get("notes") or "", max_length=1000) or "Scheduled follow-up assignment for at-risk students"

    try:
        test, generation_status, warning = _build_generated_intervention_test(
            teacher,
            title=f"Follow-up Check-in: {to_title_case(selected_subject)}",
            subject=selected_subject,
            grade=_normalize_grade_for_generation(effective_grade, teacher.grade),
            difficulty="medium",
            topic=selected_topic,
            question_count=question_count,
            time_limit=time_limit,
            description="Auto-scheduled follow-up assignment for at-risk learners.",
            seed=data.get("seed"),
        )

        created_assignments = _build_assignments_for_students(
            teacher,
            test=test,
            student_ids=target_student_ids,
            due_at=due_at,
            notes=notes,
            is_mandatory=True,
            allow_late=True,
        )
        db.session.flush()

        intervention = _create_intervention_entry(
            teacher,
            action_type="schedule_follow_up_assignment",
            title=f"Follow-up assignment for at-risk students ({selected_subject})",
            notes=notes,
            status="planned",
            subject=selected_subject,
            topic=selected_topic,
            due_at=due_at,
            related_test_id=test.id,
            student_ids=target_student_ids,
            assignment_ids=[int(row.id) for row in created_assignments],
            metadata_json={
                "days": days,
                "at_risk_threshold": at_risk_threshold,
                "at_risk_meta": at_risk_payload.get("meta") or {},
                "generation_status": generation_status,
                "warning": warning,
            },
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create follow-up action: {str(exc)}"}), 500

    return jsonify({
        "message": "Follow-up assignment action created",
        "action": "schedule_follow_up_assignment",
        "subject": selected_subject,
        "topic": selected_topic,
        "target_student_count": len(target_student_ids),
        "created_assignments": len(created_assignments),
        "test": test.as_dict(),
        "intervention": intervention.as_dict(),
        "warning": warning,
    }), 201


@teacher_bp.post("/interventions/actions/weakness-clusters")
@require_auth
@role_required("teacher")
def action_group_by_weakness_cluster():
    teacher = g.current_user
    data = request.get_json(silent=True) or {}

    days = _clamp_int(data.get("days", 30), 30, 7, 365)
    min_attempts = _clamp_int(data.get("min_attempts", 2), 2, 1, 30)
    subject = sanitize_string(data.get("subject") or "", max_length=64) or None

    students, student_ids, effective_grade = _scope_teacher_students(teacher, grade=data.get("grade"))
    if not student_ids:
        return jsonify({"error": "No students available in teacher scope"}), 400

    students_by_id = {int(s.id): s for s in students}
    cutoff = _utcnow() - timedelta(days=days)
    topic_rows = _topic_accuracy_rows(student_ids, cutoff, subject=subject)

    weakest_by_student = {}
    for row in topic_rows:
        if row.get("attempts", 0) < min_attempts or not row.get("topic"):
            continue
        sid = int(row.get("student_id"))
        current = weakest_by_student.get(sid)
        if not current or row.get("accuracy", 100.0) < current.get("accuracy", 100.0):
            weakest_by_student[sid] = row

    clusters_map = {}
    for sid, row in weakest_by_student.items():
        key = (row.get("topic"), row.get("subject"))
        bucket = clusters_map.setdefault(key, {
            "topic": row.get("topic"),
            "subject": row.get("subject"),
            "student_ids": [],
            "student_names": [],
            "accuracy_values": [],
        })
        bucket["student_ids"].append(sid)
        bucket["student_names"].append(getattr(students_by_id.get(sid), "name", f"Student {sid}"))
        bucket["accuracy_values"].append(float(row.get("accuracy", 0)))

    clusters = []
    for bucket in clusters_map.values():
        acc_values = bucket.pop("accuracy_values", [])
        avg_accuracy = round(sum(acc_values) / len(acc_values), 2) if acc_values else 0.0
        bucket["student_count"] = len(bucket["student_ids"])
        bucket["average_accuracy"] = avg_accuracy
        clusters.append(bucket)

    clusters.sort(key=lambda item: (-item.get("student_count", 0), item.get("average_accuracy", 100.0)))
    if not clusters:
        return jsonify({"error": "No weakness clusters found for selected filters"}), 400

    all_cluster_student_ids = sorted({sid for cluster in clusters for sid in cluster.get("student_ids", [])})
    notes = sanitize_string(data.get("notes") or "", max_length=1000) or "Grouped students by weakness cluster for classroom intervention planning"

    try:
        intervention = _create_intervention_entry(
            teacher,
            action_type="group_weakness_clusters",
            title="Student weakness clusters for classroom intervention",
            notes=notes,
            status="planned",
            subject=subject,
            topic=None,
            due_at=None,
            related_test_id=None,
            student_ids=all_cluster_student_ids,
            assignment_ids=[],
            cluster_payload=clusters,
            metadata_json={
                "days": days,
                "min_attempts": min_attempts,
                "grade": effective_grade,
            },
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": f"Failed to create weakness clusters: {str(exc)}"}), 500

    return jsonify({
        "message": "Weakness clusters created",
        "action": "group_weakness_clusters",
        "cluster_count": len(clusters),
        "clusters": clusters,
        "intervention": intervention.as_dict(),
    }), 201


def to_title_case(value):
    return " ".join(
        token.capitalize() for token in str(value or "").replace("_", " ").replace("-", " ").split() if token
    )
