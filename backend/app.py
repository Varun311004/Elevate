"""
backend/app.py  — Application factory
======================================
Changes from original:
  • Registers the new  /api/ai/emotion  blueprint (ai_emotion.py)
  • All other code is identical to the original
"""

from flask import Flask, send_from_directory, Response, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import inspect, text
import os
import re
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=False)

from .config import get_config
from .models import (
    db, User, School, Question, UserProgress, EmotionLog,
    SubjectPerformance, AnswerLog, Test, TestQuestion, TestResult,
    TeacherIntervention, TeacherDocument, TeacherDocumentChunk,
    SyllabusTopic, UserSetting, RagRetrievalEvent,
    AuditLog, ModelVersion, TrainingJob, MCQPipelineEvent,
)
from .routes.auth      import auth_bp
from .routes.questions import questions_bp
from .routes.progress  import progress_bp
from .routes.reports   import reports_bp
from .routes.emotions  import emotions_bp
from .routes.admin     import admin_bp
from .routes.teacher   import teacher_bp
from .routes.student   import student_bp
from .routes.settings  import settings_bp

# ── NEW: AI emotion inference blueprint ─────────────────────────────────────
from .routes.ai_emotion import ai_emotion_bp, inspect_emotion_artifacts

from .logging_config import configure_logging


def _is_truthy_env(raw_value: str | None) -> bool:
    value = str(raw_value or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _assert_gemini_key_startup_health(app: Flask, environment_name: str) -> None:
    """Fail fast when Gemini credentials are required but not configured."""
    if bool(app.config.get("TESTING")):
        return

    normalized_env = str(environment_name or "").strip().lower()
    is_production_runtime = (
        normalized_env == "production"
        or bool(str(os.environ.get("RENDER_SERVICE_ID") or "").strip())
        or _is_truthy_env(os.environ.get("RENDER"))
        or bool(str(os.environ.get("K_SERVICE") or "").strip())
    )

    explicit_requirement = os.environ.get("ELEVATE_REQUIRE_GEMINI_KEY")
    if explicit_requirement is None:
        require_key = is_production_runtime or (not bool(app.config.get("DEBUG")))
    else:
        require_key = _is_truthy_env(explicit_requirement)

    if not require_key:
        return

    gemini_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )
    if str(gemini_key).strip():
        return

    raise RuntimeError(
        "Gemini API key is required at startup but missing. "
        "Set GEMINI_API_KEY (or GOOGLE_API_KEY). "
        "To disable this assertion outside production, set ELEVATE_REQUIRE_GEMINI_KEY=0."
    )


def _log_emotion_deploy_status(app: Flask) -> None:
    """Emit startup diagnostics for emotion model artifacts."""
    try:
        details = inspect_emotion_artifacts()
        summary = details.get("training_summary") or {}

        app.logger.info(
            "[EmotionDeploy] backend_model_exists=%s metadata_exists=%s tfjs_model_exists=%s tfjs_weights_exists=%s",
            details.get("backend_model_exists"),
            details.get("metadata_exists"),
            details.get("tfjs_model_exists"),
            details.get("tfjs_weights_exists"),
        )
        app.logger.info(
            "[EmotionDeploy] backend_model_path=%s metadata_path=%s",
            details.get("backend_model_path"),
            details.get("metadata_path"),
        )

        if summary:
            app.logger.info(
                "[EmotionDeploy] training_summary model_type=%s accuracy=%s timestamp=%s classes=%s",
                summary.get("model_type"),
                summary.get("accuracy"),
                summary.get("timestamp"),
                summary.get("class_names"),
            )

        app.logger.info(
            "[EmotionDeploy] Note: training is not automatic at backend startup. Use scripts/train_strict_pipeline.py in HF/CI training pipelines."
        )
    except Exception as exc:
        app.logger.warning("[EmotionDeploy] startup diagnostics failed: %s", exc)


def _ensure_assignment_integrity_columns(app: Flask) -> None:
    # """Backfill integrity columns if schema migration was skipped in an existing DB.

    # Silently no-ops on fresh databases where core tables don't exist yet.
    # """
    try:
        inspector = inspect(db.engine)
        table_names = set(inspector.get_table_names())
        # Guard: only run when the core schema is already established
        if "users" not in table_names or "test_assignments" not in table_names:
            return

        existing_columns = {
            str(col.get("name"))
            for col in inspector.get_columns("test_assignments")
            if col.get("name")
        }
        missing_columns = [
            name for name in ("require_camera", "require_emotion")
            if name not in existing_columns
        ]
        if not missing_columns:
            return

        app.logger.warning(
            "[SchemaGuard] Missing test_assignments columns detected: %s. Applying compatibility patch.",
            ", ".join(missing_columns),
        )

        default_true_literal = "1" if db.engine.dialect.name == "sqlite" else "TRUE"
        for column_name in missing_columns:
            db.session.execute(
                text(
                    f"ALTER TABLE test_assignments "
                    f"ADD COLUMN {column_name} BOOLEAN NOT NULL DEFAULT {default_true_literal}"
                )
            )
        db.session.commit()
        app.logger.info(
            "[SchemaGuard] Added missing test_assignments columns: %s",
            ", ".join(missing_columns),
        )
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(
            "[SchemaGuard] Could not apply assignment integrity patch (non-fatal): %s", exc
        )


def _ensure_teacher_interventions_table(app: Flask) -> None:
    """Create teacher_interventions table if it doesn't exist yet.

    Silently no-ops on fresh databases where core tables don't exist yet.
    """
    try:
        inspector = inspect(db.engine)
        existing = set(inspector.get_table_names())
        # Guard: only run as a safety net when the base schema already exists
        if "users" not in existing:
            return
        if "teacher_interventions" in existing:
            return

        app.logger.warning(
            "[SchemaGuard] Missing teacher_interventions table detected. Applying compatibility patch."
        )
        TeacherIntervention.__table__.create(bind=db.engine, checkfirst=True)
        app.logger.info("[SchemaGuard] Created teacher_interventions table")
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(
            "[SchemaGuard] Could not create teacher_interventions table (non-fatal): %s", exc
        )


def _ensure_teacher_rag_tables(app: Flask) -> None:
    """Create teacher RAG document tables if they don't exist yet.

    Silently no-ops on fresh databases where core tables don't exist yet.
    """
    try:
        inspector = inspect(db.engine)
        existing = set(inspector.get_table_names())
        # Guard: only run as a safety net when the base schema already exists
        if "users" not in existing:
            return

        created = []

        if "teacher_documents" not in existing:
            TeacherDocument.__table__.create(bind=db.engine, checkfirst=True)
            created.append("teacher_documents")

        if "teacher_document_chunks" not in existing:
            TeacherDocumentChunk.__table__.create(bind=db.engine, checkfirst=True)
            created.append("teacher_document_chunks")

        if "rag_retrieval_events" not in existing:
            RagRetrievalEvent.__table__.create(bind=db.engine, checkfirst=True)
            created.append("rag_retrieval_events")

        if "teacher_document_chunks" in existing:
            columns = {
                str(col.get("name"))
                for col in inspector.get_columns("teacher_document_chunks")
                if col.get("name")
            }
            if "embedding_vector_pg" not in columns:
                app.logger.warning(
                    "[SchemaGuard] Missing teacher_document_chunks.embedding_vector_pg column. Patching."
                )
                db.session.execute(
                    text("ALTER TABLE teacher_document_chunks ADD COLUMN IF NOT EXISTS embedding_vector_pg TEXT")
                )
                db.session.commit()

        if created:
            app.logger.warning(
                "[SchemaGuard] Missing RAG tables detected. Applying compatibility patch for: %s",
                ", ".join(created),
            )
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(
            "[SchemaGuard] Could not create RAG tables (non-fatal): %s", exc
        )
        
def _ensure_admin_tables(app: Flask) -> None:
    """Create new admin platform tables if they don't exist (Supabase-safe)."""
    from sqlalchemy import inspect as sa_inspect
    try:
        inspector = sa_inspect(db.engine)
        existing = set(inspector.get_table_names())
        created = []

        target_tables = {
            "audit_logs": AuditLog,
            "model_versions": ModelVersion,
            "training_jobs": TrainingJob,
            "mcq_pipeline_events": MCQPipelineEvent,
        }
        for table_name, model_cls in target_tables.items():
            if table_name not in existing:
                model_cls.__table__.create(bind=db.engine, checkfirst=True)
                created.append(table_name)

        if created:
            app.logger.info("[SchemaGuard] Created admin tables: %s", ", ".join(created))
    except Exception as exc:
        app.logger.exception("[SchemaGuard] Failed to create admin tables: %s", exc)


def create_app(config_name: str | None = None) -> Flask:
    app = Flask(__name__)
    SCHOOL_SLUG_HINT_COOKIE = 'elevate_school_slug_hint'

    def _normalize_school_slug(value: str | None) -> str | None:
        normalized = str(value or '').strip().lower()
        if not normalized:
            return None
        normalized = re.sub(r'[^a-z0-9\s_-]+', '', normalized)
        normalized = re.sub(r'[\s_]+', '-', normalized)
        normalized = re.sub(r'-+', '-', normalized).strip('-')
        return normalized or None

    environment_name = str(
        config_name or os.environ.get("FLASK_ENV", "development")
    ).strip().lower() or "development"

    @app.route("/api/health", methods=["GET"])
    def health():
        return {"status": "ok"}

    @app.route("/health", methods=["GET"])
    def health_simple():
        return {"status": "ok"}

    cfg = get_config(config_name)
    app.config.from_object(cfg)
    _assert_gemini_key_startup_health(app, environment_name)

    configure_logging(app)
    _log_emotion_deploy_status(app)

    allowed_origins = app.config.get("CORS_ORIGINS") or ["*"]
    if isinstance(allowed_origins, str):
        allowed_origins = [allowed_origins]

    CORS(
        app,
        resources={r"/api/*": {"origins": allowed_origins}},
        allow_headers=["Content-Type", "Authorization", "X-Admin-Token"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    db.init_app(app)

    with app.app_context():
        _ensure_assignment_integrity_columns(app)
        _ensure_teacher_interventions_table(app)
        _ensure_teacher_rag_tables(app)
        _ensure_admin_tables(app)

    # with app.app_context():
    #     db.create_all()
    #     app.logger.info("Database tables created/verified")

    app.url_map.strict_slashes = False

    # ── Existing blueprints ──────────────────────────────────────────────────
    app.register_blueprint(auth_bp,      url_prefix="/api/auth")
    app.register_blueprint(questions_bp, url_prefix="/api/questions")
    app.register_blueprint(progress_bp,  url_prefix="/api/progress")
    app.register_blueprint(emotions_bp,  url_prefix="/api/emotions")
    app.register_blueprint(reports_bp,   url_prefix="/api/reports")
    app.register_blueprint(admin_bp,     url_prefix="/api/admin")
    app.register_blueprint(teacher_bp,   url_prefix="/api/teacher")
    app.register_blueprint(student_bp,   url_prefix="/api/student")
    app.register_blueprint(settings_bp,  url_prefix="/api/settings")

    # ── NEW: AI blueprint ────────────────────────────────────────────────────
    app.register_blueprint(ai_emotion_bp, url_prefix="/api/ai/emotion")

    # ── Frontend static serving ──────────────────────────────────────────────
    FRONTEND_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'frontend')
    )

    def _serve_frontend_file(filename: str):
        if not filename or filename.startswith('api/'):
            return None

        normalized = filename.lstrip('/')
        full = os.path.join(FRONTEND_DIR, normalized)
        if os.path.exists(full):
            return send_from_directory(FRONTEND_DIR, normalized)
        return None

    def _slug_hint_response(slug: str):
        normalized_slug = _normalize_school_slug(slug)
        response = redirect('/index.html', code=302)
        if normalized_slug:
            response.set_cookie(
                SCHOOL_SLUG_HINT_COOKIE,
                normalized_slug,
                max_age=60 * 60 * 24 * 30,
                path='/',
                samesite='Lax',
            )
        else:
            response.delete_cookie(SCHOOL_SLUG_HINT_COOKIE)
        return response

    @app.get('/')
    def serve_index():
        return send_from_directory(FRONTEND_DIR, 'index.html')

    @app.get('/admin')
    def serve_admin():
        return send_from_directory(FRONTEND_DIR, 'admin.html')

    @app.get('/admin-schools')
    @app.get('/admin-schools.html')
    def redirect_admin_schools_to_admin():
        return redirect('/admin', code=302)

    @app.get('/favicon.ico')
    def serve_favicon():
        return Response(status=204)

    @app.get('/<slug>')
    @app.get('/<slug>/')
    def serve_slug_index(slug):
        slug_str = str(slug).strip('/')
        if slug_str.lower() == 'api':
            return not_found(None)

        # Single-segment frontend files (e.g. /index.html, /dashboard.html)
        # must be served normally, not treated as school slug routes.
        if '.' in slug_str:
            served = _serve_frontend_file(slug_str)
            if served is not None:
                return served
            return not_found(None)

        # Public auth page must never expose school slug in URL.
        return _slug_hint_response(slug_str)

    @app.get('/<slug>/<path:filename>')
    def serve_slug_frontend_files(slug, filename):
        if str(slug).lower() == 'api' or str(filename).startswith('api/'):
            return not_found(None)

        normalized_filename = str(filename).strip('/')

        if str(filename).strip('/').lower() == 'index.html':
            # Canonicalize auth page to non-slug URL.
            return _slug_hint_response(str(slug).strip('/'))

        # Avoid collisions with real static asset folders (e.g. /css/styles.css, /js/main.js)
        # that would otherwise be misread as /<slug>/<file> routes.
        combined = f"{str(slug).strip('/')}/{str(filename).lstrip('/')}"
        served = _serve_frontend_file(combined)
        if served is not None:
            return served

        served = _serve_frontend_file(filename)
        if served is not None:
            return served
        return not_found(None)

    @app.get('/<path:filename>')
    def serve_frontend_files(filename):
        served = _serve_frontend_file(filename)
        if served is not None:
            return served
        return not_found(None)

    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Resource not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {str(error)}")
        return {"error": "Internal server error"}, 500

    return app

import logging
# Mute Werkzeug health check spam
log = logging.getLogger('werkzeug')
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return 'GET /health' not in record.getMessage()
log.addFilter(HealthCheckFilter())

if __name__ == "__main__":
    env_name = os.environ.get("FLASK_ENV", "development")
    app = create_app(env_name)
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=(env_name == "development"),
    )
