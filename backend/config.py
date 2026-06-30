import os
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


DEFAULT_SQLITE_DB_URL = "sqlite:///elevate_dev.db"


def _first_non_empty_env(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _coerce_int(raw_value: str | None, default: int) -> int:
    try:
        return int(raw_value or default)
    except (TypeError, ValueError):
        return default


def _build_supabase_direct_url(project_url: str) -> str | None:
    parts = urlsplit(project_url)
    host = (parts.hostname or "").strip().lower()
    if not host.endswith(".supabase.co"):
        return None

    project_ref = host.split(".", 1)[0]
    if not project_ref:
        return None

    password = _first_non_empty_env("SUPABASE_DB_PASSWORD")
    if not password:
        return None

    user = _first_non_empty_env("SUPABASE_DB_USER") or "postgres"
    database = _first_non_empty_env("SUPABASE_DB_NAME") or "postgres"
    port = _coerce_int(_first_non_empty_env("SUPABASE_DB_PORT"), 5432)
    encoded_password = quote(password, safe="")
    return f"postgresql://{user}:{encoded_password}@db.{project_ref}.supabase.co:{port}/{database}"


def normalize_database_url(raw_value: str | None) -> str:
    # First, check for the recommended environment variable for Supabase deployments.
    # This provides a direct path for production and preview environments, bypassing
    # complex normalization logic that might fail in containerized/serverless runtimes.
    pooler_url = _first_non_empty_env("SUPABASE_POOLER_CONNECTION_STRING")
    if pooler_url:
        if pooler_url.startswith("postgres://"):
            return pooler_url.replace("postgres://", "postgresql://", 1)
        if pooler_url.startswith("postgresql://"):
            return pooler_url
        raise ValueError(
            "Invalid SUPABASE_POOLER_CONNECTION_STRING: Must start with 'postgresql://' or 'postgres://'"
        )

    value = (raw_value or "").strip()
    if not value:
        value = _first_non_empty_env(
            "SUPABASE_DIRECT_CONNECTION_STRING",
            "SUPABASE_DB_URL",
            "SUPABASE_DATABASE_URL",
        )
    if not value:
        return DEFAULT_SQLITE_DB_URL

    lowered = value.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        preferred_supabase_url = _first_non_empty_env(
            "SUPABASE_DIRECT_CONNECTION_STRING",
            "SUPABASE_DB_URL",
            "SUPABASE_DATABASE_URL",
        )
        if preferred_supabase_url:
            value = preferred_supabase_url
        else:
            inferred_url = _build_supabase_direct_url(value)
            if inferred_url:
                value = inferred_url

    if value.startswith("postgres://"):
        value = value.replace("postgres://", "postgresql://", 1)

    if value.startswith("http://") or value.startswith("https://"):
        raise ValueError(
            "Invalid DATABASE_URL scheme 'http/https'. Use a PostgreSQL DSN like "
            "postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres "
            "or set SUPABASE_DIRECT_CONNECTION_STRING / SUPABASE_DB_PASSWORD helpers."
        )

    if value.startswith("postgresql://"):
        parts = urlsplit(value)
        host = (parts.hostname or "").lower()
        is_local = host in {"localhost", "127.0.0.1"}

        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if not is_local and "sslmode" not in query:
            query["sslmode"] = "require"

        value = urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(query),
                parts.fragment,
            )
        )

    if not (value.startswith("postgresql://") or value.startswith("sqlite://")):
        raise ValueError(
            "Unsupported DATABASE_URL scheme. Supported: postgresql://, postgres://, sqlite://"
        )

    return value


# Backward-compatible alias used across existing modules.
def _normalize_database_url(raw_value: str | None) -> str:
    return normalize_database_url(raw_value)


def _parse_cors_origins(raw_value: str | None):
    if not raw_value:
        return ["http://localhost:8000", "http://127.0.0.1:8000"]
    items = [item.strip() for item in str(raw_value).split(",")]
    return [item for item in items if item]


def _engine_options_for(uri: str):
    if uri.startswith("sqlite"):
        # QueuePool-only options are invalid with SQLite StaticPool
        # (commonly used in tests, especially with sqlite:///:memory:).
        return {
            "connect_args": {"check_same_thread": False}
        }

    options = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 3,
        "max_overflow": 2,
        "pool_timeout": 10,
    }

    if uri.startswith("postgresql"):
        options["connect_args"] = {
            "connect_timeout": 8,
            "options": "-c statement_timeout=25000"
        }

    return options


class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = normalize_database_url(os.environ.get("DATABASE_URL", DEFAULT_SQLITE_DB_URL))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = _engine_options_for(SQLALCHEMY_DATABASE_URI)
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET", "dev-jwt-secret")
    ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev-admin-token")
    CORS_ORIGINS = _parse_cors_origins(os.environ.get("CORS_ORIGINS"))


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False
    PREFERRED_URL_SCHEME = "https"


class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = normalize_database_url(os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:"))
    SQLALCHEMY_ENGINE_OPTIONS = _engine_options_for(SQLALCHEMY_DATABASE_URI)


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(name: str | None):
    if not name:
        name = os.environ.get("FLASK_ENV", "development")
    return CONFIG_MAP.get(name, DevelopmentConfig)


