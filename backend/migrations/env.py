from logging.config import fileConfig
import os
import sys
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from dotenv import load_dotenv


# ============================================================================
# PROJECT PATHS
# ============================================================================

# Current file:
# project/backend/migrations/env.py
#
# parents:
#   0 = migrations
#   1 = backend
#   2 = project

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

BACKEND_DIR = (
    PROJECT_ROOT / "backend"
)


# ============================================================================
# LOAD ENVIRONMENT
# ============================================================================

# Project-level .env
load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)

# Optional backend/.env
load_dotenv(
    BACKEND_DIR / ".env",
    override=False,
)


# ============================================================================
# PYTHON IMPORT PATH
# ============================================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================================
# APPLICATION METADATA
# ============================================================================

from backend.models import db

# Import models so all SQLAlchemy tables are registered
# in db.metadata for Alembic autogenerate.
import backend.models

from backend.config import normalize_database_url


# ============================================================================
# ALEMBIC CONFIG
# ============================================================================

config = context.config


def _configure_runtime_database_url() -> None:
    """
    Load DATABASE_URL from .env and inject it into Alembic.

    Primary variable:
        DATABASE_URL

    Optional fallbacks:
        TEST_DATABASE_URL
        SUPABASE_DIRECT_CONNECTION_STRING
    """

    raw_url = (
        os.environ.get(
            "DATABASE_URL"
        )
        or os.environ.get(
            "TEST_DATABASE_URL"
        )
        or os.environ.get(
            "SUPABASE_DIRECT_CONNECTION_STRING"
        )
        or ""
    )

    raw_url = str(
        raw_url
    ).strip()

    if not raw_url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add DATABASE_URL to the project .env file."
        )

    normalized = (
        normalize_database_url(
            raw_url
        )
    )

    # ConfigParser interprets '%' specially.
    config.set_main_option(
        "sqlalchemy.url",
        normalized.replace(
            "%",
            "%%",
        ),
    )


_configure_runtime_database_url()


# ============================================================================
# LOGGING
# ============================================================================

if config.config_file_name is not None:
    fileConfig(
        config.config_file_name
    )


# ============================================================================
# ALEMBIC METADATA
# ============================================================================

# Do NOT call create_app() here.
target_metadata = db.metadata


# ============================================================================
# OFFLINE MODE
# ============================================================================

def run_migrations_offline() -> None:
    """
    Run migrations without opening a database connection.
    """

    url = config.get_main_option(
        "sqlalchemy.url"
    )

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named"
        },
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================================
# ONLINE MODE
# ============================================================================

def run_migrations_online() -> None:
    """
    Run migrations against the PostgreSQL/Supabase database.
    """

    section = config.get_section(
        config.config_ini_section
    )

    if section is None:
        raise RuntimeError(
            "Alembic configuration section "
            "could not be loaded."
        )

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# ============================================================================
# ENTRY POINT
# ============================================================================

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()