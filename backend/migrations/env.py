from logging.config import fileConfig
import os
import sys

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add backend to path so we can import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.models import db
# This import is essential for Alembic to 'see' the models.
# It executes the backend/models.py file, which registers the model
# classes (User, School, etc.) with the `db.metadata` object.
import backend.models
from backend.config import normalize_database_url

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config


def _configure_runtime_database_url() -> None:
    raw_url = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("TEST_DATABASE_URL")
        or os.environ.get("SUPABASE_DIRECT_CONNECTION_STRING")
        or ""
    )
    if not str(raw_url).strip():
        return

    normalized = normalize_database_url(raw_url)
    # ConfigParser treats % as interpolation; escape it for alembic config.
    config.set_main_option("sqlalchemy.url", normalized.replace("%", "%%"))


_configure_runtime_database_url()

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Get model metadata WITHOUT calling create_app ────────────────────────
# We only need the SQLAlchemy metadata for Alembic autogenerate support.
# Calling create_app() during migrations triggers SchemaGuard functions
# that try to CREATE TABLE before Alembic has run, causing confusing errors
# on fresh databases. The metadata is already populated by importing
# backend.models above.
target_metadata = db.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
