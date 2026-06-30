"""Add user disable/suspension fields

Revision ID: 6b7c8d9e0f11
Revises: f4d91c2e7b11
Create Date: 2026-04-21 11:20:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6b7c8d9e0f11"
down_revision = "f4d91c2e7b11"
branch_labels = None
depends_on = None


def upgrade():
    # Use ADD COLUMN IF NOT EXISTS so this migration is safe to re-run
    # (e.g. if the column was already added manually or via db.create_all).
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        # Native PG statements — safe with IF NOT EXISTS
        op.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "is_disabled BOOLEAN NOT NULL DEFAULT FALSE"
        )
        op.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "disabled_at TIMESTAMP WITHOUT TIME ZONE"
        )
        op.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "disabled_reason VARCHAR(255)"
        )
        op.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
            "disabled_by INTEGER REFERENCES users(id) ON DELETE SET NULL"
        )
        # Create index only if it doesn't exist (PG 9.5+)
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_users_is_disabled ON users (is_disabled)"
        )
        # Remove the DEFAULT FALSE so the model owns the default going forward
        op.execute(
            "ALTER TABLE users ALTER COLUMN is_disabled DROP DEFAULT"
        )
    else:
        # SQLite fallback (batch mode required for SQLite)
        with op.batch_alter_table("users") as batch:
            batch.add_column(
                sa.Column("is_disabled", sa.Boolean(), nullable=False,
                          server_default=sa.false())
            )
            batch.add_column(sa.Column("disabled_at", sa.DateTime(), nullable=True))
            batch.add_column(sa.Column("disabled_reason", sa.String(length=255), nullable=True))
            batch.add_column(sa.Column("disabled_by", sa.Integer(), nullable=True))
            batch.create_index("ix_users_is_disabled", ["is_disabled"])
        with op.batch_alter_table("users") as batch:
            batch.alter_column("is_disabled", server_default=None)


def downgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_users_is_disabled")
        op.execute("ALTER TABLE users DROP COLUMN IF EXISTS disabled_by")
        op.execute("ALTER TABLE users DROP COLUMN IF EXISTS disabled_reason")
        op.execute("ALTER TABLE users DROP COLUMN IF EXISTS disabled_at")
        op.execute("ALTER TABLE users DROP COLUMN IF EXISTS is_disabled")
    else:
        with op.batch_alter_table("users") as batch:
            batch.drop_index("ix_users_is_disabled")
            batch.drop_column("disabled_by")
            batch.drop_column("disabled_reason")
            batch.drop_column("disabled_at")
            batch.drop_column("is_disabled")
