"""Add assignment integrity policy fields

Revision ID: 9f5d7de6d2ab
Revises: ca8a064eb00a
Create Date: 2026-04-19 14:05:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9f5d7de6d2ab"
down_revision = "ca8a064eb00a"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "test_assignments",
        sa.Column("require_camera", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "test_assignments",
        sa.Column("require_emotion", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade():
    op.drop_column("test_assignments", "require_emotion")
    op.drop_column("test_assignments", "require_camera")
