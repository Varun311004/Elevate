"""Add teacher interventions table

Revision ID: c4a1e9b7f2d1
Revises: 9f5d7de6d2ab
Create Date: 2026-04-19 22:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c4a1e9b7f2d1"
down_revision = "9f5d7de6d2ab"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "teacher_interventions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("teacher_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action_type", sa.String(length=64), nullable=False, server_default="note"),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="planned"),
        sa.Column("subject", sa.String(length=64), nullable=True),
        sa.Column("topic", sa.String(length=128), nullable=True),
        sa.Column("due_at", sa.DateTime(), nullable=True),
        sa.Column("classroom_id", sa.Integer(), sa.ForeignKey("classrooms.id", ondelete="SET NULL"), nullable=True),
        sa.Column("related_test_id", sa.Integer(), sa.ForeignKey("tests.id", ondelete="SET NULL"), nullable=True),
        sa.Column("student_ids", sa.JSON(), nullable=True),
        sa.Column("assignment_ids", sa.JSON(), nullable=True),
        sa.Column("cluster_payload", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_index("ix_teacher_interventions_teacher_id", "teacher_interventions", ["teacher_id"])
    op.create_index("ix_teacher_interventions_action_type", "teacher_interventions", ["action_type"])
    op.create_index("ix_teacher_interventions_status", "teacher_interventions", ["status"])
    op.create_index("ix_teacher_interventions_subject", "teacher_interventions", ["subject"])
    op.create_index("ix_teacher_interventions_topic", "teacher_interventions", ["topic"])
    op.create_index("ix_teacher_interventions_due_at", "teacher_interventions", ["due_at"])
    op.create_index("ix_teacher_interventions_classroom_id", "teacher_interventions", ["classroom_id"])
    op.create_index("ix_teacher_interventions_related_test_id", "teacher_interventions", ["related_test_id"])
    op.create_index("ix_teacher_interventions_created_at", "teacher_interventions", ["created_at"])


def downgrade():
    op.drop_index("ix_teacher_interventions_created_at", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_related_test_id", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_classroom_id", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_due_at", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_topic", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_subject", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_status", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_action_type", table_name="teacher_interventions")
    op.drop_index("ix_teacher_interventions_teacher_id", table_name="teacher_interventions")
    op.drop_table("teacher_interventions")
