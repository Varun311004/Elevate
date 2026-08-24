"""Create practice question bank.

Revision ID: c453e90ab6f3
Revises: a1b2c3d4e5f6
Create Date: 2026-08-14 20:55:21.391462

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql



# revision identifiers, used by Alembic.
revision: str = 'c453e90ab6f3'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "practice_questions",

        # Large ID range for a potentially multi-million-row table.
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            nullable=False,
        ),

        # Selection dimensions.
        sa.Column(
            "grade",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "subject",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "syllabus_topic",
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            "difficulty",
            sa.String(length=16),
            nullable=False,
        ),

        # Question content.
        sa.Column(
            "question_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "options",
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.Column(
            "correct_index",
            sa.SmallInteger(),
            nullable=False,
        ),
        sa.Column(
            "explanation",
            sa.Text(),
            nullable=False,
        ),

        # Exact duplicate prevention.
        sa.Column(
            "question_fingerprint",
            sa.String(length=64),
            nullable=False,
        ),

        # Generation/audit information.
        sa.Column(
            "generation_model",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "generation_batch_id",
            sa.String(length=128),
            nullable=True,
        ),
        sa.Column(
            "generation_meta",
            postgresql.JSONB(),
            nullable=True,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),

        # Allowed grade values.
        sa.CheckConstraint(
            "grade IN ('elementary', 'middle', 'high', 'college')",
            name="ck_practice_questions_grade",
        ),

        # Allowed STEM subjects.
        sa.CheckConstraint(
            "subject IN ('science', 'technology', 'engineering', 'mathematics')",
            name="ck_practice_questions_subject",
        ),

        # Allowed difficulty values.
        sa.CheckConstraint(
            "difficulty IN ('easy', 'medium', 'hard')",
            name="ck_practice_questions_difficulty",
        ),

        # 0=A, 1=B, 2=C, 3=D.
        sa.CheckConstraint(
            "correct_index >= 0 AND correct_index <= 3",
            name="ck_practice_questions_correct_index",
        ),

        # Prevent exact duplicate fingerprints.
        sa.UniqueConstraint(
            "question_fingerprint",
            name="uq_practice_questions_fingerprint",
        ),
    )

    # Main Practice retrieval index.
    op.create_index(
        "ix_practice_questions_combination",
        "practice_questions",
        [
            "grade",
            "subject",
            "syllabus_topic",
            "difficulty",
        ],
    )

    # Individual indexes.
    op.create_index(
        "ix_practice_questions_grade",
        "practice_questions",
        ["grade"],
    )

    op.create_index(
        "ix_practice_questions_subject",
        "practice_questions",
        ["subject"],
    )

    op.create_index(
        "ix_practice_questions_syllabus_topic",
        "practice_questions",
        ["syllabus_topic"],
    )

    op.create_index(
        "ix_practice_questions_difficulty",
        "practice_questions",
        ["difficulty"],
    )

    op.create_index(
        "ix_practice_questions_question_fingerprint",
        "practice_questions",
        ["question_fingerprint"],
    )

    op.create_index(
        "ix_practice_questions_generation_model",
        "practice_questions",
        ["generation_model"],
    )

    op.create_index(
        "ix_practice_questions_generation_batch_id",
        "practice_questions",
        ["generation_batch_id"],
    )

    op.create_index(
        "ix_practice_questions_created_at",
        "practice_questions",
        ["created_at"],
    )


def downgrade():
    op.drop_index(
        "ix_practice_questions_created_at",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_generation_batch_id",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_generation_model",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_question_fingerprint",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_difficulty",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_syllabus_topic",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_subject",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_grade",
        table_name="practice_questions",
    )

    op.drop_index(
        "ix_practice_questions_combination",
        table_name="practice_questions",
    )

    op.drop_table("practice_questions")