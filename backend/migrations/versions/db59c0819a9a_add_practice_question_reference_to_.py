"""add practice question reference to answer logs

Revision ID: db59c0819a9a
Revises: c453e90ab6f3
Create Date: 2026-08-16 21:45:15.101086

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'db59c0819a9a'
down_revision: Union[str, Sequence[str], None] = 'c453e90ab6f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.alter_column(
        "answer_logs",
        "question_id",
        existing_type=sa.Integer(),
        nullable=True,
    )

    op.add_column(
        "answer_logs",
        sa.Column(
            "practice_question_id",
            sa.BigInteger(),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_answer_logs_practice_question_id",
        "answer_logs",
        ["practice_question_id"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_answer_logs_practice_question_id",
        "answer_logs",
        "practice_questions",
        ["practice_question_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    op.drop_constraint(
        "fk_answer_logs_practice_question_id",
        "answer_logs",
        type_="foreignkey",
    )

    op.drop_index(
        "ix_answer_logs_practice_question_id",
        table_name="answer_logs",
    )

    op.drop_column(
        "answer_logs",
        "practice_question_id",
    )