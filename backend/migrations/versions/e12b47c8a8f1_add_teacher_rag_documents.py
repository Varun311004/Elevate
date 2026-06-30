"""Add teacher RAG document and chunk tables

Revision ID: e12b47c8a8f1
Revises: c4a1e9b7f2d1
Create Date: 2026-04-20 00:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e12b47c8a8f1"
down_revision = "c4a1e9b7f2d1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "teacher_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("teacher_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("school_id", sa.Integer(), sa.ForeignKey("schools.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_ext", sa.String(length=16), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="processing"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=False),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_index("ix_teacher_documents_teacher_id", "teacher_documents", ["teacher_id"])
    op.create_index("ix_teacher_documents_school_id", "teacher_documents", ["school_id"])
    op.create_index("ix_teacher_documents_file_ext", "teacher_documents", ["file_ext"])
    op.create_index("ix_teacher_documents_content_sha256", "teacher_documents", ["content_sha256"])
    op.create_index("ix_teacher_documents_status", "teacher_documents", ["status"])
    op.create_index("ix_teacher_documents_uploaded_at", "teacher_documents", ["uploaded_at"])

    op.create_table(
        "teacher_document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("teacher_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("teacher_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_vector", sa.JSON(), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("embedding_provider", sa.String(length=64), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("embedding_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("vector_store", sa.String(length=32), nullable=False, server_default="python"),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uix_teacher_document_chunk_order"),
        sa.UniqueConstraint("document_id", "chunk_id", name="uix_teacher_document_chunk_id"),
    )

    op.create_index("ix_teacher_document_chunks_document_id", "teacher_document_chunks", ["document_id"])
    op.create_index("ix_teacher_document_chunks_teacher_id", "teacher_document_chunks", ["teacher_id"])
    op.create_index("ix_teacher_document_chunks_chunk_id", "teacher_document_chunks", ["chunk_id"])
    op.create_index("ix_teacher_document_chunks_text_hash", "teacher_document_chunks", ["text_hash"])
    op.create_index("ix_teacher_document_chunks_embedding_status", "teacher_document_chunks", ["embedding_status"])


def downgrade():
    op.drop_index("ix_teacher_document_chunks_embedding_status", table_name="teacher_document_chunks")
    op.drop_index("ix_teacher_document_chunks_text_hash", table_name="teacher_document_chunks")
    op.drop_index("ix_teacher_document_chunks_chunk_id", table_name="teacher_document_chunks")
    op.drop_index("ix_teacher_document_chunks_teacher_id", table_name="teacher_document_chunks")
    op.drop_index("ix_teacher_document_chunks_document_id", table_name="teacher_document_chunks")
    op.drop_table("teacher_document_chunks")

    op.drop_index("ix_teacher_documents_uploaded_at", table_name="teacher_documents")
    op.drop_index("ix_teacher_documents_status", table_name="teacher_documents")
    op.drop_index("ix_teacher_documents_content_sha256", table_name="teacher_documents")
    op.drop_index("ix_teacher_documents_file_ext", table_name="teacher_documents")
    op.drop_index("ix_teacher_documents_school_id", table_name="teacher_documents")
    op.drop_index("ix_teacher_documents_teacher_id", table_name="teacher_documents")
    op.drop_table("teacher_documents")
