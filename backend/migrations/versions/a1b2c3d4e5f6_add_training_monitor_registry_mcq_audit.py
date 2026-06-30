"""add training_jobs model_versions mcq_generation_events admin_audit_logs

Revision ID: a1b2c3d4e5f6
Revises: 6b7c8d9e0f11
Create Date: 2026-04-21
"""
from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = '6b7c8d9e0f11'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()

    # ── Phase 4: training_jobs ────────────────────────────────────────────────
    if not _table_exists(bind, 'training_jobs'):
        op.create_table(
            'training_jobs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('job_id', sa.String(128), nullable=False),
            sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
            sa.Column('triggered_by', sa.Integer(),
                      sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
            sa.Column('source', sa.String(64), nullable=True),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('finished_at', sa.DateTime(), nullable=True),
            sa.Column('duration_ms', sa.BigInteger(), nullable=True),
            sa.Column('error_summary', sa.Text(), nullable=True),
            sa.Column('metrics_json', sa.JSON(), nullable=True),
            sa.Column('artifact_manifest_json', sa.JSON(), nullable=True),
            sa.Column('stdout_tail', sa.Text(), nullable=True),
            sa.Column('stderr_tail', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('job_id'),
        )
        op.create_index('ix_training_jobs_job_id', 'training_jobs', ['job_id'])
        op.create_index('ix_training_jobs_status', 'training_jobs', ['status'])
        op.create_index('ix_training_jobs_started_at', 'training_jobs', ['started_at'])
        op.create_index('ix_training_jobs_created_at', 'training_jobs', ['created_at'])
        op.create_index('ix_training_jobs_triggered_by', 'training_jobs', ['triggered_by'])

    # ── Phase 5: model_versions ───────────────────────────────────────────────
    if not _table_exists(bind, 'model_versions'):
        op.create_table(
            'model_versions',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('model_name', sa.String(128), nullable=False),
            sa.Column('version_tag', sa.String(64), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('promoted_at', sa.DateTime(), nullable=True),
            sa.Column('is_production', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('is_rollback_candidate', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('parent_version', sa.String(64), nullable=True),
            sa.Column('artifact_uri', sa.String(512), nullable=True),
            sa.Column('metrics_json', sa.JSON(), nullable=True),
            sa.Column('training_job_id', sa.String(128), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('model_name', 'version_tag', name='uix_model_version_tag'),
        )
        op.create_index('ix_model_versions_model_name', 'model_versions', ['model_name'])
        op.create_index('ix_model_versions_version_tag', 'model_versions', ['version_tag'])
        op.create_index('ix_model_versions_is_production', 'model_versions', ['is_production'])
        op.create_index('ix_model_versions_created_at', 'model_versions', ['created_at'])

    # ── Phase 6: mcq_generation_events ───────────────────────────────────────
    if not _table_exists(bind, 'mcq_generation_events'):
        op.create_table(
            'mcq_generation_events',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(),
                      sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
            sa.Column('school_id', sa.Integer(),
                      sa.ForeignKey('schools.id', ondelete='SET NULL'), nullable=True),
            sa.Column('test_id', sa.Integer(),
                      sa.ForeignKey('tests.id', ondelete='SET NULL'), nullable=True),
            sa.Column('request_source', sa.String(64), nullable=True),
            sa.Column('generation_mode', sa.String(32), nullable=True),
            sa.Column('success', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('fallback_used', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('latency_ms', sa.Float(), nullable=True),
            sa.Column('reason_code', sa.String(64), nullable=True),
            sa.Column('questions_requested', sa.Integer(), nullable=True),
            sa.Column('questions_generated', sa.Integer(), nullable=True),
            sa.Column('subject', sa.String(64), nullable=True),
            sa.Column('grade', sa.String(32), nullable=True),
            sa.Column('metadata_json', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_mcq_gen_events_created_at', 'mcq_generation_events', ['created_at'])
        op.create_index('ix_mcq_gen_events_success', 'mcq_generation_events', ['success'])
        op.create_index('ix_mcq_gen_events_fallback_used', 'mcq_generation_events', ['fallback_used'])
        op.create_index('ix_mcq_gen_events_user_id', 'mcq_generation_events', ['user_id'])
        op.create_index('ix_mcq_gen_events_school_id', 'mcq_generation_events', ['school_id'])
        op.create_index('ix_mcq_gen_events_subject', 'mcq_generation_events', ['subject'])
        op.create_index('ix_mcq_gen_events_generation_mode', 'mcq_generation_events', ['generation_mode'])

    # ── Phase 7: admin_audit_logs ─────────────────────────────────────────────
    if not _table_exists(bind, 'admin_audit_logs'):
        op.create_table(
            'admin_audit_logs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('action', sa.String(128), nullable=False),
            sa.Column('actor_id', sa.Integer(),
                      sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
            sa.Column('target_type', sa.String(64), nullable=True),
            sa.Column('target_id', sa.String(64), nullable=True),
            sa.Column('before_json', sa.JSON(), nullable=True),
            sa.Column('after_json', sa.JSON(), nullable=True),
            sa.Column('ip', sa.String(64), nullable=True),
            sa.Column('user_agent', sa.Text(), nullable=True),
            sa.Column('request_id', sa.String(128), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('ix_admin_audit_logs_action', 'admin_audit_logs', ['action'])
        op.create_index('ix_admin_audit_logs_actor_id', 'admin_audit_logs', ['actor_id'])
        op.create_index('ix_admin_audit_logs_target_type', 'admin_audit_logs', ['target_type'])
        op.create_index('ix_admin_audit_logs_target_id', 'admin_audit_logs', ['target_id'])
        op.create_index('ix_admin_audit_logs_created_at', 'admin_audit_logs', ['created_at'])


def downgrade():
    op.drop_table('admin_audit_logs')
    op.drop_table('mcq_generation_events')
    op.drop_table('model_versions')
    op.drop_table('training_jobs')


def _table_exists(bind, table_name: str) -> bool:
    """Check if a table already exists — prevents DuplicateTable on re-run."""
    from sqlalchemy import inspect
    try:
        return table_name in inspect(bind).get_table_names()
    except Exception:
        return False
