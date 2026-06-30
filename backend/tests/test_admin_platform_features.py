import pytest
from datetime import datetime

from backend.app import create_app
from backend.models import (
    db,
    User,
    TestResult,
    TrainingJob,
    ModelVersion,
    McqGenerationEvent,
    AdminAuditLog,
)


@pytest.fixture
def app():
    app = create_app('testing')
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
    })
    with app.app_context():
        db.drop_all()
        db.create_all()
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def admin_headers(client):
    return {'X-Admin-Token': client.application.config.get('ADMIN_TOKEN')}


def _create_user(name: str, email: str, role: str = 'student') -> User:
    user = User(name=name, email=email, password_hash='x', role=role, is_verified=True)
    db.session.add(user)
    db.session.commit()
    return user


def _create_test_result(user_id: int, subject: str, total_questions: int, correct_answers: int) -> TestResult:
    result = TestResult(
        user_id=user_id,
        subject=subject,
        total_questions=total_questions,
        correct_answers=correct_answers,
        average_time_per_question=7.5,
    )
    db.session.add(result)
    db.session.commit()
    return result


def test_test_results_score_filter_pagination_is_server_side(client, admin_headers):
    with client.application.app_context():
        student = _create_user('Student One', 'student1@example.com')
        _create_test_result(student.id, 'Math', 10, 9)   # 90%
        _create_test_result(student.id, 'Math', 20, 15)  # 75%
        _create_test_result(student.id, 'Math', 10, 5)   # 50%

    res_page_1 = client.get(
        '/api/admin/test-results?min_score=70&max_score=100&per_page=1&page=1',
        headers=admin_headers,
    )
    assert res_page_1.status_code == 200
    data_1 = res_page_1.get_json()
    assert data_1['total'] == 2
    assert len(data_1['items']) == 1
    assert 70 <= data_1['items'][0]['score_pct'] <= 100

    res_page_2 = client.get(
        '/api/admin/test-results?min_score=70&max_score=100&per_page=1&page=2',
        headers=admin_headers,
    )
    assert res_page_2.status_code == 200
    data_2 = res_page_2.get_json()
    assert data_2['total'] == 2
    assert len(data_2['items']) == 1
    assert 70 <= data_2['items'][0]['score_pct'] <= 100


def test_model_registry_summary_includes_current_previous_and_rollback(client, admin_headers):
    with client.application.app_context():
        db.session.add_all([
            ModelVersion(
                model_name='emotion_model',
                version_tag='v1',
                is_production=True,
                promoted_at=datetime(2026, 4, 20, 10, 0, 0),
                created_at=datetime(2026, 4, 20, 9, 0, 0),
            ),
            ModelVersion(
                model_name='emotion_model',
                version_tag='v2',
                created_at=datetime(2026, 4, 20, 11, 0, 0),
            ),
            ModelVersion(
                model_name='emotion_model',
                version_tag='v3',
                is_rollback_candidate=True,
                created_at=datetime(2026, 4, 20, 12, 0, 0),
            ),
        ])
        db.session.commit()

    res = client.get('/api/admin/ml/versions/registry-summary', headers=admin_headers)
    assert res.status_code == 200
    payload = res.get_json()
    assert payload['total'] == 1

    item = payload['items'][0]
    assert item['model_name'] == 'emotion_model'
    assert item['current_production']['version_tag'] == 'v1'
    assert item['previous']['version_tag'] == 'v3'
    assert item['rollback_target']['version_tag'] == 'v3'


def test_training_jobs_sync_persists_logs_artifacts_and_metrics(client, admin_headers, monkeypatch):
    with client.application.app_context():
        job = TrainingJob(
            job_id='job-sync-123',
            status='running',
            source='hf_strict',
            started_at=datetime(2026, 4, 21, 10, 0, 0),
        )
        db.session.add(job)
        db.session.commit()

    from backend.routes import admin as admin_routes

    def _fake_status(job_id):
        assert job_id == 'job-sync-123'
        return {
            'ok': True,
            'payload': {
                'status': 'succeeded',
                'duration_ms': 42000,
                'metrics': {'accuracy': 0.94},
                'artifact_manifest': {'model': 'r2://bucket/model.bin'},
                'stdout_tail': 'training complete',
                'stderr_tail': '',
            },
            'latency_ms': 25,
            'endpoint': 'http://example.local/training/strict/status/job-sync-123',
        }

    monkeypatch.setattr(admin_routes, 'get_hf_strict_training_status', _fake_status)

    res = client.get('/api/admin/ml/jobs?sync=1', headers=admin_headers)
    assert res.status_code == 200
    data = res.get_json()
    assert data['total'] == 1

    row = data['items'][0]
    assert row['status'] == 'succeeded'
    assert row['duration_ms'] == 42000
    assert row['metrics']['accuracy'] == 0.94
    assert row['artifact_manifest']['model'] == 'r2://bucket/model.bin'
    assert row['stdout_tail'] == 'training complete'


def test_mcq_observability_includes_failure_rate_and_generation_totals(client, admin_headers):
    with client.application.app_context():
        db.session.add_all([
            McqGenerationEvent(success=True, fallback_used=False, questions_requested=10, questions_generated=10, subject='Math'),
            McqGenerationEvent(success=False, fallback_used=True, questions_requested=8, questions_generated=4, subject='Math'),
            McqGenerationEvent(success=True, fallback_used=False, questions_requested=6, questions_generated=6, subject='Math'),
        ])
        db.session.commit()

    res = client.get('/api/admin/mcq/observability?days=30', headers=admin_headers)
    assert res.status_code == 200
    payload = res.get_json()
    summary = payload['summary']

    assert summary['total'] == 3
    assert summary['failures'] == 1
    assert summary['fallbacks'] == 1
    assert summary['questions_requested_total'] == 24
    assert summary['questions_generated_total'] == 20
    assert summary['failure_rate'] == pytest.approx(33.33, abs=0.01)
    assert summary['fallback_rate'] == pytest.approx(33.33, abs=0.01)


def test_disabling_user_writes_admin_audit_log(client, admin_headers):
    with client.application.app_context():
        target = _create_user('Target User', 'target@example.com')
        target_id = target.id

    res = client.post(
        f'/api/admin/users/{target_id}/disable',
        json={'reason': 'policy violation'},
        headers=admin_headers,
    )
    assert res.status_code == 200

    with client.application.app_context():
        log = (
            AdminAuditLog.query
            .filter_by(action='user.disable', target_type='user', target_id=str(target_id))
            .order_by(AdminAuditLog.id.desc())
            .first()
        )
        assert log is not None
        assert 'policy violation' in (log.notes or '')
