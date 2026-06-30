from datetime import datetime, timedelta, timezone

import pytest

from backend.app import create_app
from backend.models import (
    db,
    EmotionLog,
    Question,
    Test,
    TestAssignment,
    TestQuestion,
    User,
)
from backend.security import create_access_token, hash_password


@pytest.fixture
def app():
    app = create_app('testing')
    app.config.update({'TESTING': True, 'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    with app.app_context():
        db.drop_all()
        db.create_all()
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def _utcnow_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _create_user(email: str, *, role: str, grade: str | None = None) -> User:
    return User(
        name=email.split('@')[0].title(),
        email=email,
        password_hash=hash_password('Password1!'),
        role=role,
        grade=grade,
    )


def _auth_header(app, user_id: int):
    with app.app_context():
        user = db.session.get(User, user_id)
        token = create_access_token(user)
        return {'Authorization': f'Bearer {token}'}


def _seed_test_with_question(app, *, grade: str, teacher_id: int, is_published: bool):
    with app.app_context():
        test = Test(
            title='Integrity Test',
            subject='science',
            grade=grade,
            difficulty='easy',
            time_limit=30,
            question_count=1,
            total_points=10,
            is_active=True,
            is_published=is_published,
            created_by=teacher_id,
        )
        question = Question(
            subject='science',
            grade=grade,
            difficulty='easy',
            text='What is H2O?',
            options=['Water', 'Oxygen'],
            correct_index=0,
            explanation='H2O is water.',
        )
        db.session.add_all([test, question])
        db.session.flush()

        db.session.add(TestQuestion(test_id=test.id, question_id=question.id, order=1, points=1))
        db.session.commit()

        return test.id, question.id


def _add_emotion_log(app, *, user_id: int, seconds_ago: int = 0):
    with app.app_context():
        db.session.add(
            EmotionLog(
                user_id=user_id,
                emotion='focused',
                confidence=0.95,
                timestamp=_utcnow_naive() - timedelta(seconds=seconds_ago),
            )
        )
        db.session.commit()


def _age_emotion_logs_outside_window(app, *, user_id: int):
    with app.app_context():
        stale_time = _utcnow_naive() - timedelta(minutes=10)
        logs = EmotionLog.query.filter_by(user_id=user_id).all()
        for log in logs:
            log.timestamp = stale_time
        db.session.commit()


def test_assigned_start_requires_recent_emotion_log(client, app):
    with app.app_context():
        teacher = _create_user('teacher.integrity@example.com', role='teacher', grade='middle')
        student = _create_user('student.integrity@example.com', role='student', grade='middle')
        db.session.add_all([teacher, student])
        db.session.flush()
        teacher_id = teacher.id
        student_id = student.id
        db.session.commit()

    test_id, _ = _seed_test_with_question(app, grade='middle', teacher_id=teacher_id, is_published=False)

    with app.app_context():
        db.session.add(
            TestAssignment(
                test_id=test_id,
                student_id=student_id,
                assigned_by=teacher_id,
                status='assigned',
                allow_late=True,
                require_camera=True,
                require_emotion=True,
            )
        )
        db.session.commit()

    response = client.post(f'/api/student/tests/{test_id}/start', headers=_auth_header(app, student_id))

    assert response.status_code == 403
    payload = response.get_json()
    assert payload.get('code') == 'assignment_integrity_required'
    assert payload.get('phase') == 'start'


def test_assigned_start_returns_policy_with_recent_emotion_log(client, app):
    with app.app_context():
        teacher = _create_user('teacher.policy@example.com', role='teacher', grade='middle')
        student = _create_user('student.policy@example.com', role='student', grade='middle')
        db.session.add_all([teacher, student])
        db.session.flush()
        teacher_id = teacher.id
        student_id = student.id
        db.session.commit()

    test_id, _ = _seed_test_with_question(app, grade='middle', teacher_id=teacher_id, is_published=False)

    with app.app_context():
        db.session.add(
            TestAssignment(
                test_id=test_id,
                student_id=student_id,
                assigned_by=teacher_id,
                status='assigned',
                allow_late=True,
                require_camera=True,
                require_emotion=True,
            )
        )
        db.session.commit()

    _add_emotion_log(app, user_id=student_id, seconds_ago=0)

    response = client.post(f'/api/student/tests/{test_id}/start', headers=_auth_header(app, student_id))

    assert response.status_code == 201
    payload = response.get_json()
    assert payload.get('assignment_policy', {}).get('require_camera') is True
    assert payload.get('assignment_policy', {}).get('require_emotion') is True


def test_assigned_submit_answer_requires_recent_emotion_log(client, app):
    with app.app_context():
        teacher = _create_user('teacher.answer@example.com', role='teacher', grade='middle')
        student = _create_user('student.answer@example.com', role='student', grade='middle')
        db.session.add_all([teacher, student])
        db.session.flush()
        teacher_id = teacher.id
        student_id = student.id
        db.session.commit()

    test_id, question_id = _seed_test_with_question(app, grade='middle', teacher_id=teacher_id, is_published=False)

    with app.app_context():
        db.session.add(
            TestAssignment(
                test_id=test_id,
                student_id=student_id,
                assigned_by=teacher_id,
                status='assigned',
                allow_late=True,
                require_camera=True,
                require_emotion=True,
            )
        )
        db.session.commit()

    _add_emotion_log(app, user_id=student_id, seconds_ago=0)

    start_response = client.post(f'/api/student/tests/{test_id}/start', headers=_auth_header(app, student_id))
    assert start_response.status_code == 201

    _age_emotion_logs_outside_window(app, user_id=student_id)

    answer_response = client.post(
        f'/api/student/tests/{test_id}/answer',
        headers=_auth_header(app, student_id),
        json={'question_id': question_id, 'selected_index': 0, 'time_spent': 5},
    )

    assert answer_response.status_code == 403
    payload = answer_response.get_json()
    assert payload.get('code') == 'assignment_integrity_required'
    assert payload.get('phase') == 'answer'


def test_assigned_finish_requires_recent_emotion_log(client, app):
    with app.app_context():
        teacher = _create_user('teacher.finish@example.com', role='teacher', grade='middle')
        student = _create_user('student.finish@example.com', role='student', grade='middle')
        db.session.add_all([teacher, student])
        db.session.flush()
        teacher_id = teacher.id
        student_id = student.id
        db.session.commit()

    test_id, question_id = _seed_test_with_question(app, grade='middle', teacher_id=teacher_id, is_published=False)

    with app.app_context():
        db.session.add(
            TestAssignment(
                test_id=test_id,
                student_id=student_id,
                assigned_by=teacher_id,
                status='assigned',
                allow_late=True,
                require_camera=True,
                require_emotion=True,
            )
        )
        db.session.commit()

    _add_emotion_log(app, user_id=student_id, seconds_ago=0)

    start_response = client.post(f'/api/student/tests/{test_id}/start', headers=_auth_header(app, student_id))
    assert start_response.status_code == 201

    # Keep a recent log for answer, then age logs before finish.
    _add_emotion_log(app, user_id=student_id, seconds_ago=0)

    answer_response = client.post(
        f'/api/student/tests/{test_id}/answer',
        headers=_auth_header(app, student_id),
        json={'question_id': question_id, 'selected_index': 0, 'time_spent': 5},
    )
    assert answer_response.status_code == 200

    _age_emotion_logs_outside_window(app, user_id=student_id)

    finish_response = client.post(f'/api/student/tests/{test_id}/finish', headers=_auth_header(app, student_id))

    assert finish_response.status_code == 403
    payload = finish_response.get_json()
    assert payload.get('code') == 'assignment_integrity_required'
    assert payload.get('phase') == 'finish'


def test_practice_start_without_assignment_is_not_blocked(client, app):
    with app.app_context():
        teacher = _create_user('teacher.practice@example.com', role='teacher', grade='middle')
        student = _create_user('student.practice@example.com', role='student', grade='middle')
        db.session.add_all([teacher, student])
        db.session.flush()
        teacher_id = teacher.id
        student_id = student.id
        db.session.commit()

    test_id, _ = _seed_test_with_question(app, grade='middle', teacher_id=teacher_id, is_published=True)

    response = client.post(f'/api/student/tests/{test_id}/start', headers=_auth_header(app, student_id))

    assert response.status_code == 201
    payload = response.get_json()
    assert payload.get('message') == 'Test started successfully'
