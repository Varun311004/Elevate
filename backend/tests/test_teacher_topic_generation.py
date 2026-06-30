from backend.app import create_app
from backend.models import Question, Test as OrmTest, TestQuestion as OrmTestQuestion, User, db
import pytest


@pytest.fixture
def app():
    app = create_app("testing")
    app.config.update({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.app_context():
        db.drop_all()
        db.create_all()
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def make_teacher(client, email="teacher-topic@example.com"):
    signup = client.post(
        "/api/auth/signup",
        json={"name": "Topic Teacher", "email": email, "password": "Password1!"},
    )
    assert signup.status_code == 201

    with client.application.app_context():
        teacher = User.query.filter_by(email=email).first()
        teacher.role = "teacher"
        db.session.commit()

    login = client.post("/api/auth/login", json={"email": email, "password": "Password1!"})
    assert login.status_code == 200
    return login.get_json()["token"]


def test_question_bank_persist_uses_topic_ai_service(client, monkeypatch):
    token = make_teacher(client, email="teacher-bank@example.com")

    def fake_topic_service(**kwargs):
        _ = kwargs
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "questions": [
                {
                    "text": "What is 4 + 5?",
                    "options": ["7", "8", "9", "10"],
                    "correct_index": 2,
                    "topic": "arithmetic",
                    "hint": "Add the two integers.",
                    "explanation": "4 + 5 equals 9.",
                },
                {
                    "text": "What is 6 x 3?",
                    "options": ["12", "18", "24", "30"],
                    "correct_index": 1,
                    "topic": "arithmetic",
                    "hint": "Think repeated addition.",
                    "explanation": "6 multiplied by 3 equals 18.",
                },
            ],
            "meta": {"llm_count": 2, "template_count": 0, "cache_hit": False},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fake_topic_service)

    response = client.post(
        "/api/teacher/question-bank/generate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "subject": "mathematics",
            "grade": "middle",
            "difficulty": "easy",
            "topic": "arithmetic",
            "count": 2,
            "persist": True,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["generated_count"] == 2
    assert payload["generation_status"]["service_llm_count"] == 2
    assert payload["generation_status"]["service_template_count"] == 0

    with client.application.app_context():
        rows = Question.query.filter_by(subject="mathematics", grade="middle", difficulty="easy").all()
        assert len(rows) == 2
        assert all(row.is_generated for row in rows)
        assert all((row.generation_meta or {}).get("source") == "topic_ai_service" for row in rows)


def test_create_test_partial_generation_keeps_available_questions_only(client, monkeypatch):
    token = make_teacher(client, email="teacher-test@example.com")

    def fake_topic_service(**kwargs):
        _ = kwargs
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "questions": [
                {
                    "text": "Which planet is known as the Red Planet?",
                    "options": ["Earth", "Mars", "Venus", "Jupiter"],
                    "correct_index": 1,
                    "topic": "astronomy",
                    "hint": "It has iron oxide on its surface.",
                    "explanation": "Mars appears red because of iron oxide dust.",
                }
            ],
            "meta": {"llm_count": 1, "template_count": 0, "cache_hit": False},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fake_topic_service)

    response = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Space Quiz",
            "subject": "science",
            "grade": "middle",
            "difficulty": "medium",
            "topic": "astronomy",
            "question_count": 3,
            "time_limit": 20,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["generated_count"] == 1
    assert payload["requested_count"] == 3
    assert payload["test"]["question_count"] == 1
    assert payload["generation_status"]["service_generated_count"] == 1
    assert payload["generation_status"]["technical_sample_count"] == 0

    with client.application.app_context():
        assert OrmTest.query.count() == 1
        assert OrmTestQuestion.query.count() == 1


def test_create_test_uses_technical_samples_when_topic_service_is_down(client, monkeypatch):
    token = make_teacher(client, email="teacher-preview@example.com")

    def fail_topic_service(**kwargs):
        _ = kwargs
        return {
            "ok": False,
            "status_code": 503,
            "error": "Topic AI service unavailable: timeout",
            "questions": [],
            "meta": {},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fail_topic_service)

    response = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Fallback Sample Quiz",
            "subject": "science",
            "grade": "middle",
            "difficulty": "easy",
            "topic": "cells",
            "question_count": 2,
            "time_limit": 20,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["generated_count"] == 2
    assert payload["requested_count"] == 2
    assert payload["generation_status"]["technical_fallback_used"] is True
    assert payload["generation_status"]["technical_sample_count"] >= 2

    with client.application.app_context():
        assert OrmTest.query.count() == 1
        assert OrmTestQuestion.query.count() == 2


def test_create_test_passes_title_and_description_to_topic_service(client, monkeypatch):
    token = make_teacher(client, email="teacher-context@example.com")
    captured = {}

    def fake_topic_service(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "questions": [
                {
                    "text": "Which quantity is measured in newtons?",
                    "options": ["Force", "Velocity", "Mass", "Energy"],
                    "correct_index": 0,
                    "topic": "motion",
                    "hint": "Think SI unit of force.",
                    "explanation": "Newton is the SI unit of force.",
                }
            ],
            "meta": {"llm_count": 1, "template_count": 0, "cache_hit": False},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fake_topic_service)

    response = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Motion Unit Test",
            "description": "Focus on equations of motion and force applications.",
            "subject": "science",
            "grade": "high",
            "difficulty": "medium",
            "topic": "motion",
            "question_count": 1,
            "time_limit": 20,
        },
    )

    assert response.status_code == 201
    assert captured.get("test_title") == "Motion Unit Test"
    assert captured.get("test_description") == "Focus on equations of motion and force applications."


def test_question_bank_uses_technical_samples_when_topic_service_is_down(client, monkeypatch):
    token = make_teacher(client, email="teacher-unavailable@example.com")

    def fake_topic_service(**kwargs):
        _ = kwargs
        return {
            "ok": False,
            "status_code": 503,
            "error": "Topic AI service unavailable: connection refused",
            "questions": [],
            "meta": {},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fake_topic_service)

    response = client.post(
        "/api/teacher/question-bank/generate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "subject": "science",
            "grade": "high",
            "difficulty": "hard",
            "topic": "optics",
            "count": 4,
            "persist": True,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["generated_count"] == 4
    assert payload["requested_count"] == 4
    assert payload["persisted"] is True
    assert payload["generation_status"]["technical_sample_count"] >= 4
    assert payload["generation_status"]["technical_fallback_used"] is True
    assert payload["generation_status"]["service_error"]
