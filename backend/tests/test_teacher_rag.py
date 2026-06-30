import io

import pytest

from backend.app import create_app
from backend.models import Question, TeacherDocument, User, db


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


def make_teacher(client, email="teacher-rag@example.com"):
    signup = client.post(
        "/api/auth/signup",
        json={"name": "RAG Teacher", "email": email, "password": "Password1!"},
    )
    assert signup.status_code == 201

    with client.application.app_context():
        teacher = User.query.filter_by(email=email).first()
        teacher.role = "teacher"
        db.session.commit()
        teacher_id = teacher.id

    login = client.post("/api/auth/login", json={"email": email, "password": "Password1!"})
    assert login.status_code == 200
    return login.get_json()["token"], teacher_id


def test_teacher_document_upload_list_delete(client):
    token, _ = make_teacher(client, email="teacher-rag-docs@example.com")

    upload = client.post(
        "/api/teacher/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "Algebra Notes",
            "file": (io.BytesIO(b"Linear equations basics. Solve for x by balancing both sides."), "algebra-notes.txt"),
        },
        content_type="multipart/form-data",
    )

    assert upload.status_code == 201, upload.get_json()
    payload = upload.get_json()
    assert payload["document"]["title"] == "Algebra Notes"
    assert int(payload["document"]["chunk_count"] or 0) >= 1
    doc_id = int(payload["document"]["id"])

    listed = client.get(
        "/api/teacher/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listed.status_code == 200
    assert listed.get_json()["count"] == 1

    deleted = client.delete(
        f"/api/teacher/documents/{doc_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert deleted.status_code == 200

    listed_after = client.get(
        "/api/teacher/documents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert listed_after.status_code == 200
    assert listed_after.get_json()["count"] == 0


def test_create_test_rag_mode_adds_provenance_and_metrics(client, monkeypatch):
    token, teacher_id = make_teacher(client, email="teacher-rag-provenance@example.com")

    upload = client.post(
        "/api/teacher/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "Motion Notes",
            "file": (
                io.BytesIO(
                    (
                        b"Newton's first law explains inertia. Newton's second law links force, mass, "
                        b"and acceleration. Velocity and acceleration are key motion concepts."
                    )
                ),
                "motion-notes.txt",
            ),
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201, upload.get_json()
    doc_id = int(upload.get_json()["document"]["id"])

    captured = {}

    def fake_topic_service(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "questions": [
                {
                    "text": "Which law describes inertia?",
                    "options": ["First law", "Second law", "Third law", "Law of gravitation"],
                    "correct_index": 0,
                    "topic": "motion",
                    "explanation": "Inertia is described in Newton's first law.",
                },
                {
                    "text": "Force equals what expression?",
                    "options": ["m/a", "m+v", "m*a", "v/t"],
                    "correct_index": 2,
                    "topic": "motion",
                    "explanation": "Newton's second law gives F = m * a.",
                },
            ],
            "meta": {"llm_count": 2, "template_count": 0, "cache_hit": False},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fake_topic_service)

    created = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Physics Motion Quiz",
            "subject": "science",
            "grade": "high",
            "difficulty": "medium",
            "topic": "motion",
            "question_count": 2,
            "time_limit": 20,
            "generation_mode": "rag",
            "selected_document_ids": [doc_id],
            "rag_min_confidence": 0.0,
        },
    )

    assert created.status_code == 201, created.get_json()
    body = created.get_json()
    status = body["generation_status"]

    assert status["generation_mode_requested"] == "rag"
    assert status["generation_mode_effective"] == "rag"
    assert status["rag_retrieval_count"] >= 1
    assert int(status["rag_metrics"].get("provenance_count") or 0) >= 1
    assert captured.get("generation_mode") == "rag"
    assert bool(captured.get("rag_context"))

    with client.application.app_context():
        generated = Question.query.filter_by(generated_by=teacher_id).all()
        assert len(generated) >= 2
        assert any(isinstance((q.generation_meta or {}).get("provenance"), dict) for q in generated)


def test_create_test_rag_requires_processed_documents(client):
    token, _ = make_teacher(client, email="teacher-rag-fallback@example.com")

    created = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Biology Quick Check",
            "subject": "science",
            "grade": "high",
            "difficulty": "easy",
            "topic": "biology",
            "question_count": 1,
            "time_limit": 15,
            "generation_mode": "rag",
        },
    )

    assert created.status_code == 400, created.get_json()
    assert "processed document" in str(created.get_json().get("error", "")).lower()


def test_create_test_topic_mode_requires_sub_topic(client):
    token, _ = make_teacher(client, email="teacher-rag-topic-required@example.com")

    created = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Chemistry Topic Missing",
            "subject": "science",
            "grade": "high",
            "difficulty": "easy",
            "question_count": 1,
            "time_limit": 15,
            "generation_mode": "standard",
        },
    )

    assert created.status_code == 400, created.get_json()
    assert "sub topic" in str(created.get_json().get("error", "")).lower()


def test_teacher_document_upload_uses_r2_storage_when_enabled(client, monkeypatch):
    token, _ = make_teacher(client, email="teacher-rag-r2@example.com")
    deleted_objects = []

    monkeypatch.setattr(
        "backend.routes.teacher.resolve_document_storage_backend",
        lambda *_args, **_kwargs: {
            "requested": "r2",
            "effective": "r2",
            "fallback_reason": None,
        },
    )

    monkeypatch.setattr(
        "backend.routes.teacher.upload_document_to_r2",
        lambda *_args, **_kwargs: {
            "backend": "r2",
            "storage_path": "r2://unit-test-bucket/teacher-1/r2-doc.txt",
            "bucket": "unit-test-bucket",
            "key": "teacher-1/r2-doc.txt",
            "endpoint": "https://example.r2.cloudflarestorage.com",
            "public_url": "https://cdn.example.com/teacher-1/r2-doc.txt",
        },
    )

    def fake_delete_from_r2(storage_path=None, metadata=None):
        deleted_objects.append({"storage_path": storage_path, "metadata": metadata or {}})
        return True

    monkeypatch.setattr("backend.routes.teacher.delete_document_from_r2", fake_delete_from_r2)

    upload = client.post(
        "/api/teacher/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "R2 Physics Notes",
            "file": (
                io.BytesIO(b"Momentum equals mass times velocity. Conservation applies in closed systems."),
                "physics-r2.txt",
            ),
        },
        content_type="multipart/form-data",
    )

    assert upload.status_code == 201, upload.get_json()
    doc_payload = upload.get_json().get("document") or {}
    metadata = doc_payload.get("metadata") or {}
    doc_id = int(doc_payload.get("id") or 0)

    assert metadata.get("storage_backend") == "r2"
    assert metadata.get("r2_bucket") == "unit-test-bucket"
    assert metadata.get("r2_key")

    with client.application.app_context():
        saved = TeacherDocument.query.filter_by(id=doc_id).first()
        assert saved is not None
        assert str(saved.storage_path or "").startswith("r2://")

    deleted = client.delete(
        f"/api/teacher/documents/{doc_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert deleted.status_code == 200
    assert len(deleted_objects) == 1


def test_teacher_document_upload_enforces_document_quota(client, monkeypatch):
    token, _ = make_teacher(client, email="teacher-rag-quota@example.com")
    monkeypatch.setenv("RAG_MAX_DOCS_PER_TEACHER", "1")

    first = client.post(
        "/api/teacher/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "Doc One",
            "file": (io.BytesIO(b"first document body"), "doc-one.txt"),
        },
        content_type="multipart/form-data",
    )
    assert first.status_code == 201, first.get_json()

    second = client.post(
        "/api/teacher/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "Doc Two",
            "file": (io.BytesIO(b"second document body"), "doc-two.txt"),
        },
        content_type="multipart/form-data",
    )

    assert second.status_code == 409, second.get_json()
    assert "quota" in str(second.get_json().get("error", "")).lower()


def test_rag_observability_and_test_detail_citations(client, monkeypatch):
    token, _ = make_teacher(client, email="teacher-rag-observability@example.com")

    upload = client.post(
        "/api/teacher/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "title": "Electricity Notes",
            "file": (
                io.BytesIO(
                    b"Current is flow of charge. Resistance opposes current. Voltage is potential difference."
                ),
                "electricity-notes.txt",
            ),
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201, upload.get_json()
    doc_id = int(upload.get_json()["document"]["id"])

    def fake_topic_service(**_kwargs):
        return {
            "ok": True,
            "status_code": 200,
            "error": None,
            "questions": [
                {
                    "text": "What opposes electric current?",
                    "options": ["Resistance", "Voltage", "Charge", "Power"],
                    "correct_index": 0,
                    "topic": "electricity",
                    "explanation": "Resistance is opposition to current flow.",
                }
            ],
            "meta": {"llm_count": 1, "template_count": 0, "cache_hit": False},
            "service_url": "http://127.0.0.1:7860",
        }

    monkeypatch.setattr("backend.routes.teacher.generate_topic_mcqs", fake_topic_service)

    created = client.post(
        "/api/teacher/tests",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Electric Basics",
            "subject": "science",
            "grade": "high",
            "difficulty": "easy",
            "topic": "electricity",
            "question_count": 1,
            "time_limit": 15,
            "generation_mode": "rag",
            "selected_document_ids": [doc_id],
            "rag_min_confidence": 0.0,
        },
    )
    assert created.status_code == 201, created.get_json()
    test_id = int(created.get_json()["test"]["id"])

    detail = client.get(
        f"/api/teacher/tests/{test_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail.status_code == 200, detail.get_json()
    questions = detail.get_json().get("questions") or []
    assert len(questions) == 1
    assert isinstance(questions[0].get("provenance"), dict)
    assert isinstance(questions[0].get("retrieval_trace"), list)

    observability = client.get(
        "/api/teacher/rag/observability?days=30",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert observability.status_code == 200, observability.get_json()
    payload = observability.get_json()
    assert int(payload.get("summary", {}).get("total_events") or 0) >= 1
    assert int(payload.get("summary", {}).get("rag_events") or 0) >= 1
