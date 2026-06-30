from backend.app import create_app
from backend.models import db, User, Question
import pytest

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


def make_teacher(client):
    # signup (use a valid name per validation rules)
    res = client.post('/api/auth/signup', json={'name': 'Teacher One', 'email': 't@example.com', 'password': 'Password1!'})
    assert res.status_code == 201
    # set as teacher
    from backend.models import User, School
    user = User.query.filter_by(email='t@example.com').first()
    user.role = 'teacher'
    db.session.commit()
    # login
    r = client.post('/api/auth/login', json={'email': 't@example.com', 'password': 'Password1!'})
    assert r.status_code == 200
    return r.get_json()['token']


def test_generate_questions_persisted(client):
    token = make_teacher(client)

    res = client.post('/api/questions/generate', json={'topic': 'algebra', 'difficulty': 'easy', 'count': 3, 'subject': 'math'}, headers={'Authorization': f'Bearer {token}'})
    assert res.status_code == 201
    data = res.get_json()
    assert 'generated' in data
    assert len(data['generated']) == 3

    # ensure questions persisted and flagged
    with client.application.app_context():
        cnt = Question.query.filter_by(is_generated=True).count()
        assert cnt >= 3


def test_generate_with_seed_persists_seed(client):
    token = make_teacher(client)

    res = client.post('/api/questions/generate', json={'topic': 'algebra', 'difficulty': 'easy', 'count': 2, 'subject': 'math', 'seed': 12345}, headers={'Authorization': f'Bearer {token}'})
    assert res.status_code == 201
    data = res.get_json()
    assert 'generated' in data
    assert len(data['generated']) == 2

    # ensure seed is included in returned generation_meta and persisted
    gen_meta = data['generated'][0].get('generation_meta')
    assert gen_meta is not None
    assert gen_meta.get('seed') == 12345

    with client.application.app_context():
        from backend.models import Question
        q = Question.query.filter_by(is_generated=True).first()
        assert q is not None
        assert q.generation_meta.get('seed') == 12345


def test_generate_stem_questions_deterministic(client):
    # Import generator and run with the same seed twice
    from backend.question_generator import generate_stem_questions
    with client.application.app_context():
        a = generate_stem_questions('mathematics', 'middle', 'easy', count=5, seed=42)
        b = generate_stem_questions('mathematics', 'middle', 'easy', count=5, seed=42)
        assert a == b

        # Different seed should usually produce different outputs
        c = generate_stem_questions('mathematics', 'middle', 'easy', count=5, seed=43)
        assert c != a
