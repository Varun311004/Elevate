from backend.app import create_app
from backend.models import db, School, User
from backend.security import hash_password
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


def test_signup_login_me_and_refresh(client):
    # signup
    r = client.post('/api/auth/signup', json={'name': 'Alice', 'email': 'a@example.com', 'password': 'Password1!'})
    assert r.status_code == 201
    data = r.get_json()
    assert 'token' in data

    # login
    r2 = client.post('/api/auth/login', json={'email': 'a@example.com', 'password': 'Password1!'})
    assert r2.status_code == 200
    token = r2.get_json()['token']

    # me
    r3 = client.get('/api/auth/me', headers={'Authorization': f'Bearer {token}'})
    assert r3.status_code == 200
    assert r3.get_json()['user']['email'] == 'a@example.com'

    # refresh
    r4 = client.post('/api/auth/refresh', headers={'Authorization': f'Bearer {token}'})
    assert r4.status_code == 200
    assert 'token' in r4.get_json()


def test_role_enforcement(client):
    # signup student
    client.post('/api/auth/signup', json={'name': 'Stu', 'email': 'stu@example.com', 'password': 'Password1!'})
    login = client.post('/api/auth/login', json={'email': 'stu@example.com', 'password': 'Password1!'})
    tok = login.get_json()['token']

    # student trying to access teacher endpoint should be forbidden
    res = client.get('/api/teacher/reports', headers={'Authorization': f'Bearer {tok}'})
    assert res.status_code == 403 or res.status_code == 401

    # promote to teacher and re-login
    from backend.models import User, db as _db
    with client.application.app_context():
        u = User.query.filter_by(email='stu@example.com').first()
        u.role = 'teacher'
        _db.session.commit()

    login2 = client.post('/api/auth/login', json={'email': 'stu@example.com', 'password': 'Password1!'})
    tok2 = login2.get_json()['token']
    res2 = client.get('/api/teacher/reports', headers={'Authorization': f'Bearer {tok2}'})
    # teacher endpoint should now be accessible (may return empty list)
    assert res2.status_code == 200 or res2.status_code == 204 or res2.status_code == 200


def test_signup_allows_admin_role(client):
    response = client.post(
        '/api/auth/signup',
        json={
            'name': 'Admin Seed',
            'email': 'admin.seed@example.com',
            'password': 'Password1!',
            'role': 'admin',
            'school_name': 'North Ridge Academy',
            'school_slug': 'north-ridge-academy',
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload['user']['role'] == 'admin'
    assert payload['user']['school_slug'] == 'north-ridge-academy'

    with client.application.app_context():
        user = User.query.filter_by(email='admin.seed@example.com').first()
        school = db.session.get(School, user.school_id)
        assert user is not None
        assert user.grade is None
        assert user.school_id is not None
        assert school is not None
        assert school.name == 'North Ridge Academy'


def test_signup_admin_requires_school_name(client):
    response = client.post(
        '/api/auth/signup',
        json={
            'name': 'No School Admin',
            'email': 'no-school-admin@example.com',
            'password': 'Password1!',
            'role': 'admin',
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert 'School name is required' in payload.get('error', '')


def test_signup_admin_requires_school_slug(client):
    response = client.post(
        '/api/auth/signup',
        json={
            'name': 'No Slug Admin',
            'email': 'no-slug-admin@example.com',
            'password': 'Password1!',
            'role': 'admin',
            'school_name': 'North Ridge Academy',
        },
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert 'School slug is required' in payload.get('error', '')


def test_student_login_uses_slug_hint_cookie_when_school_missing(client):
    signup_response = client.post(
        '/api/auth/signup',
        json={
            'name': 'Slug Hint Student',
            'email': 'slug-hint-student@example.com',
            'password': 'Password1!',
            'role': 'student',
            'grade': 'middle',
        },
    )
    assert signup_response.status_code == 201

    # Simulate entering via slug route so backend sets cookie hint.
    slug_entry = client.get('/nhitm/', follow_redirects=False)
    assert slug_entry.status_code in (301, 302)

    login_response = client.post(
        '/api/auth/login',
        json={'email': 'slug-hint-student@example.com', 'password': 'Password1!'},
    )
    assert login_response.status_code == 200
    payload = login_response.get_json()
    assert payload.get('user', {}).get('school_slug') == 'nhitm'


def test_login_normalizes_school_name_when_school_slug_is_missing(client):
    with client.application.app_context():
        school = School(name='Legacy Ridge Academy', slug=None)
        db.session.add(school)
        db.session.flush()

        student = User(
            name='Legacy Student',
            email='legacy-student@example.com',
            password_hash=hash_password('Password1!'),
            role='student',
            grade='middle',
            school_id=school.id,
            is_verified=False,
        )
        db.session.add(student)
        db.session.commit()

    login_response = client.post(
        '/api/auth/login',
        json={'email': 'legacy-student@example.com', 'password': 'Password1!'},
    )
    assert login_response.status_code == 200
    payload = login_response.get_json()
    assert payload.get('user', {}).get('school_slug') == 'legacy-ridge-academy'


def test_logout_clears_slug_hint_cookie(client):
    signup_response = client.post(
        '/api/auth/signup',
        json={
            'name': 'Logout Student',
            'email': 'logout-student@example.com',
            'password': 'Password1!',
            'role': 'student',
            'grade': 'middle',
        },
    )
    assert signup_response.status_code == 201

    login_response = client.post(
        '/api/auth/login',
        json={'email': 'logout-student@example.com', 'password': 'Password1!'},
    )
    token = login_response.get_json().get('token')

    logout_response = client.post(
        '/api/auth/logout',
        headers={
            'Authorization': f'Bearer {token}',
            'Cookie': 'elevate_school_slug_hint=nhitm',
        },
    )
    assert logout_response.status_code == 200
    set_cookie_headers = logout_response.headers.getlist('Set-Cookie')
    assert any('elevate_school_slug_hint=' in header and 'expires=' in header.lower() for header in set_cookie_headers)
