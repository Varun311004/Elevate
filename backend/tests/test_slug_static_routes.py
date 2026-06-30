from backend.app import create_app
from backend.models import db
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


def test_slug_and_non_slug_static_routes_are_served(client):
    served_paths = [
        '/',
        '/index.html',
        '/dashboard.html',
        '/css/styles.css',
        '/js/main.js',
        '/nhitm/dashboard.html',
        '/nhitm/css/styles.css',
        '/nhitm/js/main.js',
    ]

    for path in served_paths:
        response = client.get(path)
        assert response.status_code == 200, f'Expected 200 for {path}, got {response.status_code}'

    canonical_redirect_paths = ['/nhitm/', '/nhitm/index.html']
    for path in canonical_redirect_paths:
        response = client.get(path)
        assert response.status_code in (301, 302), f'Expected redirect for {path}, got {response.status_code}'
        assert (response.headers.get('Location') or '').endswith('/index.html')
        set_cookie = response.headers.get('Set-Cookie', '')
        assert 'elevate_school_slug_hint=' in set_cookie


def test_slug_routes_do_not_capture_api_prefix(client):
    response = client.get('/api/not-a-real-endpoint')
    assert response.status_code == 404
