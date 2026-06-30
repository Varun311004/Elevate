from backend.app import create_app
from backend.models import db, School, User
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

ADMIN_TOKEN = 'dev-admin-token'


def test_create_list_delete_school(client):
    # create
    r = client.post('/api/admin/schools', json={'name': 'Test School', 'slug': 'test-school'}, headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r.status_code == 201
    data = r.get_json()
    sid = data['school']['id']

    # list
    r2 = client.get('/api/admin/schools', headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r2.status_code == 200
    items = r2.get_json()['items']
    assert any(s['name'] == 'Test School' for s in items)

    # delete
    r3 = client.delete(f'/api/admin/schools/{sid}', headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r3.status_code == 200

    r4 = client.get('/api/admin/schools', headers={'X-Admin-Token': ADMIN_TOKEN})
    assert all(s['id'] != sid for s in r4.get_json()['items'])


def test_create_school_requires_slug(client):
    r = client.post('/api/admin/schools', json={'name': 'Slugless School'}, headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r.status_code == 400
    data = r.get_json()
    assert 'slug required' in str(data.get('error', '')).lower()
