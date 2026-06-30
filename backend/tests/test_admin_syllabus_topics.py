from backend.app import create_app
from backend.models import db, SyllabusTopic

import json
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


def test_create_list_delete_topics(client):
    # create topic
    payload = {'subject':'science','grade':'middle','slug':'astronomy','title':'Astronomy','description':'Space topics'}
    r = client.post('/api/admin/syllabus-topics', json=payload, headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r.status_code == 201
    data = r.get_json()
    topic_id = data['topic']['id']

    # list topics
    r2 = client.get('/api/admin/syllabus-topics?subject=science&grade=middle', headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r2.status_code == 200
    items = r2.get_json()['items']
    assert any(t['slug']=='astronomy' for t in items)

    # delete topic
    r3 = client.delete(f'/api/admin/syllabus-topics/{topic_id}', headers={'X-Admin-Token': ADMIN_TOKEN})
    assert r3.status_code == 200

    # ensure deleted
    r4 = client.get('/api/admin/syllabus-topics?subject=science&grade=middle', headers={'X-Admin-Token': ADMIN_TOKEN})
    assert not any(t['slug']=='astronomy' for t in r4.get_json()['items'])
