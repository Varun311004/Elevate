import pytest
from datetime import datetime, timezone
from backend.app import create_app
from backend.models import db, User, TestResult, AnswerLog, Question


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


def make_user(app, name, email):
    u = User(name=name, email=email, password_hash='x', is_verified=True)
    db.session.add(u)
    db.session.commit()
    return u


def make_test_result(user_id, subject, total, correct, started_at=None, finished_at=None):
    tr = TestResult(user_id=user_id, subject=subject, total_questions=total, correct_answers=correct, average_time_per_question=10, started_at=started_at, finished_at=finished_at)
    db.session.add(tr)
    db.session.commit()
    return tr


def make_answer(user_id, question_id, test_id, selected_index, is_correct, time_spent=5):
    a = AnswerLog(user_id=user_id, question_id=question_id, test_id=test_id, selected_index=selected_index, is_correct=is_correct, time_spent=time_spent, difficulty_at_time='medium')
    db.session.add(a)
    db.session.commit()
    return a

def test_list_test_results_and_history(client):
    with client.application.app_context():
        u1 = make_user(client.application, 'S1', 's1@example.com')
        u2 = make_user(client.application, 'S2', 's2@example.com')
        tr1 = make_test_result(u1.id, 'Math', 10, 8)
        tr2 = make_test_result(u1.id, 'Science', 8, 6)
        tr3 = make_test_result(u2.id, 'Math', 10, 5)

        # Create some answer logs tied to tests
        # ensure a question exists
        q = Question(subject='Math', grade='high', difficulty='medium', text='Q?', options=['A','B'], correct_index=0)
        db.session.add(q)
        db.session.commit()
        make_answer(u1.id, q.id, tr1.id, 0, True)
        make_answer(u1.id, q.id, tr2.id, 1, False)
        make_answer(u2.id, q.id, tr3.id, 0, True)
        # capture id before leaving app context
        u1_id = u1.id

    # List as admin
    res = client.get('/api/admin/test-results', headers={'X-Admin-Token': client.application.config.get('ADMIN_TOKEN')})
    assert res.status_code == 200
    data = res.get_json()
    assert 'items' in data
    assert any(item['user']['email'] == 's1@example.com' for item in data['items'])

    # Test CSV export for the same period
    # use start/end as today's date to include created rows
    today = datetime.now(timezone.utc).date().isoformat()
    res_csv = client.get(f'/api/admin/test-results?format=csv&start={today}&end={today}', headers={'X-Admin-Token': client.application.config.get('ADMIN_TOKEN')})
    assert res_csv.status_code == 200
    assert res_csv.content_type.startswith('text/csv')
    csv_text = res_csv.get_data(as_text=True)
    assert 'test_id' in csv_text
    assert 'avg_time_per_question' in csv_text
    assert '10' in csv_text
    assert 's1@example.com' in csv_text
    # History for u1
    user_id = u1_id

    res2 = client.get(f'/api/admin/test-results/{user_id}/history', headers={'X-Admin-Token': client.application.config.get('ADMIN_TOKEN')})
    assert res2.status_code == 200
    hist = res2.get_json()['items']
    assert len(hist) >= 2
    assert any(h['subject'] == 'Math' for h in hist)