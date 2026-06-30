from datetime import datetime, timezone
from backend.app import create_app
from backend.models import (
    db,
    User,
    TestResult,
    School,
    Question,
    AnswerLog,
    TeacherIntervention,
    TestAssignment,
)
import backend.routes.teacher as teacher_routes
import pytest


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

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


def test_teacher_sees_only_own_school_reports(client):
    with client.application.app_context():
        # create school
        s1 = School(name='Sunrise High', slug='sunrise')
        s2 = School(name='Other School', slug='other')
        db.session.add_all([s1, s2])
        db.session.commit()

        # create teacher via signup endpoint then update role and school
        signup = client.post('/api/auth/signup', json={'name': 'Ms Teacher', 'email': 'teacher@example.com', 'password': 'Password1!'})
        assert signup.status_code == 201
        # set role to teacher and assign school
        teacher = User.query.filter_by(email='teacher@example.com').first()
        teacher.role = 'teacher'
        teacher.school_id = s1.id
        db.session.commit()

        # create students in both schools
        st1 = User(name='Student One', email='s1@example.com', password_hash='x', role='student', school_id=s1.id, is_verified=True)
        st2 = User(name='Student Two', email='s2@example.com', password_hash='x', role='student', school_id=s1.id, is_verified=True)
        st3 = User(name='Other Student', email='o1@example.com', password_hash='x', role='student', school_id=s2.id, is_verified=True)
        db.session.add_all([st1, st2, st3])
        db.session.commit()

        # create test results
        tr1 = TestResult(user_id=st1.id, subject='Math', total_questions=10, correct_answers=8, average_time_per_question=5, started_at=_utcnow())
        tr2 = TestResult(user_id=st2.id, subject='Science', total_questions=8, correct_answers=6, average_time_per_question=6, started_at=_utcnow())
        tr3 = TestResult(user_id=st3.id, subject='Math', total_questions=10, correct_answers=2, average_time_per_question=10, started_at=_utcnow())
        db.session.add_all([tr1, tr2, tr3])
        db.session.commit()

    # login to get token
    login = client.post('/api/auth/login', json={'email': 'teacher@example.com', 'password': 'Password1!'})
    assert login.status_code == 200
    token = login.get_json()['token']

    # fetch reports as teacher
    res = client.get('/api/teacher/reports', headers={'Authorization': f'Bearer {token}'})
    assert res.status_code == 200
    items = res.get_json()['items']
    emails = [it['student_email'] for it in items]
    assert 's1@example.com' in emails
    assert 's2@example.com' in emails
    assert 'o1@example.com' not in emails


def test_teacher_create_classroom_auto_provisions_school(client):
    with client.application.app_context():
        signup = client.post('/api/auth/signup', json={
            'name': 'No School Teacher',
            'email': 'noschool@example.com',
            'password': 'Password1!'
        })
        assert signup.status_code == 201

        teacher = User.query.filter_by(email='noschool@example.com').first()
        teacher.role = 'teacher'
        teacher.school_id = None
        db.session.commit()

    login = client.post('/api/auth/login', json={'email': 'noschool@example.com', 'password': 'Password1!'})
    assert login.status_code == 200
    token = login.get_json()['token']

    create = client.post(
        '/api/teacher/classrooms',
        headers={'Authorization': f'Bearer {token}'},
        json={'name': 'Class 1', 'grade': 'high'},
    )
    assert create.status_code == 201, create.get_json()

    with client.application.app_context():
        refreshed = User.query.filter_by(email='noschool@example.com').first()
        assert refreshed.school_id is not None


@pytest.fixture
def teacher_scope(client):
    with client.application.app_context():
        school = School(name='Insight School', slug='insight-school')
        db.session.add(school)
        db.session.commit()

    signup = client.post('/api/auth/signup', json={
        'name': 'Teacher Prime',
        'email': 'teacher-prime@example.com',
        'password': 'Password1!',
    })
    assert signup.status_code == 201

    with client.application.app_context():
        teacher = User.query.filter_by(email='teacher-prime@example.com').first()
        teacher.role = 'teacher'
        teacher.school_id = School.query.filter_by(slug='insight-school').first().id
        teacher.grade = 'high'

        students = [
            User(
                name='Student A',
                email='student-a@example.com',
                password_hash='x',
                role='student',
                school_id=teacher.school_id,
                grade='high',
                is_verified=True,
            ),
            User(
                name='Student B',
                email='student-b@example.com',
                password_hash='x',
                role='student',
                school_id=teacher.school_id,
                grade='high',
                is_verified=True,
            ),
            User(
                name='Student C',
                email='student-c@example.com',
                password_hash='x',
                role='student',
                school_id=teacher.school_id,
                grade='high',
                is_verified=True,
            ),
        ]
        db.session.add_all(students)
        db.session.commit()

        teacher_id = teacher.id
        school_id = teacher.school_id
        student_ids = [s.id for s in students]

    login = client.post('/api/auth/login', json={
        'email': 'teacher-prime@example.com',
        'password': 'Password1!',
    })
    assert login.status_code == 200
    token = login.get_json()['token']

    return {
        'token': token,
        'teacher_id': teacher_id,
        'school_id': school_id,
        'student_ids': student_ids,
    }


def _seed_student_performance(client, student_ids):
    with client.application.app_context():
        q_fraction = Question(
            subject='math',
            grade='high',
            difficulty='easy',
            text='1/2 + 1/4 = ?',
            options=['3/4', '2/6', '1/8', '4/4'],
            correct_index=0,
            explanation='Add denominators by LCM.',
            syllabus_topic='fractions',
            is_generated=True,
        )
        q_decimal = Question(
            subject='math',
            grade='high',
            difficulty='easy',
            text='0.2 + 0.5 = ?',
            options=['0.3', '0.7', '0.9', '0.6'],
            correct_index=1,
            explanation='Simple decimal addition.',
            syllabus_topic='decimals',
            is_generated=True,
        )
        db.session.add_all([q_fraction, q_decimal])
        db.session.flush()

        results = []
        score_pairs = [(2, 10), (4, 10), (8, 10)]
        for idx, student_id in enumerate(student_ids):
            correct, total = score_pairs[idx % len(score_pairs)]
            result = TestResult(
                user_id=student_id,
                subject='math',
                total_questions=total,
                correct_answers=correct,
                earned_points=correct,
                total_points=total,
                average_time_per_question=6,
                started_at=_utcnow(),
                status='completed',
            )
            db.session.add(result)
            db.session.flush()
            results.append(result)

        # Student A: 0/2 on fractions -> strong weakness
        db.session.add_all([
            AnswerLog(
                user_id=student_ids[0],
                question_id=q_fraction.id,
                selected_index=2,
                is_correct=False,
                time_spent=12,
                answered_at=_utcnow(),
                test_id=results[0].id,
            ),
            AnswerLog(
                user_id=student_ids[0],
                question_id=q_fraction.id,
                selected_index=3,
                is_correct=False,
                time_spent=10,
                answered_at=_utcnow(),
                test_id=results[0].id,
            ),
            # Student B: 1/2 on fractions
            AnswerLog(
                user_id=student_ids[1],
                question_id=q_fraction.id,
                selected_index=0,
                is_correct=True,
                time_spent=9,
                answered_at=_utcnow(),
                test_id=results[1].id,
            ),
            AnswerLog(
                user_id=student_ids[1],
                question_id=q_fraction.id,
                selected_index=2,
                is_correct=False,
                time_spent=11,
                answered_at=_utcnow(),
                test_id=results[1].id,
            ),
            # Student C: strong on decimals
            AnswerLog(
                user_id=student_ids[2],
                question_id=q_decimal.id,
                selected_index=1,
                is_correct=True,
                time_spent=6,
                answered_at=_utcnow(),
                test_id=results[2].id,
            ),
        ])
        db.session.commit()


@pytest.fixture
def stub_generation(monkeypatch):
    def _fake_pick_or_generate_questions(teacher_id, subject, grade, difficulty, topic, count, **kwargs):
        generated = []
        for idx in range(int(count)):
            q = Question(
                subject=subject,
                grade=grade,
                difficulty=difficulty,
                text=f'Generated question {idx + 1} on {topic or subject}',
                options=['A', 'B', 'C', 'D'],
                correct_index=0,
                explanation='Stub explanation',
                syllabus_topic=topic or 'general',
                is_generated=True,
                generated_by=teacher_id,
                generation_meta={'source': 'test_stub'},
            )
            db.session.add(q)
            generated.append(q)

        db.session.flush()
        return generated, {
            'requested_count': int(count),
            'generated_count': int(count),
            'service_error': None,
            'source_mode': 'test_stub',
        }

    monkeypatch.setattr(teacher_routes, '_pick_or_generate_questions', _fake_pick_or_generate_questions)


def test_teacher_intervention_crud(client, teacher_scope):
    token = teacher_scope['token']

    create_resp = client.post(
        '/api/teacher/interventions',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'title': 'Monitor fractions support',
            'notes': 'Start with 15-minute guided practice.',
            'status': 'planned',
            'subject': 'math',
            'topic': 'fractions',
            'student_ids': teacher_scope['student_ids'],
        },
    )
    assert create_resp.status_code == 201, create_resp.get_json()
    intervention_id = create_resp.get_json()['intervention']['id']

    list_resp = client.get('/api/teacher/interventions', headers={'Authorization': f'Bearer {token}'})
    assert list_resp.status_code == 200
    assert list_resp.get_json()['count'] >= 1

    patch_resp = client.patch(
        f'/api/teacher/interventions/{intervention_id}',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'status': 'in_progress',
            'notes': 'First check-in completed.',
        },
    )
    assert patch_resp.status_code == 200
    assert patch_resp.get_json()['intervention']['status'] == 'in_progress'


def test_teacher_action_assign_remedial_test(client, teacher_scope, stub_generation):
    token = teacher_scope['token']
    _seed_student_performance(client, teacher_scope['student_ids'])

    resp = client.post(
        '/api/teacher/interventions/actions/remedial-assignment',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'subject': 'math',
            'topic': 'fractions',
            'question_count': 5,
            'max_students': 2,
            'due_days': 5,
        },
    )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()
    assert body['action'] == 'assign_remedial_test'
    assert int(body['created_assignments']) >= 1

    with client.application.app_context():
        assert TeacherIntervention.query.filter_by(action_type='assign_remedial_test').count() == 1
        assert TestAssignment.query.count() >= 1


def test_teacher_action_create_focused_practice(client, teacher_scope, stub_generation):
    token = teacher_scope['token']
    _seed_student_performance(client, teacher_scope['student_ids'])

    resp = client.post(
        '/api/teacher/interventions/actions/focused-practice',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'subject': 'math',
            'question_count': 4,
            'max_students': 2,
            'low_accuracy_threshold': 70,
        },
    )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()
    assert body['action'] == 'create_focused_practice_set'
    assert int(body['target_student_count']) >= 1

    with client.application.app_context():
        assert TeacherIntervention.query.filter_by(action_type='create_focused_practice_set').count() == 1
        assert TestAssignment.query.count() >= 1


def test_teacher_action_schedule_follow_up_assignment(client, teacher_scope, stub_generation, monkeypatch):
    token = teacher_scope['token']
    _seed_student_performance(client, teacher_scope['student_ids'])

    def _fake_at_risk(student_ids, **kwargs):
        return {
            'at_risk_students': [
                {'student_id': student_ids[0], 'at_risk_probability': 0.91},
                {'student_id': student_ids[1], 'at_risk_probability': 0.73},
            ],
            'meta': {'source': 'test_stub'},
        }

    monkeypatch.setattr(teacher_routes, 'get_at_risk_predictions_for_students', _fake_at_risk)

    resp = client.post(
        '/api/teacher/interventions/actions/follow-up-assignment',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'subject': 'math',
            'question_count': 4,
            'max_students': 2,
            'due_days': 6,
            'at_risk_threshold': 0.5,
        },
    )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()
    assert body['action'] == 'schedule_follow_up_assignment'
    assert int(body['created_assignments']) >= 1

    with client.application.app_context():
        assert TeacherIntervention.query.filter_by(action_type='schedule_follow_up_assignment').count() == 1


def test_teacher_action_group_weakness_clusters(client, teacher_scope):
    token = teacher_scope['token']
    _seed_student_performance(client, teacher_scope['student_ids'])

    resp = client.post(
        '/api/teacher/interventions/actions/weakness-clusters',
        headers={'Authorization': f'Bearer {token}'},
        json={
            'subject': 'math',
            'min_attempts': 1,
        },
    )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()
    assert body['action'] == 'group_weakness_clusters'
    assert int(body['cluster_count']) >= 1

    with client.application.app_context():
        entry = TeacherIntervention.query.filter_by(action_type='group_weakness_clusters').first()
        assert entry is not None
        assert isinstance(entry.cluster_payload, list)
