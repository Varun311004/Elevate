from flask import Blueprint, jsonify, request, g
from datetime import datetime, timedelta, timezone
from sqlalchemy import or_, false
from ..models import (
    Test,
    TestResult,
    User,
    Question,
    TestQuestion,
    AnswerLog,
    EmotionLog,
    UserProgress,
    SubjectPerformance,
    ClassroomStudent,
    TestAssignment,
    db,
)
from ..security import require_auth, role_required

student_bp = Blueprint("student", __name__)

ASSIGNMENT_INTEGRITY_WINDOW_SECONDS = 180


def _utcnow():
    # Keep UTC semantics while preserving existing naive DB datetime behavior.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _active_assignment_for_student_test(student_id, test_id):
    classroom_ids = [m.classroom_id for m in ClassroomStudent.query.filter_by(student_id=student_id, is_active=True).all()]

    classroom_filter = TestAssignment.classroom_id.in_(classroom_ids) if classroom_ids else false()

    q = TestAssignment.query.filter(
        TestAssignment.test_id == test_id,
        TestAssignment.status.in_(['assigned', 'started', 'submitted']),
    ).filter(
        or_(
            TestAssignment.student_id == student_id,
            classroom_filter,
        )
    ).order_by(TestAssignment.created_at.desc())

    return q.first()


def _is_assignment_expired(assignment):
    if not assignment or not assignment.due_at:
        return False
    if assignment.allow_late:
        return False
    return assignment.due_at < _utcnow()


def _enforce_assignment_integrity(student_id, assignment, *, phase: str):
    if not assignment:
        return None

    require_camera = bool(getattr(assignment, "require_camera", True))
    require_emotion = bool(getattr(assignment, "require_emotion", True))
    if not require_camera and not require_emotion:
        return None

    now = _utcnow()
    cutoff = now - timedelta(seconds=ASSIGNMENT_INTEGRITY_WINDOW_SECONDS)
    recent_log = EmotionLog.query.filter(
        EmotionLog.user_id == int(student_id),
        EmotionLog.timestamp >= cutoff,
    ).order_by(EmotionLog.timestamp.desc()).first()

    if not recent_log:
        return jsonify({
            "error": "Camera and emotion tracking must be active for assigned tests",
            "code": "assignment_integrity_required",
            "phase": phase,
            "require_camera": require_camera,
            "require_emotion": require_emotion,
            "window_seconds": ASSIGNMENT_INTEGRITY_WINDOW_SECONDS,
        }), 403

    return None


@student_bp.get('/dashboard')
@require_auth
@role_required('student')
def student_dashboard():
    """Get student dashboard data with progress and upcoming tests."""
    student = g.current_user
    
    # Get available tests (published tests for student's grade)
    available_tests = Test.query.filter(
        Test.grade == student.grade,
        Test.is_published == True,
        Test.is_active == True
    ).filter(
        (Test.expires_at.is_(None)) | (Test.expires_at > _utcnow())
    ).all()
    
    # Get recent test results
    recent_results = TestResult.query.filter_by(user_id=student.id).order_by(
        TestResult.started_at.desc()
    ).limit(5).all()
    
    # Get subject progress
    subject_progress = UserProgress.query.filter_by(user_id=student.id).all()
    
    # Get subject performance
    subject_performance = SubjectPerformance.query.filter_by(user_id=student.id).all()
    
    # Calculate overall stats
    total_tests = TestResult.query.filter_by(user_id=student.id).count()
    completed_tests = TestResult.query.filter_by(user_id=student.id, status='completed').count()
    
    avg_score = 0
    if completed_tests > 0:
        scores = TestResult.query.filter_by(user_id=student.id, status='completed').all()
        if scores:
            avg_score = sum((r.correct_answers / r.total_questions * 100) for r in scores if r.total_questions > 0) / len(scores)
    
    return jsonify({
        'total_tests': total_tests,
        'completed_tests': completed_tests,
        'average_score': round(avg_score, 2),
        'available_tests': [test.as_dict() for test in available_tests],
        'recent_results': [result.as_dict() for result in recent_results],
        'subject_progress': [
            {
                'subject': sp.subject,
                'total_questions': sp.total_questions,
                'correct_answers': sp.correct_answers,
                'accuracy': round((sp.correct_answers / sp.total_questions * 100) if sp.total_questions > 0 else 0, 2),
                'current_difficulty': sp.current_difficulty
            } for sp in subject_progress
        ],
        'subject_performance': [
            {
                'subject': perf.subject,
                'accuracy': round(perf.accuracy, 2),
                'streak': perf.streak,
                'best_streak': perf.best_streak,
                'total_time_spent': perf.total_time_spent
            } for perf in subject_performance
        ]
    })


@student_bp.get('/tests')
@require_auth
@role_required('student')
def get_available_tests():
    """Get available tests for the student."""
    student = g.current_user
    
    # Get published tests for student's grade level
    tests = Test.query.filter(
        Test.grade == student.grade,
        Test.is_published == True,
        Test.is_active == True
    ).filter(
        (Test.expires_at.is_(None)) | (Test.expires_at > _utcnow())
    ).order_by(Test.created_at.desc()).all()
    
    # Check if student has already taken each test
    test_data = []
    for test in tests:
        existing_result = TestResult.query.filter_by(
            user_id=student.id,
            test_id=test.id,
            status='completed'
        ).first()

        assignment = _active_assignment_for_student_test(student.id, test.id)
        
        test_info = test.as_dict()
        test_info['already_taken'] = existing_result is not None
        test_info['assignment'] = assignment.as_dict() if assignment else None
        if existing_result:
            test_info['previous_score'] = round((existing_result.correct_answers / existing_result.total_questions * 100) if existing_result.total_questions > 0 else 0, 2)
        
        test_data.append(test_info)
    
    return jsonify({
        'tests': test_data
    })


@student_bp.get('/assigned-tests')
@require_auth
@role_required('student')
def get_assigned_tests():
    """Get tests assigned directly or via classroom with lifecycle metadata."""
    student = g.current_user

    memberships = ClassroomStudent.query.filter_by(student_id=student.id, is_active=True).all()
    classroom_ids = [m.classroom_id for m in memberships]

    classroom_filter = TestAssignment.classroom_id.in_(classroom_ids) if classroom_ids else false()

    assignments = TestAssignment.query.filter(
        or_(
            TestAssignment.student_id == student.id,
            classroom_filter,
        )
    ).order_by(TestAssignment.created_at.desc()).all()

    test_ids = {assignment.test_id for assignment in assignments}
    tests = {
        test.id: test
        for test in Test.query.filter(Test.id.in_(test_ids)).all()
    } if test_ids else {}

    payload = []
    for assignment in assignments:
        test = tests.get(assignment.test_id)
        if not test:
            continue

        completed = TestResult.query.filter_by(user_id=student.id, test_id=test.id, status='completed').first()

        item = assignment.as_dict()
        item['test'] = test.as_dict()
        item['already_taken'] = completed is not None
        item['is_overdue'] = _is_assignment_expired(assignment)
        item['previous_score'] = round((completed.correct_answers / completed.total_questions * 100), 2) if completed and completed.total_questions else None
        payload.append(item)

    return jsonify({'assignments': payload})


@student_bp.post('/tests/<int:test_id>/start')
@require_auth
@role_required('student')
def start_test(test_id):
    """Start a test for the student."""
    student = g.current_user

    assignment = _active_assignment_for_student_test(student.id, test_id)

    # Assigned tests can be started even if grade/published filters do not match,
    # but the test itself must still be active.
    test = Test.query.filter(
        Test.id == test_id,
        Test.is_active == True
    ).first()

    if not test:
        return jsonify({"error": "Test not found or not available"}), 404

    if not assignment:
        if test.grade != student.grade or not test.is_published:
            return jsonify({"error": "Test not found or not available"}), 404
    
    # Check if test has expired
    if test.expires_at and test.expires_at < _utcnow():
        return jsonify({"error": "Test has expired"}), 400

    if assignment and _is_assignment_expired(assignment):
        assignment.status = 'expired'
        db.session.commit()
        return jsonify({"error": "Assigned test is overdue"}), 400

    integrity_error = _enforce_assignment_integrity(student.id, assignment, phase="start")
    if integrity_error:
        return integrity_error
    
    # Check if student has already completed this test
    existing_result = TestResult.query.filter_by(
        user_id=student.id,
        test_id=test_id,
        status='completed'
    ).first()
    
    if existing_result:
        return jsonify({"error": "You have already completed this test"}), 400
    
    # Check if there's an in-progress test
    in_progress_result = TestResult.query.filter_by(
        user_id=student.id,
        test_id=test_id,
        status='in_progress'
    ).first()
    
    if in_progress_result:
        # Check if the time limit has expired
        time_elapsed = _utcnow() - in_progress_result.started_at
        if time_elapsed.total_seconds() > (test.time_limit * 60):
            in_progress_result.status = 'expired'
            db.session.commit()
            return jsonify({"error": "Previous attempt expired. Please start a new test."}), 400
        
        # Return existing in-progress test
        return jsonify({
            "message": "Test already in progress",
            "test_result": in_progress_result.as_dict()
        })
    
    try:
        # Create new test result
        test_result = TestResult(
            user_id=student.id,
            test_id=test_id,
            subject=test.subject,
            total_questions=test.question_count,
            correct_answers=0,
            total_points=test.total_points,
            status='in_progress'
        )
        
        db.session.add(test_result)

        if assignment and assignment.status == 'assigned':
            assignment.status = 'started'
            assignment.started_at = _utcnow()

        db.session.commit()
        
        return jsonify({
            "message": "Test started successfully",
            "test_result": test_result.as_dict(),
            "time_limit": test.time_limit,
            "assignment_policy": {
                "require_camera": bool(getattr(assignment, "require_camera", False)) if assignment else False,
                "require_emotion": bool(getattr(assignment, "require_emotion", False)) if assignment else False,
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to start test: {str(e)}"}), 500


@student_bp.get('/tests/<int:test_id>/questions')
@require_auth
@role_required('student')
def get_test_questions(test_id):
    """Get questions for a specific test attempt."""
    student = g.current_user
    
    # Get the test result in progress
    test_result = TestResult.query.filter_by(
        user_id=student.id,
        test_id=test_id,
        status='in_progress'
    ).first()
    
    if not test_result:
        return jsonify({"error": "No active test found"}), 404
    
    # Check if time limit has expired
    test = db.session.get(Test, test_id)
    time_elapsed = _utcnow() - test_result.started_at
    if time_elapsed.total_seconds() > (test.time_limit * 60):
        test_result.status = 'expired'
        db.session.commit()
        return jsonify({"error": "Test time limit expired"}), 400
    
    # Get test questions
    test_questions = TestQuestion.query.filter_by(test_id=test_id).order_by(TestQuestion.order).all()
    
    questions_data = []
    for tq in test_questions:
        question = db.session.get(Question, tq.question_id)
        if question:
            questions_data.append({
                'id': question.id,
                'text': question.text,
                'options': question.options,
                'order': tq.order,
                'points': tq.points,
                'time_limit': test.time_limit * 60,  # Convert to seconds
                'time_remaining': max(0, (test.time_limit * 60) - int(time_elapsed.total_seconds()))
            })
    
    return jsonify({
        'questions': questions_data,
        'test_result': test_result.as_dict(),
        'time_remaining': max(0, (test.time_limit * 60) - int(time_elapsed.total_seconds()))
    })


@student_bp.post('/tests/<int:test_id>/answer')
@require_auth
@role_required('student')
def submit_answer(test_id):
    """Submit an answer for a test question."""
    student = g.current_user
    data = request.get_json(silent=True) or {}
    
    question_id = data.get('question_id')
    selected_index = data.get('selected_index')
    time_spent = data.get('time_spent', 0)

    if not question_id or selected_index is None:
        return jsonify({"error": "Missing question_id or selected_index"}), 400

    try:
        question_id = int(question_id)
        selected_index = int(selected_index)
        time_spent = int(time_spent or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid answer payload"}), 400
    
    # Get the active test result
    test_result = TestResult.query.filter_by(
        user_id=student.id,
        test_id=test_id,
        status='in_progress'
    ).first()
    
    if not test_result:
        return jsonify({"error": "No active test found"}), 404
    
    # Check if time limit has expired
    test = db.session.get(Test, test_id)
    time_elapsed = _utcnow() - test_result.started_at
    if time_elapsed.total_seconds() > (test.time_limit * 60):
        test_result.status = 'expired'
        db.session.commit()
        return jsonify({"error": "Test time limit expired"}), 400

    assignment = _active_assignment_for_student_test(student.id, test_id)
    integrity_error = _enforce_assignment_integrity(student.id, assignment, phase="answer")
    if integrity_error:
        return integrity_error
    
    # Get the question
    question = db.session.get(Question, question_id)
    if not question:
        return jsonify({"error": "Question not found"}), 404

    # Ensure the submitted question belongs to this test.
    test_question = TestQuestion.query.filter_by(test_id=test_id, question_id=question_id).first()
    if not test_question:
        return jsonify({"error": "Question does not belong to this test"}), 400

    if selected_index < -1 or selected_index >= len(question.options or []):
        return jsonify({"error": "Invalid option index"}), 400

    is_correct = selected_index >= 0 and selected_index == question.correct_index
    
    # Check if answer already exists for this question
    existing_answer = AnswerLog.query.filter_by(
        user_id=student.id,
        question_id=question_id,
        test_id=test_result.id
    ).first()
    
    if existing_answer:
        # Update existing answer
        existing_answer.selected_index = selected_index
        existing_answer.is_correct = is_correct
        existing_answer.time_spent = time_spent
        existing_answer.answered_at = _utcnow()
    else:
        # Create new answer
        answer = AnswerLog(
            user_id=student.id,
            question_id=question_id,
            test_id=test_result.id,
            selected_index=selected_index,
            is_correct=is_correct,
            time_spent=time_spent,
            difficulty_at_time=question.difficulty
        )
        db.session.add(answer)
    
    try:
        db.session.commit()
        return jsonify({
            "message": "Answer submitted successfully",
            "is_correct": is_correct,
            "correct_index": question.correct_index,
            "explanation": question.explanation,
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to submit answer: {str(e)}"}), 500


@student_bp.post('/tests/<int:test_id>/finish')
@require_auth
@role_required('student')
def finish_test(test_id):
    """Finish a test and calculate the final score."""
    student = g.current_user
    
    # Get the active test result
    test_result = TestResult.query.filter_by(
        user_id=student.id,
        test_id=test_id,
        status='in_progress'
    ).first()
    
    if not test_result:
        return jsonify({"error": "No active test found"}), 404

    assignment = _active_assignment_for_student_test(student.id, test_id)
    integrity_error = _enforce_assignment_integrity(student.id, assignment, phase="finish")
    if integrity_error:
        return integrity_error
    
    try:
        # Get all answers for this test
        answers = AnswerLog.query.filter_by(
            user_id=student.id,
            test_id=test_result.id
        ).all()
        
        # Calculate scores
        correct_count = sum(1 for answer in answers if answer.is_correct)
        total_time = sum(int(answer.time_spent or 0) for answer in answers)
        avg_time = total_time / len(answers) if answers else 0
        
        # Calculate earned points
        test = db.session.get(Test, test_id)
        total_possible_points = test.total_points
        points_per_question = (total_possible_points / test.question_count) if test.question_count else 0
        earned_points = int(correct_count * points_per_question)
        
        # Update test result
        test_result.correct_answers = correct_count
        test_result.earned_points = earned_points
        test_result.average_time_per_question = avg_time
        test_result.finished_at = _utcnow()
        test_result.status = 'completed'

        if assignment:
            assignment.status = 'submitted'
            assignment.submitted_at = _utcnow()
        
        # Update user progress
        for subject in set([answer.question.subject for answer in answers if answer.question]):
            progress = UserProgress.query.filter_by(
                user_id=student.id,
                subject=subject
            ).first()
            
            if not progress:
                progress = UserProgress(
                    user_id=student.id,
                    subject=subject,
                    total_questions=0,
                    correct_answers=0
                )
                db.session.add(progress)

            # Defensive normalization for older rows with nullable counters.
            progress.total_questions = int(progress.total_questions or 0)
            progress.correct_answers = int(progress.correct_answers or 0)
            
            # Update progress
            subject_answers = [a for a in answers if a.question and a.question.subject == subject]
            progress.total_questions = int(progress.total_questions or 0) + len(subject_answers)
            progress.correct_answers = int(progress.correct_answers or 0) + sum(1 for a in subject_answers if a.is_correct)
            progress.last_updated = _utcnow()
        
        # Update subject performance
        subject_perf = SubjectPerformance.query.filter_by(
            user_id=student.id,
            subject=test.subject
        ).first()
        
        if not subject_perf:
            subject_perf = SubjectPerformance(
                user_id=student.id,
                subject=test.subject
            )
            db.session.add(subject_perf)

        # Defensive normalization for legacy nullable performance fields.
        subject_perf.total_time_spent = int(subject_perf.total_time_spent or 0)
        subject_perf.streak = int(subject_perf.streak or 0)
        subject_perf.best_streak = int(subject_perf.best_streak or 0)
        
        # Update performance metrics
        accuracy = (correct_count / len(answers) * 100) if answers else 0
        subject_perf.accuracy = accuracy
        subject_perf.total_time_spent = int(subject_perf.total_time_spent or 0) + int(total_time)
        subject_perf.last_practiced_at = _utcnow()
        
        # Update streak (simplified logic)
        if accuracy >= 70:  # Good performance threshold
            subject_perf.streak = int(subject_perf.streak or 0) + 1
            if subject_perf.streak > subject_perf.best_streak:
                subject_perf.best_streak = subject_perf.streak
        else:
            subject_perf.streak = 0
        
        db.session.commit()
        
        return jsonify({
            "message": "Test completed successfully",
            "test_result": test_result.as_dict(),
            "score": round((correct_count / len(answers) * 100) if answers else 0, 2),
            "correct_answers": correct_count,
            "total_questions": len(answers),
            "earned_points": earned_points
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to finish test: {str(e)}"}), 500


@student_bp.get('/results')
@require_auth
@role_required('student')
def get_test_results():
    """Get all test results for the student."""
    student = g.current_user
    
    results = TestResult.query.filter_by(user_id=student.id).order_by(
        TestResult.started_at.desc()
    ).all()
    
    results_data = []
    for result in results:
        test = db.session.get(Test, result.test_id) if result.test_id else None
        
        result_info = result.as_dict()
        if test:
            result_info['test_title'] = test.title
            result_info['test_subject'] = test.subject
        
        results_data.append(result_info)
    
    return jsonify({
        'results': results_data
    })


@student_bp.get('/results/<int:result_id>')
@require_auth
@role_required('student')
def get_test_result_detail(result_id):
    """Get detailed results for a specific test."""
    student = g.current_user
    
    result = TestResult.query.filter_by(id=result_id, user_id=student.id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404
    
    # Get answers for this test
    answers = AnswerLog.query.filter_by(test_id=result_id).all()
    
    answers_data = []
    for answer in answers:
        question = db.session.get(Question, answer.question_id)
        if question:
            answers_data.append({
                'question_text': question.text,
                'question_options': question.options,
                'selected_index': answer.selected_index,
                'correct_index': question.correct_index,
                'is_correct': answer.is_correct,
                'time_spent': answer.time_spent,
                'explanation': question.explanation,
                'hint': question.hint
            })
    
    test = db.session.get(Test, result.test_id) if result.test_id else None
    
    return jsonify({
        'result': result.as_dict(),
        'test': test.as_dict() if test else None,
        'answers': answers_data
    })


@student_bp.get('/progress')
@require_auth
@role_required('student')
def get_student_progress():
    """Get detailed progress data for the student."""
    student = g.current_user
    
    # Get user progress
    progress = UserProgress.query.filter_by(user_id=student.id).all()
    
    # Get subject performance
    performance = SubjectPerformance.query.filter_by(user_id=student.id).all()
    
    # Get recent activity
    recent_answers = AnswerLog.query.filter_by(user_id=student.id).order_by(
        AnswerLog.answered_at.desc()
    ).limit(20).all()
    
    recent_activity = []
    for answer in recent_answers:
        question = db.session.get(Question, answer.question_id)
        if question:
            recent_activity.append({
                'question_text': question.text[:100] + '...' if len(question.text) > 100 else question.text,
                'subject': question.subject,
                'difficulty': question.difficulty,
                'is_correct': answer.is_correct,
                'answered_at': answer.answered_at.isoformat()
            })
    
    return jsonify({
        'progress': [
            {
                'subject': p.subject,
                'total_questions': p.total_questions,
                'correct_answers': p.correct_answers,
                'accuracy': round((p.correct_answers / p.total_questions * 100) if p.total_questions > 0 else 0, 2),
                'current_difficulty': p.current_difficulty,
                'last_updated': p.last_updated.isoformat() if p.last_updated else None
            } for p in progress
        ],
        'performance': [
            {
                'subject': perf.subject,
                'accuracy': round(perf.accuracy, 2),
                'streak': perf.streak,
                'best_streak': perf.best_streak,
                'total_time_spent': perf.total_time_spent,
                'last_practiced_at': perf.last_practiced_at.isoformat() if perf.last_practiced_at else None
            } for perf in performance
        ],
        'recent_activity': recent_activity
    })
