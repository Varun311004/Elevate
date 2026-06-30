"""Progress routes for tracking user learning progress."""
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, g
from sqlalchemy import func
from ..models import db, UserProgress, SubjectPerformance, AnswerLog
from ..security import require_auth

progress_bp = Blueprint("progress", __name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@progress_bp.get("/")
@require_auth
def get_progress():
    """
    Get user's overall progress across all subjects.
    
    Returns:
        JSON with summary stats and per-subject breakdown
    """
    from flask import g
    current_user = g.current_user
    try:
        # Get all user progress records
        progress_records = UserProgress.query.filter_by(user_id=current_user.id).all()
        
        # Calculate overall statistics
        total_questions = sum(p.total_questions for p in progress_records)
        total_correct = sum(p.correct_answers for p in progress_records)
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # Calculate streak (consecutive days with activity)
        today = _utcnow().date()
        streak = 0
        check_date = today
        
        while True:
            # Check if user answered questions on this date
            start_of_day = datetime.combine(check_date, datetime.min.time())
            end_of_day = datetime.combine(check_date, datetime.max.time())
            
            has_activity = AnswerLog.query.filter(
                AnswerLog.user_id == current_user.id,
                AnswerLog.answered_at >= start_of_day,
                AnswerLog.answered_at <= end_of_day
            ).first() is not None
            
            if has_activity:
                streak += 1
                check_date = check_date - timedelta(days=1)
            else:
                # If today has no activity, streak is 0
                if check_date == today:
                    streak = 0
                break
        
        # Determine level based on total questions
        if total_questions < 10:
            level = "Beginner"
        elif total_questions < 50:
            level = "Intermediate"
        elif total_questions < 100:
            level = "Advanced"
        else:
            level = "Expert"
        
        # Format subject-specific progress
        subjects = [
            {
                "subject": p.subject,
                "total_questions": p.total_questions,
                "correct_answers": p.correct_answers,
                "accuracy": round((p.correct_answers / p.total_questions * 100) if p.total_questions > 0 else 0, 2),
                "current_difficulty": p.current_difficulty,
                "last_updated": p.last_updated.isoformat() if p.last_updated else None
            }
            for p in progress_records
        ]
        
        return jsonify({
            "summary": {
                "totalQuestions": total_questions,
                "totalCorrect": total_correct,
                "accuracy": round(overall_accuracy, 2),
                "streak": streak,
                "level": level
            },
            "subjects": subjects
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch progress", "details": str(e)}), 500


@progress_bp.get("/<subject>")
@require_auth
def get_subject_progress(subject):
    """
    Get detailed progress for a specific subject.
    
    Args:
        subject: Subject name (e.g., "mathematics", "science")
        
    Returns:
        JSON with detailed subject progress including recent answers
    """
    from flask import g
    current_user = g.current_user
    try:
        # Get progress record
        progress = UserProgress.query.filter_by(
            user_id=current_user.id,
            subject=subject
        ).first()
        
        if not progress:
            return jsonify({
                "subject": subject,
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0,
                "current_difficulty": "medium",
                "recent_answers": []
            }), 200
        
        # Get subject performance if available
        performance = SubjectPerformance.query.filter_by(
            user_id=current_user.id,
            subject=subject
        ).first()
        
        # Get recent answers for this subject (last 10)
        recent_answers = db.session.query(AnswerLog).join(
            AnswerLog.question
        ).filter(
            AnswerLog.user_id == current_user.id,
            db.text("questions.subject = :subject")
        ).params(subject=subject).order_by(
            AnswerLog.answered_at.desc()
        ).limit(10).all()
        
        recent_answers_data = [
            {
                "question_id": a.question_id,
                "is_correct": a.is_correct,
                "time_spent": a.time_spent,
                "difficulty": a.difficulty_at_time,
                "emotion": a.emotion_at_time,
                "answered_at": a.answered_at.isoformat()
            }
            for a in recent_answers
        ]
        
        accuracy = (progress.correct_answers / progress.total_questions * 100) if progress.total_questions > 0 else 0
        
        response = {
            "subject": subject,
            "total_questions": progress.total_questions,
            "correct_answers": progress.correct_answers,
            "accuracy": round(accuracy, 2),
            "current_difficulty": progress.current_difficulty,
            "last_updated": progress.last_updated.isoformat() if progress.last_updated else None,
            "recent_answers": recent_answers_data
        }
        
        # Add performance data if available
        if performance:
            response["performance"] = {
                "streak": performance.streak,
                "best_streak": performance.best_streak,
                "total_time_spent": performance.total_time_spent,
                "last_practiced_at": performance.last_practiced_at.isoformat() if performance.last_practiced_at else None
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch subject progress", "details": str(e)}), 500


@progress_bp.get("/stats/dashboard")
@require_auth
def get_dashboard_stats():
    """
    Get dashboard statistics for quick overview.
    
    Returns:
        JSON with key metrics for dashboard display
    """
    from flask import g
    current_user = g.current_user
    try:
        # Total questions answered
        total_questions = db.session.query(func.count(AnswerLog.id)).filter(
            AnswerLog.user_id == current_user.id
        ).scalar() or 0
        
        # Total correct answers
        total_correct = db.session.query(func.count(AnswerLog.id)).filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.is_correct == True
        ).scalar() or 0
        
        # Overall accuracy
        accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # Calculate question streak (consecutive correct answers)
        # Get most recent answers ordered by time
        recent_answers = AnswerLog.query.filter_by(
            user_id=current_user.id
        ).order_by(AnswerLog.answered_at.desc()).limit(100).all()
        
        # Count consecutive correct from most recent
        streak = 0
        for answer in recent_answers:
            if answer.is_correct:
                streak += 1
            else:
                break  # Streak ends at first wrong answer
        
        # Determine level
        if total_questions < 10:
            level = "Beginner"
        elif total_questions < 50:
            level = "Intermediate"
        elif total_questions < 100:
            level = "Advanced"
        else:
            level = "Expert"
        
        # Get subject breakdown
        subjects = db.session.query(
            UserProgress.subject,
            UserProgress.total_questions,
            UserProgress.correct_answers
        ).filter(
            UserProgress.user_id == current_user.id
        ).all()
        
        subject_stats = [
            {
                "subject": s[0],
                "questions": s[1],
                "accuracy": round((s[2] / s[1] * 100) if s[1] > 0 else 0, 2)
            }
            for s in subjects
        ]
        
        return jsonify({
            "totalQuestions": total_questions,
            "accuracy": round(accuracy, 2),
            "streak": streak,
            "level": level,
            "subjects": subject_stats
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch dashboard stats", "details": str(e)}), 500


