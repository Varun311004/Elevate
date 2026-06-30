"""Reports routes for comprehensive analytics and insights."""
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, g
from sqlalchemy import func, desc
from ..models import db, UserProgress, AnswerLog, EmotionLog, SubjectPerformance, Question
from ..security import require_auth

reports_bp = Blueprint("reports", __name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_difficulty(value):
    """Normalize legacy/variant difficulty values to canonical buckets."""
    raw = (value or "").strip().lower()
    aliases = {
        "beginner": "easy",
        "basic": "easy",
        "normal": "medium",
        "intermediate": "medium",
        "advanced": "hard",
        "challenging": "hard",
        "pro": "hard",
        "extreme": "hard",
        "expert": "hard",
    }
    canonical = aliases.get(raw, raw)
    if canonical in {"easy", "medium", "hard"}:
        return canonical
    return "unknown"


def _resolve_timeline_window(raw_days, user_id):
    """Resolve timeline date window. Supports both numeric and `all`."""
    today = _utcnow().date()

    if str(raw_days).strip().lower() == "all":
        first_row = db.session.query(func.min(AnswerLog.answered_at)).filter(
            AnswerLog.user_id == user_id
        ).first()
        first_answered_at = first_row[0] if first_row else None
        if first_answered_at:
            start_date = first_answered_at.date()
        else:
            start_date = today
        resolved_days = (today - start_date).days + 1
        return start_date, today, max(1, resolved_days)

    try:
        days = int(raw_days)
    except (TypeError, ValueError):
        days = 30

    days = max(1, min(days, 3650))
    start_date = today - timedelta(days=days - 1)
    return start_date, today, days


def _build_timeline_payload(user_id, raw_days):
    """Build canonical timeline payload used by timeline and integrity endpoints."""
    start_date, today, days = _resolve_timeline_window(raw_days, user_id)

    day_buckets = {}
    for day_offset in range(days):
        bucket_date = start_date + timedelta(days=day_offset)
        day_buckets[bucket_date] = {
            "questions": 0,
            "correct": 0,
            "time_spent_seconds": 0,
        }

    difficulty_breakdown = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "unknown": 0,
    }

    def _to_date(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        # Defensive parse for drivers that may return serialized datetime strings.
        try:
            return datetime.fromisoformat(str(value)).date()
        except Exception:
            return None

    answer_rows = db.session.query(
        AnswerLog.answered_at,
        AnswerLog.is_correct,
        AnswerLog.time_spent,
        AnswerLog.difficulty_at_time,
        Question.difficulty,
    ).outerjoin(
        Question,
        AnswerLog.question_id == Question.id,
    ).filter(
        AnswerLog.user_id == user_id,
    ).all()

    for answered_at, is_correct, time_spent, difficulty_at_time, question_difficulty in answer_rows:
        answered_day = _to_date(answered_at)
        if not answered_day:
            continue
        if answered_day < start_date or answered_day > today:
            continue

        bucket = day_buckets.get(answered_day)
        if not bucket:
            continue

        bucket["questions"] += 1
        if bool(is_correct):
            bucket["correct"] += 1
        bucket["time_spent_seconds"] += int(time_spent or 0)

        normalized_difficulty = _normalize_difficulty(difficulty_at_time or question_difficulty)
        difficulty_breakdown[normalized_difficulty] += 1

    daily_stats = []
    labels = []
    correct_data = []
    incorrect_data = []

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        bucket = day_buckets[current_date]
        day_total = int(bucket["questions"])
        day_correct = int(bucket["correct"])
        day_incorrect = max(0, day_total - day_correct)
        day_accuracy = round((day_correct / day_total * 100) if day_total > 0 else 0, 2)
        day_time_minutes = round((bucket["time_spent_seconds"] or 0) / 60, 1)

        if days <= 7:
            label = current_date.strftime('%a %d')
        elif days <= 120:
            label = current_date.strftime('%b %d')
        else:
            label = current_date.strftime('%Y-%m-%d')

        labels.append(label)
        correct_data.append(day_correct)
        incorrect_data.append(day_incorrect)

        daily_stats.append({
            "date": current_date.isoformat(),
            "questions": day_total,
            "correct": day_correct,
            "accuracy": day_accuracy,
            "time_minutes": day_time_minutes,
        })

    non_zero_days = [d for d in daily_stats if d['questions'] > 0]
    if len(non_zero_days) >= 2:
        first_half = non_zero_days[:len(non_zero_days) // 2]
        second_half = non_zero_days[len(non_zero_days) // 2:]
        first_avg = sum(d['accuracy'] for d in first_half) / len(first_half)
        second_avg = sum(d['accuracy'] for d in second_half) / len(second_half)
        trend = "improving" if second_avg > first_avg + 5 else "declining" if second_avg < first_avg - 5 else "stable"
    else:
        trend = "insufficient_data"

    total_attempts = sum(correct_data) + sum(incorrect_data)

    return {
        "period_days": days,
        "date_window": {
            "start": start_date.isoformat(),
            "end": today.isoformat(),
        },
        "daily_stats": daily_stats,
        "trend": trend,
        "active_days": len(non_zero_days),
        "difficulty_breakdown": difficulty_breakdown,
        "labels": labels,
        "correct": correct_data,
        "incorrect": incorrect_data,
        "total_attempts": total_attempts,
    }


@reports_bp.get("/summary")
@require_auth
def get_summary():
    """
    Get high-level learning summary with key metrics.
    
    Query Parameters:
    - days: Filter by last N days (default 30)
    
    Returns:
        JSON with overall statistics and insights
    """
    current_user = g.current_user
    try:
        raw_days = request.args.get("days", 30)
        timeline_payload = _build_timeline_payload(current_user.id, raw_days)
        days = int(timeline_payload.get("period_days", 30))

        # Canonical totals reused from timeline payload for strict consistency.
        total_questions = int(timeline_payload.get("total_attempts", 0))
        correct_answers = int(sum(timeline_payload.get("correct", [])))
        cutoff_date = _utcnow() - timedelta(days=days)
        
        # Overall accuracy
        overall_accuracy = round((correct_answers / total_questions * 100) if total_questions > 0 else 0, 2)
        
        # Most practiced subject
        most_practiced = db.session.query(
            db.text("questions.subject"),
            func.count(AnswerLog.id).label('count')
        ).select_from(AnswerLog).join(
            AnswerLog.question
        ).filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.answered_at >= cutoff_date
        ).group_by(
            db.text("questions.subject")
        ).order_by(desc('count')).first()
        
        most_practiced_subject = most_practiced[0] if most_practiced else None
        
        # Total time spent (in minutes)
        total_time = db.session.query(
            func.sum(AnswerLog.time_spent)
        ).filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.answered_at >= cutoff_date
        ).scalar() or 0
        
        total_time_minutes = round(total_time / 60, 1)
        
        # Average time per question (in seconds)
        avg_time = round(total_time / total_questions, 1) if total_questions > 0 else 0
        
        # Emotion trend (last 7 days)
        emotion_cutoff = _utcnow() - timedelta(days=7)
        emotion_counts = db.session.query(
            EmotionLog.emotion,
            func.count(EmotionLog.id).label('count')
        ).filter(
            EmotionLog.user_id == current_user.id,
            EmotionLog.timestamp >= emotion_cutoff
        ).group_by(EmotionLog.emotion).all()
        
        emotion_trend = [
            {"emotion": e[0], "count": e[1]}
            for e in emotion_counts
        ]
        emotion_trend.sort(key=lambda x: x['count'], reverse=True)
        
        # Improvement trend (compare first half vs second half of period)
        midpoint = cutoff_date + (_utcnow() - cutoff_date) / 2
        
        first_half_correct = AnswerLog.query.filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.is_correct == True,
            AnswerLog.answered_at >= cutoff_date,
            AnswerLog.answered_at < midpoint
        ).count()
        
        first_half_total = AnswerLog.query.filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.answered_at >= cutoff_date,
            AnswerLog.answered_at < midpoint
        ).count()
        
        second_half_correct = AnswerLog.query.filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.is_correct == True,
            AnswerLog.answered_at >= midpoint
        ).count()
        
        second_half_total = AnswerLog.query.filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.answered_at >= midpoint
        ).count()
        
        first_half_accuracy = (first_half_correct / first_half_total * 100) if first_half_total > 0 else 0
        second_half_accuracy = (second_half_correct / second_half_total * 100) if second_half_total > 0 else 0
        improvement = round(second_half_accuracy - first_half_accuracy, 2)
        
        # Recommendations based on data
        recommendations = []
        
        if overall_accuracy < 50:
            recommendations.append("Consider reviewing easier questions to build confidence")
        elif overall_accuracy > 80:
            recommendations.append("Great job! Try challenging yourself with harder questions")
        
        if improvement < -10:
            recommendations.append("Take a break - performance has declined recently")
        elif improvement > 10:
            recommendations.append("Excellent improvement! Keep up the momentum")
        
        # Check emotion patterns
        negative_emotions = ['confused', 'frustrated', 'bored', 'angry', 'sad']
        negative_count = sum(e['count'] for e in emotion_trend if e['emotion'] in negative_emotions)
        total_emotions = sum(e['count'] for e in emotion_trend)
        
        if total_emotions > 0 and (negative_count / total_emotions) > 0.5:
            recommendations.append("Consider taking more breaks - high stress detected")
        
        # Calculate current streak (consecutive correct answers from most recent)
        recent_answers = AnswerLog.query.filter_by(
            user_id=current_user.id
        ).order_by(AnswerLog.answered_at.desc()).limit(50).all()
        
        current_streak = 0
        for answer in recent_answers:
            if answer.is_correct:
                current_streak += 1
            else:
                break
        
        return jsonify({
            "period_days": days,
            "overall_accuracy": overall_accuracy,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "most_practiced_subject": most_practiced_subject,
            "total_time_minutes": total_time_minutes,
            "avg_time_per_question": avg_time,
            "emotion_trend": emotion_trend,
            "improvement": improvement,
            "current_streak": current_streak,
            "recommendations": recommendations
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch summary", "details": str(e)}), 500


@reports_bp.get("/subjects")
@require_auth
def get_subject_breakdown():
    """
    Get detailed breakdown of performance by subject.
    
    Returns:
        JSON with subject-wise statistics
    """
    current_user = g.current_user
    try:
        start_date, today, days = _resolve_timeline_window(
            request.args.get("days", 30),
            current_user.id,
        )

        # Use exact same calendar-date semantics as timeline for consistency.
        timeline_payload = _build_timeline_payload(current_user.id, request.args.get("days", 30))

        rows = db.session.query(
            Question.subject,
            AnswerLog.is_correct,
            AnswerLog.time_spent,
            AnswerLog.answered_at,
            AnswerLog.difficulty_at_time,
            Question.difficulty,
        ).join(
            Question,
            AnswerLog.question_id == Question.id,
        ).filter(
            AnswerLog.user_id == current_user.id,
        ).all()

        subject_buckets = {}
        for subject, is_correct, time_spent, answered_at, difficulty_at_time, question_difficulty in rows:
            answered_day = answered_at.date() if answered_at else None
            if not answered_day or answered_day < start_date or answered_day > today:
                continue

            label = subject or "Unknown"
            if label not in subject_buckets:
                subject_buckets[label] = {
                    "subject": label,
                    "total_questions": 0,
                    "correct_answers": 0,
                    "current_difficulty": "unknown",
                    "difficulty_breakdown": {"easy": 0, "medium": 0, "hard": 0, "expert": 0, "unknown": 0},
                    "last_updated": None,
                    "total_time_spent": 0,
                }

            bucket = subject_buckets[label]
            bucket["total_questions"] += 1
            bucket["correct_answers"] += 1 if bool(is_correct) else 0
            bucket["total_time_spent"] += int(time_spent or 0)

            normalized_difficulty = _normalize_difficulty(difficulty_at_time or question_difficulty)
            bucket["difficulty_breakdown"][normalized_difficulty] += 1
            bucket["current_difficulty"] = normalized_difficulty

            last_updated = bucket["last_updated"]
            if not last_updated or (answered_at and answered_at > last_updated):
                bucket["last_updated"] = answered_at

        subjects = []
        for bucket in subject_buckets.values():
            total = int(bucket["total_questions"])
            correct = int(bucket["correct_answers"])
            accuracy = round((correct / total * 100) if total > 0 else 0, 2)
            subjects.append({
                "subject": bucket["subject"],
                "total_questions": total,
                "correct_answers": correct,
                "accuracy": accuracy,
                "current_difficulty": bucket["current_difficulty"],
                "difficulty_breakdown": bucket["difficulty_breakdown"],
                "last_updated": bucket["last_updated"].isoformat() if bucket["last_updated"] else None,
                "total_time_spent": bucket["total_time_spent"],
            })

        subjects.sort(key=lambda x: x['total_questions'], reverse=True)

        # Cross-check subject totals with timeline total for this period.
        subject_total = sum(s["total_questions"] for s in subjects)
        timeline_total = int(timeline_payload.get("total_attempts", 0))

        return jsonify({
            "period_days": days,
            "date_window": {
                "start": start_date.isoformat(),
                "end": today.isoformat(),
            },
            "subjects": subjects,
            "total_questions": subject_total,
            "totals_match_timeline": (subject_total == timeline_total),
            "timeline_total_attempts": timeline_total,
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch subject breakdown", "details": str(e)}), 500


@reports_bp.get("/emotions")
@require_auth
def get_emotion_analysis():
    """
    Get detailed emotion analysis during learning sessions.
    
    Query Parameters:
    - days: Filter by last N days (default 30)
    
    Returns:
        JSON with emotion distribution and correlations
    """
    current_user = g.current_user
    try:
        days = int(request.args.get("days", 30))
        cutoff_date = _utcnow() - timedelta(days=days)
        
        # Emotion distribution
        emotion_counts = db.session.query(
            EmotionLog.emotion,
            func.count(EmotionLog.id).label('count'),
            func.avg(EmotionLog.confidence).label('avg_confidence')
        ).filter(
            EmotionLog.user_id == current_user.id,
            EmotionLog.timestamp >= cutoff_date
        ).group_by(EmotionLog.emotion).all()
        
        total_logs = sum(e[1] for e in emotion_counts)
        
        distribution = [
            {
                "emotion": e[0],
                "count": e[1],
                "percentage": round((e[1] / total_logs * 100) if total_logs > 0 else 0, 2),
                "avg_confidence": round(e[2], 2)
            }
            for e in emotion_counts
        ]
        distribution.sort(key=lambda x: x['count'], reverse=True)
        
        # Emotion-performance correlation
        emotion_performance = []
        for emotion_data in distribution:
            emotion = emotion_data['emotion']
            
            # Get answers with this emotion
            correct_with_emotion = AnswerLog.query.filter(
                AnswerLog.user_id == current_user.id,
                AnswerLog.emotion_at_time == emotion,
                AnswerLog.is_correct == True,
                AnswerLog.answered_at >= cutoff_date
            ).count()
            
            total_with_emotion = AnswerLog.query.filter(
                AnswerLog.user_id == current_user.id,
                AnswerLog.emotion_at_time == emotion,
                AnswerLog.answered_at >= cutoff_date
            ).count()
            
            accuracy_with_emotion = round((correct_with_emotion / total_with_emotion * 100) if total_with_emotion > 0 else 0, 2)
            
            emotion_performance.append({
                "emotion": emotion,
                "accuracy": accuracy_with_emotion,
                "sample_size": total_with_emotion
            })
        
        # Calculate sentiment score
        positive_emotions = ['happy', 'focused', 'excited', 'neutral']
        negative_emotions = ['confused', 'frustrated', 'bored', 'angry', 'sad']
        
        positive_count = sum(e['count'] for e in distribution if e['emotion'] in positive_emotions)
        negative_count = sum(e['count'] for e in distribution if e['emotion'] in negative_emotions)
        
        sentiment_score = round((positive_count / total_logs * 100) if total_logs > 0 else 50, 2)
        
        return jsonify({
            "period_days": days,
            "total_emotion_logs": total_logs,
            "distribution": distribution,
            "emotion_performance": emotion_performance,
            "sentiment_score": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch emotion analysis", "details": str(e)}), 500


@reports_bp.get("/timeline")
@require_auth
def get_timeline():
    """
    Get progress timeline showing improvement over time.
    
    Query Parameters:
    - days: Number of days to include (default 30)
    
    Returns:
        JSON with daily progress data
    """
    current_user = g.current_user
    try:
        timeline_payload = _build_timeline_payload(current_user.id, request.args.get("days", 30))
        return jsonify(timeline_payload), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch timeline", "details": str(e)}), 500


@reports_bp.get("/integrity")
@require_auth
def get_reports_integrity():
    """Return consistency checks for report metrics across endpoints."""
    current_user = g.current_user
    try:
        raw_days = request.args.get("days", 30)
        timeline_payload = _build_timeline_payload(current_user.id, raw_days)

        start_date = datetime.fromisoformat(timeline_payload["date_window"]["start"]).date()
        end_date = datetime.fromisoformat(timeline_payload["date_window"]["end"]).date()
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

        summary_total = AnswerLog.query.filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.answered_at >= start_dt,
            AnswerLog.answered_at < end_dt,
        ).count()
        summary_correct = AnswerLog.query.filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.is_correct == True,
            AnswerLog.answered_at >= start_dt,
            AnswerLog.answered_at < end_dt,
        ).count()

        timeline_total = int(timeline_payload.get("total_attempts", 0))
        timeline_correct = int(sum(timeline_payload.get("correct", [])))
        timeline_incorrect = int(sum(timeline_payload.get("incorrect", [])))
        difficulty_total = int(sum((timeline_payload.get("difficulty_breakdown") or {}).values()))

        checks = {
            "summary_vs_timeline_total": summary_total == timeline_total,
            "summary_vs_timeline_correct": summary_correct == timeline_correct,
            "timeline_total_vs_components": timeline_total == (timeline_correct + timeline_incorrect),
            "timeline_total_vs_difficulty": timeline_total == difficulty_total,
        }

        mismatches = [name for name, ok in checks.items() if not ok]

        recent_logs = db.session.query(
            AnswerLog.id,
            AnswerLog.answered_at,
            AnswerLog.is_correct,
            AnswerLog.difficulty_at_time,
            Question.difficulty,
        ).outerjoin(
            Question,
            AnswerLog.question_id == Question.id,
        ).filter(
            AnswerLog.user_id == current_user.id,
            AnswerLog.answered_at >= start_dt,
            AnswerLog.answered_at < end_dt,
        ).order_by(
            AnswerLog.answered_at.desc()
        ).limit(15).all()

        sample_rows = [
            {
                "id": row[0],
                "answered_at": row[1].isoformat() if row[1] else None,
                "is_correct": bool(row[2]),
                "difficulty_at_time": row[3],
                "question_difficulty": row[4],
                "normalized_difficulty": _normalize_difficulty(row[3] or row[4]),
            }
            for row in recent_logs
        ]

        return jsonify({
            "ok": len(mismatches) == 0,
            "requested_days": raw_days,
            "period_days": timeline_payload.get("period_days"),
            "date_window": timeline_payload.get("date_window"),
            "totals": {
                "summary_total_questions": summary_total,
                "summary_correct_answers": summary_correct,
                "timeline_total_attempts": timeline_total,
                "timeline_total_correct": timeline_correct,
                "timeline_total_incorrect": timeline_incorrect,
                "difficulty_total_attempts": difficulty_total,
            },
            "checks": checks,
            "mismatches": mismatches,
            "timeline": {
                "labels": timeline_payload.get("labels", []),
                "correct": timeline_payload.get("correct", []),
                "incorrect": timeline_payload.get("incorrect", []),
                "difficulty_breakdown": timeline_payload.get("difficulty_breakdown", {}),
            },
            "sample_recent_logs": sample_rows,
            "route_version": "reports-integrity-v1",
        }), 200
    except Exception as e:
        return jsonify({"error": "Failed to run reports integrity", "details": str(e)}), 500


