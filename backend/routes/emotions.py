"""Emotion logging routes for tracking user emotional states during learning."""
from datetime import datetime, timedelta, timezone
from flask import Blueprint, jsonify, request, g
from sqlalchemy import func
from ..models import db, EmotionLog
from ..security import require_auth
from ..validation import sanitize_string

emotions_bp = Blueprint("emotions", __name__)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@emotions_bp.post("/")
@require_auth
def log_emotion():
    """
    Log an emotion sample from the user.
    
    Request Body:
    - emotion: Emotion name (e.g., "happy", "confused", "frustrated", "focused", "bored")
    - confidence: Confidence score (0-1)
    - context: Optional context (e.g., "answering_question", "reading_hint", "session_start")
    
    Returns:
        JSON with confirmation
    """
    current_user = g.current_user
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        emotion = data.get("emotion")
        confidence = data.get("confidence")
        context = data.get("context")
        
        # Validation
        if not emotion:
            return jsonify({"error": "emotion is required"}), 400
        
        if confidence is None:
            return jsonify({"error": "confidence is required"}), 400
        
        try:
            confidence = float(confidence)
            if confidence < 0 or confidence > 1:
                return jsonify({"error": "confidence must be between 0 and 1"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "confidence must be a number"}), 400
        
        # Validate emotion
        valid_emotions = ['happy', 'bored', 'focused', 'confused', 'neutral', 'angry', 'surprised']
        emotion = sanitize_string(emotion).lower()
        if emotion not in valid_emotions:
            return jsonify({"error": f"Invalid emotion. Must be one of: {', '.join(valid_emotions)}"}), 400
        
        # Create emotion log
        emotion_log = EmotionLog(
            user_id=current_user.id,
            emotion=emotion,
            confidence=confidence,
            context=sanitize_string(context) if context else None,
            timestamp=_utcnow()
        )
        
        db.session.add(emotion_log)
        db.session.commit()
        
        return jsonify({
            "status": "logged",
            "id": emotion_log.id,
            "emotion": emotion,
            "timestamp": emotion_log.timestamp.isoformat()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to log emotion", "details": str(e)}), 500


@emotions_bp.get("/history")
@require_auth
def get_emotion_history():
    """
    Get emotion history for the user.
    
    Query Parameters:
    - limit: Max results (default 50, max 200)
    - offset: Pagination offset (default 0)
    - days: Filter by last N days (optional)
    - context: Filter by context (optional)
    
    Returns:
        JSON with emotion history and summary statistics
    """
    current_user = g.current_user
    try:
        # Get query parameters
        limit = min(int(request.args.get("limit", 50)), 200)
        offset = int(request.args.get("offset", 0))
        days = request.args.get("days")
        context_filter = request.args.get("context")
        
        # Build query
        query = EmotionLog.query.filter_by(user_id=current_user.id)
        
        # Apply filters
        if days:
            try:
                days_int = int(days)
                cutoff_date = _utcnow() - timedelta(days=days_int)
                query = query.filter(EmotionLog.timestamp >= cutoff_date)
            except ValueError:
                return jsonify({"error": "Invalid days parameter"}), 400
        
        if context_filter:
            query = query.filter(EmotionLog.context == sanitize_string(context_filter))
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        emotions = query.order_by(EmotionLog.timestamp.desc()).offset(offset).limit(limit).all()
        
        # Format results
        history = [
            {
                "id": e.id,
                "emotion": e.emotion,
                "confidence": e.confidence,
                "context": e.context,
                "timestamp": e.timestamp.isoformat()
            }
            for e in emotions
        ]
        
        return jsonify({
            "history": history,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch emotion history", "details": str(e)}), 500


@emotions_bp.get("/summary")
@require_auth
def get_emotion_summary():
    """
    Get summary statistics of emotions.
    
    Query Parameters:
    - days: Filter by last N days (default 7)
    
    Returns:
        JSON with emotion distribution and trends
    """
    current_user = g.current_user
    try:
        days = int(request.args.get("days", 7))
        cutoff_date = _utcnow() - timedelta(days=days)
        
        # Get emotion distribution
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
        
        # Sort by count descending
        distribution.sort(key=lambda x: x['count'], reverse=True)
        
        # Determine dominant emotion
        dominant_emotion = distribution[0]['emotion'] if distribution else None
        
        # Calculate positive vs negative emotion ratio
        positive_emotions = ['happy', 'focused', 'neutral', 'surprised']
        negative_emotions = ['confused', 'bored', 'angry']
        
        positive_count = sum(e['count'] for e in distribution if e['emotion'] in positive_emotions)
        negative_count = sum(e['count'] for e in distribution if e['emotion'] in negative_emotions)
        
        sentiment_score = round((positive_count / total_logs * 100) if total_logs > 0 else 0, 2)
        
        return jsonify({
            "period_days": days,
            "total_logs": total_logs,
            "dominant_emotion": dominant_emotion,
            "sentiment_score": sentiment_score,
            "distribution": distribution,
            "positive_count": positive_count,
            "negative_count": negative_count
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch emotion summary", "details": str(e)}), 500


@emotions_bp.get("/timeline")
@require_auth
def get_emotion_timeline():
    """
    Get emotion timeline grouped by time periods.
    
    Query Parameters:
    - days: Number of days to include (default 7)
    - group_by: Grouping period - "hour", "day" (default "day")
    
    Returns:
        JSON with emotion data grouped by time periods
    """
    current_user = g.current_user
    try:
        days = int(request.args.get("days", 7))
        group_by = request.args.get("group_by", "day")
        
        if group_by not in ["hour", "day"]:
            return jsonify({"error": "group_by must be 'hour' or 'day'"}), 400
        
        cutoff_date = _utcnow() - timedelta(days=days)
        
        # Get all emotions in the period
        emotions = EmotionLog.query.filter(
            EmotionLog.user_id == current_user.id,
            EmotionLog.timestamp >= cutoff_date
        ).order_by(EmotionLog.timestamp.asc()).all()
        
        # Group by time period
        timeline = {}
        for emotion in emotions:
            if group_by == "day":
                period_key = emotion.timestamp.date().isoformat()
            else:  # hour
                period_key = emotion.timestamp.strftime("%Y-%m-%d %H:00")
            
            if period_key not in timeline:
                timeline[period_key] = {
                    "period": period_key,
                    "emotions": {},
                    "total_count": 0
                }
            
            if emotion.emotion not in timeline[period_key]["emotions"]:
                timeline[period_key]["emotions"][emotion.emotion] = 0
            
            timeline[period_key]["emotions"][emotion.emotion] += 1
            timeline[period_key]["total_count"] += 1
        
        # Convert to list and sort by period
        timeline_list = sorted(timeline.values(), key=lambda x: x['period'])
        
        return jsonify({
            "period_days": days,
            "group_by": group_by,
            "timeline": timeline_list
        }), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch emotion timeline", "details": str(e)}), 500


