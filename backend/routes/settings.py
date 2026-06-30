"""Settings routes for user preferences persistence."""

from flask import Blueprint, jsonify, request, g

from ..models import db, UserSetting
from ..security import require_auth

settings_bp = Blueprint("settings", __name__)

DEFAULT_SETTINGS = {
    "enableTimer": False,
    "timerDuration": 60,
    "autoSubmit": False,
    "defaultDifficulty": "adaptive",
    "questionsPerSession": 10,
    "showExplanations": True,
    "requireCamera": True,
    "enableEmotionFeedback": True,
    "detectionFrequency": "medium",
    "enableNotifications": True,
    "enableSoundEffects": False,
}


def _sanitize_settings(payload: dict) -> dict:
    """Validate and sanitize supported settings keys."""
    if not isinstance(payload, dict):
        return {}

    sanitized = {}

    if "enableTimer" in payload:
        sanitized["enableTimer"] = bool(payload.get("enableTimer"))

    if "timerDuration" in payload:
        try:
            value = int(payload.get("timerDuration", 60))
            sanitized["timerDuration"] = max(10, min(value, 300))
        except (TypeError, ValueError):
            pass

    if "autoSubmit" in payload:
        sanitized["autoSubmit"] = bool(payload.get("autoSubmit"))

    if "defaultDifficulty" in payload:
        value = str(payload.get("defaultDifficulty", "adaptive")).lower()
        sanitized["defaultDifficulty"] = value if value in {"adaptive", "easy", "medium", "hard"} else "adaptive"

    if "questionsPerSession" in payload:
        try:
            value = int(payload.get("questionsPerSession", 10))
            sanitized["questionsPerSession"] = max(5, min(value, 50))
        except (TypeError, ValueError):
            pass

    if "showExplanations" in payload:
        sanitized["showExplanations"] = bool(payload.get("showExplanations"))

    if "requireCamera" in payload:
        sanitized["requireCamera"] = bool(payload.get("requireCamera"))

    if "enableEmotionFeedback" in payload:
        sanitized["enableEmotionFeedback"] = bool(payload.get("enableEmotionFeedback"))

    if "detectionFrequency" in payload:
        value = str(payload.get("detectionFrequency", "medium")).lower()
        sanitized["detectionFrequency"] = value if value in {"high", "medium", "low"} else "medium"

    if "enableNotifications" in payload:
        sanitized["enableNotifications"] = bool(payload.get("enableNotifications"))

    if "enableSoundEffects" in payload:
        sanitized["enableSoundEffects"] = bool(payload.get("enableSoundEffects"))

    return sanitized


@settings_bp.get("/")
@require_auth
def get_settings():
    current_user = g.current_user

    try:
        record = UserSetting.query.filter_by(user_id=current_user.id).first()
        persisted = record.settings_json if record and isinstance(record.settings_json, dict) else {}
        merged = {**DEFAULT_SETTINGS, **persisted}
        return jsonify({"settings": merged}), 200
    except Exception as exc:
        return jsonify({"error": "Failed to fetch settings", "details": str(exc)}), 500


@settings_bp.put("/")
@require_auth
def update_settings():
    current_user = g.current_user
    payload = request.get_json(silent=True) or {}

    try:
        incoming = _sanitize_settings(payload)
        record = UserSetting.query.filter_by(user_id=current_user.id).first()

        if not record:
            record = UserSetting(user_id=current_user.id, settings_json={})
            db.session.add(record)

        current = record.settings_json if isinstance(record.settings_json, dict) else {}
        record.settings_json = {**DEFAULT_SETTINGS, **current, **incoming}

        db.session.commit()
        return jsonify({"settings": record.settings_json}), 200
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": "Failed to update settings", "details": str(exc)}), 500
