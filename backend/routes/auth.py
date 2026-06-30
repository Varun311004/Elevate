from flask import Blueprint, current_app, jsonify, request, g
from datetime import datetime, timedelta, timezone
import secrets
import re
from sqlalchemy import func

from ..models import User, School, db
from ..security import create_access_token, hash_password, verify_password, require_auth
from ..validation import (
    validate_email, validate_password, validate_name, validate_grade,
    sanitize_string, validate_required_fields
)


auth_bp = Blueprint("auth", __name__)
SCHOOL_SLUG_HINT_COOKIE = "elevate_school_slug_hint"
_SCHOOL_SLUG_MAX_LEN = 128


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_school_slug(value) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None

    normalized = re.sub(r"[^a-z0-9\s_-]+", "", normalized)
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        return None
    return normalized[:_SCHOOL_SLUG_MAX_LEN]


def _school_slug_from_user(user: User) -> str | None:
    school = user.school if getattr(user, "school_id", None) else None
    if not school:
        return None

    slug = _normalize_school_slug(school.slug)
    if slug:
        return slug
    return _normalize_school_slug(school.name)


def _apply_slug_hint_cookie(response, slug_value):
    slug = _normalize_school_slug(slug_value)
    if slug:
        response.set_cookie(
            SCHOOL_SLUG_HINT_COOKIE,
            slug,
            max_age=60 * 60 * 24 * 30,
            path="/",
            samesite="Lax",
        )
    else:
        response.delete_cookie(SCHOOL_SLUG_HINT_COOKIE)
    return response


def _auth_user_payload(user: User, school_slug_hint: str | None = None) -> dict:
    school = user.school if getattr(user, "school_id", None) else None

    school_name = None
    if school:
        school_name = school.name

    resolved_slug = _school_slug_from_user(user)
    if not resolved_slug:
        resolved_slug = _normalize_school_slug(school_slug_hint)

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "grade": user.grade,
        "role": user.role,
        "is_verified": user.is_verified,
        "is_disabled": bool(getattr(user, "is_disabled", False)),
        "disabled_at": user.disabled_at.isoformat() if getattr(user, "disabled_at", None) else None,
        "disabled_reason": getattr(user, "disabled_reason", None),
        "school_id": user.school_id,
        "school_slug": resolved_slug,
        "school_name": school_name,
    }


@auth_bp.post("/signup")
def signup():
    """Register a new user with validation."""
    data = request.get_json(silent=True) or {}
    error = validate_required_fields(data, ['name', 'email', 'password'])
    if error:
        return jsonify({"error": error}), 400
    name = sanitize_string(data.get("name"), max_length=100)
    email = sanitize_string(data.get("email"), max_length=255).lower()
    password = data.get("password")
    grade = sanitize_string(data.get("grade", ""), max_length=10) or None
    role = sanitize_string(data.get("role", "student"), max_length=20).lower()
    school_name = sanitize_string(data.get("school_name", ""), max_length=255) or None
    school_slug = _normalize_school_slug(data.get("school_slug"))
    if role not in ("student", "teacher", "admin"):
        return jsonify({"error": "Invalid role. Must be 'student', 'teacher', or 'admin'"}), 400

    if role == "admin":
        grade = None
        if not school_name:
            return jsonify({"error": "School name is required for admin account setup"}), 400
        if not school_slug:
            return jsonify({"error": "School slug is required for admin account setup"}), 400

    if not validate_name(name):
        return jsonify({"error": "Invalid name format. Must be 2-100 characters with letters only"}), 400
    if not validate_email(email):
        return jsonify({"error": "Invalid email address format"}), 400
    password_validation = validate_password(password)
    if not password_validation['valid']:
        return jsonify({"error": "Password validation failed", "details": password_validation['errors']}), 400
    if grade and not validate_grade(grade):
        return jsonify({"error": "Invalid grade level. Must be: elementary, middle, high, or college"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409
    try:
        school = None
        if role == "admin":
            school = School.query.filter(func.lower(School.slug) == school_slug.lower()).first()
            if not school:
                school = School.query.filter(func.lower(School.name) == school_name.lower()).first()

            if school:
                if not school.slug:
                    school.slug = school_slug
                if school.name != school_name:
                    school.name = school_name
            else:
                school = School(name=school_name, slug=school_slug)
                db.session.add(school)
                db.session.flush()

        user = User(
            name=name,
            email=email,
            password_hash=hash_password(password),
            grade=grade,
            role=role,
            school_id=school.id if school else None,
            is_verified=False,
        )
        db.session.add(user)
        db.session.commit()
        current_app.logger.info(f"New user registered: {email} as {role}")
        token = create_access_token(user)
        response = jsonify({"token": token, "user": _auth_user_payload(user)})
        _apply_slug_hint_cookie(response, _school_slug_from_user(user))
        return response, 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": "Failed to create account. Please try again"}), 500


@auth_bp.post("/login")
def login():
    """Log in an existing user and return a JWT."""
    data = request.get_json(silent=True) or {}
    error = validate_required_fields(data, ['email', 'password'])
    if error:
        return jsonify({"error": error}), 400
    email = sanitize_string(data.get("email"), max_length=255).lower()
    password = data.get("password")
    if not validate_email(email):
        return jsonify({"error": "Invalid email address format"}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        current_app.logger.warning(f"Failed login attempt for email: {email}")
        return jsonify({"error": "Invalid email or password"}), 401
    if getattr(user, "is_disabled", False):
        current_app.logger.warning(f"Disabled user login attempt: {email}")
        return jsonify({"error": "Your account is disabled. Please contact an administrator."}), 403
    current_app.logger.info(f"User logged in: {email}")

    hint_slug = _normalize_school_slug(request.cookies.get(SCHOOL_SLUG_HINT_COOKIE))
    resolved_slug = _school_slug_from_user(user) or hint_slug
    token = create_access_token(user)
    response = jsonify({"token": token, "user": _auth_user_payload(user, school_slug_hint=resolved_slug)})
    _apply_slug_hint_cookie(response, resolved_slug)
    return response


@auth_bp.get("/me")
@require_auth
def me():
    """Get current user information from JWT token."""
    user = g.current_user
    payload = _auth_user_payload(user)
    payload["created_at"] = user.created_at.isoformat() if user.created_at else None
    response = jsonify({"user": payload})
    _apply_slug_hint_cookie(response, payload.get("school_slug"))
    return response, 200


@auth_bp.post("/logout")
@require_auth
def logout():
    """Logout endpoint (JWT is stateless, so this is mostly for logging)."""
    user = g.current_user
    current_app.logger.info(f"User logged out: {user.email}")
    response = jsonify({"message": "Logged out successfully"})
    response.delete_cookie(SCHOOL_SLUG_HINT_COOKIE)
    return response, 200


@auth_bp.post('/refresh')
@require_auth
def refresh_token():
    """Refresh the user's access token (simple refresh mechanism using existing token)."""
    user = g.current_user
    # create a new token with same claims
    token = create_access_token(user)
    return jsonify({"token": token}), 200


@auth_bp.post("/request-reset")
def request_password_reset():
    """Request password reset (stub for now)."""
    data = request.get_json(silent=True) or {}
    error = validate_required_fields(data, ['email'])
    if error:
        return jsonify({"error": error}), 400
    email = sanitize_string(data.get("email"), max_length=255).lower()
    if not validate_email(email):
        return jsonify({"error": "Invalid email address format"}), 400
    user = User.query.filter_by(email=email).first()
    if user:
        reset_token = secrets.token_urlsafe(32)
        user.reset_token = reset_token
        user.reset_token_expires = _utcnow() + timedelta(hours=1)
        db.session.commit()
        current_app.logger.info(f"Password reset requested for: {email}")
    return jsonify({"message": "If the email exists, a reset link has been sent"}), 200


@auth_bp.post("/reset-password")
def reset_password():
    """Reset password using token (stub for now)."""
    data = request.get_json(silent=True) or {}
    error = validate_required_fields(data, ['token', 'new_password'])
    if error:
        return jsonify({"error": error}), 400
    token = sanitize_string(data.get("token"), max_length=100)
    new_password = data.get("new_password")
    password_validation = validate_password(new_password)
    if not password_validation['valid']:
        return jsonify({"error": "Password validation failed", "details": password_validation['errors']}), 400
    user = User.query.filter_by(reset_token=token).first()
    if not user or not user.reset_token_expires or user.reset_token_expires < _utcnow():
        return jsonify({"error": "Invalid or expired reset token"}), 400
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.session.commit()
    current_app.logger.info(f"Password reset completed for: {user.email}")
    return jsonify({"message": "Password reset successful"}), 200


@auth_bp.post("/verify-email")
def verify_email():
    """Email verification endpoint (stub for now)."""
    data = request.get_json(silent=True) or {}
    error = validate_required_fields(data, ['token'])
    if error:
        return jsonify({"error": error}), 400
    token = sanitize_string(data.get("token"), max_length=100)
    return jsonify({"message": "Email verification feature coming soon"}), 200
