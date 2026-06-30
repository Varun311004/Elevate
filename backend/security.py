from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from functools import wraps

import jwt
from werkzeug.security import check_password_hash, generate_password_hash
from flask import current_app, request, jsonify, g

from .models import User, db


def hash_password(password: str) -> str:
    """Hash a password using werkzeug's security functions."""
    return generate_password_hash(password, method='pbkdf2:sha256')


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return check_password_hash(password_hash, password)


def create_access_token(user: User, expires_in: int = 86400) -> str:
    """
    Create a JWT access token for a user.
    
    Args:
        user: User object
        expires_in: Token expiration time in seconds (default 24 hours)
        
    Returns:
        JWT token string
    """
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": str(user.id),  # JWT spec requires 'sub' to be a string
        "email": user.email,
        "role": user.role,
        "school_id": user.school_id,
        "grade": user.grade,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    secret = current_app.config["JWT_SECRET_KEY"]
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload or None if invalid
    """
    try:
        secret = current_app.config["JWT_SECRET_KEY"]
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError as e:
        current_app.logger.warning(f"Token has expired: {str(e)}")
        return None
    except jwt.InvalidTokenError as e:
        current_app.logger.warning(f"Invalid token error: {str(e)}")
        return None
    except Exception as e:
        current_app.logger.error(f"Unexpected error decoding token: {str(e)}")
        return None


def get_token_from_request() -> Optional[str]:
    """
    Extract JWT token from request headers.
    
    Returns:
        Token string or None
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    # Expected format: "Bearer <token>"
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]


def require_auth(f):
    """
    Decorator to protect routes that require authentication.
    
    Usage:
        @app.route('/protected')
        @require_auth
        def protected_route():
            user = g.current_user
            return jsonify({"message": f"Hello {user.name}"})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_request()
        
        if not token:
            return jsonify({"error": "Missing authentication token"}), 401
        
        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        # Load user from database (convert sub back to int)
        user_id = int(payload.get('sub'))
        user = db.session.get(User, user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 401

        # Ensure we have the latest DB state (in case role or related fields changed)
        try:
            db.session.refresh(user)
        except Exception:
            # If refresh fails (detached instance), ignore and continue with available user object
            pass

        if getattr(user, "is_disabled", False):
            return jsonify({"error": "Account disabled"}), 403
        
        # Store user in Flask's g object for access in route
        g.current_user = user
        
        return f(*args, **kwargs)
    
    return decorated_function


def optional_auth(f):
    """
    Decorator for routes that optionally use authentication.
    
    If a valid token is provided, g.current_user will be set.
    If not, g.current_user will be None and the route continues normally.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_request()
        
        if token:
            payload = decode_token(token)
            if payload:
                user_id = int(payload.get('sub'))  # Convert sub back to int
                user = db.session.get(User, user_id)
                if user:
                    g.current_user = user
                else:
                    g.current_user = None
            else:
                g.current_user = None
        else:
            g.current_user = None
        
        return f(*args, **kwargs)
    
    return decorated_function


def role_required(*roles):
    """Require the current user to have one of the given roles.

    Usage:
        @require_auth
        @role_required('teacher')
        def endpoint():
            ...
    """
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user = getattr(g, "current_user", None)
            if not user or user.role not in roles:
                return jsonify({"error": "forbidden"}), 403
            if getattr(user, "is_disabled", False):
                return jsonify({"error": "Account disabled"}), 403
            return f(*args, **kwargs)
        return wrapped
    return wrapper


