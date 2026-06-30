"""Input validation and sanitization utilities."""
import re
import bleach
from typing import Optional, Dict, Any


class ValidationError(Exception):
    """Custom validation error."""
    pass


def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input to prevent XSS attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not text:
        return ""
    
    # Remove any HTML tags
    sanitized = bleach.clean(text, tags=[], strip=True)
    
    # Strip whitespace
    sanitized = sanitized.strip()
    
    # Truncate if needed
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def validate_email(email: str) -> bool:
    """
    Validate email format using regex.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    
    # RFC 5322 simplified email regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email.strip()) is not None


def validate_password(password: str) -> Dict[str, Any]:
    """
    Validate password strength.
    
    Rules:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    
    Args:
        password: Password to validate
        
    Returns:
        Dict with 'valid' bool and 'errors' list
    """
    errors = []
    
    if not password:
        return {'valid': False, 'errors': ['Password is required']}
    
    if len(password) < 8:
        errors.append('Password must be at least 8 characters long')
    
    if len(password) > 128:
        errors.append('Password must not exceed 128 characters')
    
    if not re.search(r'[A-Z]', password):
        errors.append('Password must contain at least one uppercase letter')
    
    if not re.search(r'[a-z]', password):
        errors.append('Password must contain at least one lowercase letter')
    
    if not re.search(r'\d', password):
        errors.append('Password must contain at least one digit')
    
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>?/\\|`~]', password):
        errors.append('Password must contain at least one special character')
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }


def validate_name(name: str) -> bool:
    """
    Validate name field.
    
    Args:
        name: Name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or not name.strip():
        return False
    
    # Name should be 2-100 characters, letters, spaces, hyphens, apostrophes
    if len(name.strip()) < 2 or len(name.strip()) > 100:
        return False
    
    # Allow letters, spaces, hyphens, apostrophes, and common international characters
    name_pattern = r"^[a-zA-Z\u00C0-\u017F\s'-]+$"
    return re.match(name_pattern, name.strip()) is not None


def validate_grade(grade: str) -> bool:
    """
    Validate grade field.
    
    Args:
        grade: Grade level to validate (e.g., "elementary", "middle", "high", "college")
        
    Returns:
        True if valid, False otherwise
    """
    if not grade:
        return True  # Grade is optional
    
    # Valid grade levels: Elementary (K-5), Middle School (6-8), High School (9-12), College
    valid_grades = ['elementary', 'middle', 'high', 'college']
    return grade.strip().lower() in valid_grades


def validate_subject(subject: str) -> bool:
    """
    Validate subject field.
    
    Args:
        subject: Subject to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_subjects = [
        'mathematics', 'science', 'english', 'history', 
        'geography', 'physics', 'chemistry', 'biology',
        'computer_science', 'social_studies'
    ]
    return subject.lower() in valid_subjects


def validate_difficulty(difficulty: str) -> bool:
    """
    Validate difficulty level.
    
    Args:
        difficulty: Difficulty level to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_difficulties = ['easy', 'medium', 'hard']
    return difficulty.lower() in valid_difficulties


def sanitize_json_input(data: Dict[str, Any], allowed_keys: set) -> Dict[str, Any]:
    """
    Sanitize JSON input by removing unexpected keys and sanitizing string values.
    
    Args:
        data: Input dictionary
        allowed_keys: Set of allowed keys
        
    Returns:
        Sanitized dictionary
    """
    sanitized = {}
    
    for key in allowed_keys:
        if key in data:
            value = data[key]
            
            # Sanitize string values
            if isinstance(value, str):
                sanitized[key] = sanitize_string(value, max_length=1000)
            else:
                sanitized[key] = value
    
    return sanitized


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> Optional[str]:
    """
    Check if all required fields are present and non-empty.
    
    Args:
        data: Input dictionary
        required_fields: List of required field names
        
    Returns:
        Error message if validation fails, None otherwise
    """
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
        
        value = data[field]
        if value is None or (isinstance(value, str) and not value.strip()):
            return f"Field '{field}' cannot be empty"
    
    return None
