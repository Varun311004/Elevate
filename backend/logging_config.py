"""Logging configuration for the Elevate backend."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import uuid
from flask import g, request, has_request_context


class RequestIdFilter(logging.Filter):
    """Add request ID to log records."""
    
    def filter(self, record):
        if has_request_context():
            record.request_id = getattr(g, 'request_id', 'N/A')
            record.method = request.method if request else 'N/A'
            record.path = request.path if request else 'N/A'
        else:
            record.request_id = 'N/A'
            record.method = 'N/A'
            record.path = 'N/A'
        return True


def configure_logging(app):
    """Configure structured logging for the Flask app."""
    
    # Remove default Flask handlers
    app.logger.handlers.clear()
    
    # Set log level based on environment
    log_level = logging.DEBUG if app.config.get('DEBUG') else logging.INFO
    app.logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(request_id)s] %(method)s %(path)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    app.logger.addHandler(console_handler)
    
    # File handler (rotating) - only in production.
    # Render/runtime containers may not have this directory pre-created.
    if not app.config.get('DEBUG') and not app.config.get('TESTING'):
        try:
            log_path = os.path.join('logs', 'elevate.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=10485760,  # 10MB
                backupCount=10
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(RequestIdFilter())
            app.logger.addHandler(file_handler)
        except Exception as exc:
            # Do not block startup if file logging cannot be configured.
            app.logger.warning(f"File logging disabled: {exc}")
    
    # Add request ID to each request
    @app.before_request
    def before_request():
        g.request_id = str(uuid.uuid4())[:8]
    
    log_requests = str(app.config.get("LOG_REQUESTS", os.environ.get("ELEVATE_LOG_REQUESTS", "0"))).strip().lower() in {
        "1", "true", "yes", "on"
    }

    # Log each request only when enabled.
    @app.after_request
    def after_request(response):
        if log_requests:
            app.logger.info(
                f"Request completed - Status: {response.status_code}"
            )
        return response
    
    app.logger.info("Logging configured successfully")
