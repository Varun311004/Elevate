import smtplib
from email.message import EmailMessage
from flask import current_app


def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email using configured SMTP or memory backend for tests/dev.

    Behavior:
    - If app.config['EMAIL_BACKEND'] == 'memory' or app.config['TESTING'] is True: append to current_app.extensions['sent_emails']
    - Otherwise, use smtplib with SMTP_HOST/SMTP_PORT/TLS/USER/PASS
    """
    app = current_app._get_current_object()
    backend = app.config.get('EMAIL_BACKEND', 'console')

    if app.config.get('TESTING') or backend == 'memory':
        sent = app.extensions.setdefault('sent_emails', [])
        sent.append({'to': to, 'subject': subject, 'body': body})
        app.logger.info(f"[email memory] to={to} subject={subject}")
        return True

    # Console backend: just log
    if backend == 'console':
        app.logger.info(f"[email console] to={to} subject={subject}\n{body}")
        return True

    # SMTP backend
    host = app.config.get('SMTP_HOST')
    port = int(app.config.get('SMTP_PORT', 587))
    user = app.config.get('SMTP_USER')
    password = app.config.get('SMTP_PASSWORD')
    use_tls = app.config.get('SMTP_USE_TLS', True)
    from_addr = app.config.get('SMTP_FROM', user or 'no-reply@localhost')

    if not host:
        app.logger.error('SMTP_HOST not configured')
        return False

    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_addr
        msg['To'] = to
        msg.set_content(body)

        with smtplib.SMTP(host, port, timeout=10) as s:
            if use_tls:
                s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)
        app.logger.info(f"[email smtp] sent to {to}")
        return True
    except Exception as e:
        app.logger.exception(f"Failed to send email to {to}: {e}")
        return False
