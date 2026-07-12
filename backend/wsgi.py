import os

from backend.app import create_app

app = create_app(
    os.environ.get("FLASK_ENV", "production")
)