import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.config import get_config
from backend.models import db, User
from backend.security import hash_password

config = get_config('development')

from flask import Flask
app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)

with app.app_context():
    db.create_all()  # Ensure all tables exist before seeding

    # Create admin user
    admin_email = 'admin@elevate.com'
    if not User.query.filter_by(email=admin_email).first():
        admin = User(
            name='Admin User',
            email=admin_email,
            password_hash=hash_password('admin123'),
            grade='college',
            role='admin',
            is_verified=True
        )
        db.session.add(admin)
        print(f'Created admin user: {admin_email} (password: admin123)')

    # Create student user
    student_email = 'student@elevate.com'
    if not User.query.filter_by(email=student_email).first():
        student = User(
            name='Test Student',
            email=student_email,
            password_hash=hash_password('student123'),
            grade='high',
            role='student',
            is_verified=True
        )
        db.session.add(student)
        print(f'Created student user: {student_email} (password: student123)')

    # Create teacher user
    teacher_email = 'teacher@elevate.com'
    if not User.query.filter_by(email=teacher_email).first():
        teacher = User(
            name='Test Teacher',
            email=teacher_email,
            password_hash=hash_password('teacher123'),
            grade='college',
            role='teacher',
            is_verified=True
        )
        db.session.add(teacher)
        print(f'Created teacher user: {teacher_email} (password: teacher123)')

    db.session.commit()
    print('All test users created successfully!')