<!-- # Elevate - Adaptive STEM Learning Platform

Elevate is a full-stack adaptive learning platform focused on STEM practice, emotion-aware learning signals, and teacher-managed assessment workflows.

The project includes:
- Student learning and progress tracking
- Teacher classroom, assignment, and test lifecycle management
- Admin management routes
- AI-assisted question generation (Gemini or OpenAI-compatible providers)
- Reports and analytics across progress, tests, and emotion signals

## 1. Current Architecture

### Backend
- Framework: Flask
- ORM: Flask-SQLAlchemy
- Migrations: Alembic
- Auth: JWT-based bearer token flow
- API organization: feature blueprints under backend/routes

Primary backend files:
- backend/app.py: app factory, CORS, blueprint registration, static serving
- backend/config.py: environment-driven configuration
- backend/models.py: relational schema and lifecycle entities
- backend/routes/*.py: feature endpoints by role/domain
- backend/question_generator.py: AI + fallback question generation engine

### Frontend
- Stack: Vanilla HTML, CSS, JavaScript (ES modules)
- API client: frontend/js/api.js
- Teacher experience: frontend/teacher-dashboard.html + frontend/js/teacher-dashboard.js
- Student experience: dashboard, learning, reports, profile, settings pages

### Deployment shape in development
- Backend serves API under /api/*
- Backend also serves frontend pages (/, /admin, static files)
- Default local URL: http://127.0.0.1:5000

## 2. Core Features Implemented

### Student domain
- Authentication and role-aware session usage
- Practice and test flows
- Assigned test intake from teacher assignments
- Answer logging, test submission, and result persistence
- Progress, emotion logs, and report endpoints

### Teacher domain
- Dashboard summary metrics
- Question bank generation (preview and persistence)
- Test CRUD + publication lifecycle
- Classroom CRUD and student enrollment mapping
- Assignment creation and status management
- Reports and analytics scoped to teacher context

### Admin domain
- School and request management
- Admin-protected operations via dedicated routes

### Reporting and analytics
- Timeline and distribution reporting
- Subject and difficulty performance tracking
- Teacher analytics for weak-topic identification

## 3. Data Model (Implementation-Level)

Key entities in backend/models.py:
- User: student/teacher/admin identities, role, school linkage
- School: institutional scope for users and classrooms
- Question: bank entries with metadata, difficulty, options, AI-generation fields
- Test: teacher-owned assessment containers
- TestQuestion: ordered question mapping for each test
- TestResult: student test outcomes and scoring
- AnswerLog: per-question responses and correctness
- UserProgress, SubjectPerformance, EmotionLog: adaptive and behavioral tracking
- Classroom: teacher-owned grouped student container
- ClassroomStudent: classroom membership bridge table
- TestAssignment: assignment lifecycle across test/classroom/student
- UserSetting: per-user learning/profile settings

### Lifecycle design
The classroom-assignment-test workflow is modeled as explicit relational entities, not ad-hoc flags.
This supports:
- Teacher-to-student linkage through classrooms
- Assignment status transitions (assigned, started, submitted, reviewed, expired, cancelled)
- Traceability across assignment, test result, and answer-level logs

## 4. AI Question Generation

Implemented in backend/question_generator.py.

### Provider support
- Gemini native path
  - Uses GEMINI_API_KEY or GOOGLE_API_KEY
  - Optional GEMINI_MODEL (defaults are handled in code)
- OpenAI-compatible path
  - Uses AI_API_KEY
  - Optional AI_MODEL and AI_API_BASE

### Generation behavior
- Structured JSON request/response handling with sanitization
- Deduplication by question text
- Multi-attempt top-up to reduce shortfall failures
- Provider/model fallback logic for Gemini path
- Strict mode support for endpoints that require complete AI count

### Endpoint usage pattern
Teacher endpoints use require_ai controls:
- Preview mode can return partial generated sets
- Persist/create-test paths enforce sufficient generated questions before commit

## 5. API Surface (High-Level)

Registered blueprints in backend/app.py:
- /api/auth
- /api/questions
- /api/progress
- /api/emotions
- /api/reports
- /api/admin
- /api/teacher
- /api/student
- /api/settings

Selected teacher routes:
- POST /api/teacher/question-bank/generate
- POST /api/teacher/tests
- GET /api/teacher/tests
- POST /api/teacher/classrooms
- POST /api/teacher/classrooms/<id>/students
- POST /api/teacher/assignments
- PATCH /api/teacher/assignments/<id>
- GET /api/teacher/reports
- GET /api/teacher/analytics

Selected student routes:
- GET /api/student/assigned-tests
- GET /api/student/tests
- POST /api/student/tests/<id>/start
- GET /api/student/tests/<id>/questions
- POST /api/student/tests/<id>/answer
- POST /api/student/tests/<id>/finish

## 6. Frontend-Backend Integration

Client-side API abstraction is centralized in frontend/js/api.js.

Important implementation details:
- JWT is read from localStorage/sessionStorage
- Authorization header is attached automatically
- GET requests include cache-busting query timestamp
- 401 responses clear stored session and redirect to login
- Feature namespaces map to backend domains (auth, questions, teacher, student, reports, settings)

## 7. Local Setup and Run

## Prerequisites
- Python 3.11+
- PostgreSQL (default config uses PostgreSQL URI)

## Option A: one-command Windows bootstrap
Run from project root:
- start.bat

This script:
- creates .venv if needed
- installs dependencies
- seeds users and questions
- starts backend runner

## Option B: manual setup
1. Create and activate virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Configure environment variables
4. Run backend:
   python -m backend.app

App URL:
- http://127.0.0.1:5000

## Seed data users
Created by seed_users.py:
- admin@elevate.com / admin123
- teacher@elevate.com / teacher123
- student@elevate.com / student123

## 8. Configuration

Core config in backend/config.py.

Common environment variables:
- FLASK_ENV
- SECRET_KEY
- JWT_SECRET
- ADMIN_TOKEN
- DATABASE_URL

AI provider variables:
- AI_PROVIDER (gemini or openai, optional auto-detect)
- GEMINI_API_KEY or GOOGLE_API_KEY
- GEMINI_MODEL (optional)
- AI_API_KEY (OpenAI-compatible)
- AI_MODEL (optional)
- AI_API_BASE (optional)

## 9. Testing

Run key backend tests:
- python -m pytest backend/tests/test_teacher_reports.py backend/tests/test_auth.py

Current test coverage includes:
- role-protected auth flow basics
- teacher report scoping
- classroom school auto-provision regression scenario

## 10. Implementation Notes and Design Decisions

- App factory pattern keeps environment bootstrapping clean and testable.
- Teacher school auto-provision logic prevents null school_id failures when creating first classroom.
- AI generation paths avoid silent fake success in strict workflows.
- Assignment lifecycle is persisted explicitly to support reporting and operational workflows.
- Frontend remains framework-free for low build complexity and easy deployment in academic setups.

## 11. Troubleshooting

### AI generation is not configured
Cause:
- Backend process cannot see key env vars

Fix:
- Set provider and key in the same shell that launches backend
- Restart backend process after changing env vars

### AI provider returned insufficient questions (502)
Cause:
- Provider quota/rate/output shortfall

Fix:
- retry request
- lower requested count
- verify provider model and key limits

### Classroom creation school_id errors
Handled in current implementation with teacher school auto-provision.
If this reappears, verify backend is running latest code and restart service.

## 12. Repository Layout

Top-level:
- backend: Flask API, models, routes, migrations, tests
- frontend: HTML/CSS/JS pages and modules
- dataset: emotion dataset folders
- instance: runtime DB artifacts (environment dependent)
- requirements.txt: Python dependencies
- start.bat: Windows bootstrap runner

---

If you want, the next improvement can be an API status endpoint (provider/model/key-detected and generation diagnostics) so AI failures are visible directly inside the teacher dashboard. -->


# Elevate: AI-Powered Adaptive STEM Learning Platform

Elevate is an advanced, full-stack educational ecosystem designed to optimize STEM learning through artificial intelligence. Moving beyond traditional static assessments, Elevate utilizes **Cognitive AI (Bayesian Knowledge Tracing)**, **Affective Computing (Computer Vision Emotion Tracking)**, and **Predictive Analytics** to create a highly responsive, personalized learning environment.

This project serves as a comprehensive demonstration of integrating deep machine learning algorithms and LLM pipelines into a robust web architecture.

---

## 🚀 Core AI Architecture

Elevate shifts the paradigm from simple CRUD operations to an intelligent, state-driven platform:

### 1. Algorithmic Adaptive Learning (Cognitive AI)
* **Bayesian Knowledge Tracing (BKT):** The engine dynamically estimates a student's probability of mastering a specific STEM concept, updating in real-time after every interaction.
* **Item Response Theory (IRT) Integration:** Questions are served dynamically based on their mathematical difficulty and the student's demonstrated skill boundary, preventing frustration or boredom.

### 2. Affective Computing Engine (Emotion-Aware Routing)
* **Real-Time Webcam Tracking:** Integrates lightweight browser-based computer vision (via MediaPipe/TensorFlow.js) to track seven core states during learning: happy, angry, neutral, confused, bored, focused, and surprised.
* **Contextual Adaptation:** If high frustration is detected alongside repeated failures, the system automatically down-scales difficulty or triggers the AI Tutor.
* **Dataset Setup Guide:** See [EMOTION_DATASET_GUIDE.md](EMOTION_DATASET_GUIDE.md) for class-folder layout, balancing targets, and distribution guidance.
* **Deployment Visibility:** Use `python scripts/train_strict_pipeline.py` for fail-fast full training and `python scripts/prepare_emotion_deploy.py` for strict artifact/accuracy verification.

### 3. Predictive Analytics for Educators
* **At-Risk Modeling:** Utilizes tabular ML classifiers (e.g., Random Forest/XGBoost) trained on historical interaction data (time-per-question, emotion logs, scores) to flag students at high risk of failing upcoming modules.
* **Automated Pod Clustering:** Employs K-Means clustering to automatically group students into optimal learning pods based on distinct learning gaps rather than flat grades.

### 4. RAG-Based Socratic AI Tutor (NLP)
* **Context-Aware Assistance:** Replaces standard "give me the answer" AI generation. When students struggle, a Retrieval-Augmented Generation (RAG) pipeline cross-references the teacher's specific curriculum to provide Socratic hints, guiding the student to the answer conceptually.
* **Provider Fallback:** Seamlessly shifts between native Gemini pathways and OpenAI-compatible endpoints to ensure 100% uptime for AI generation.

---

## 💻 Technical Stack

### Backend (API & AI Pipelines)
* **Framework:** Python, Flask
* **Database/ORM:** PostgreSQL / SQLite, Flask-SQLAlchemy, Alembic
* **Auth:** JWT-based bearer token flow
* **AI/ML Layer:** `scikit-learn` (Predictive), `FAISS`/`ChromaDB` (Vector Store for RAG), native LLM API integrations (Gemini/OpenAI)

### Frontend (Client Experience)
* **Stack:** Vanilla HTML5, CSS3, Modern JavaScript (ES Modules)
* **Computer Vision:** `MediaPipe` / `TensorFlow.js` (Client-side emotion capture)
* **Architecture:** Zero-build-step framework for ultra-low latency and easy deployment in academic environments.

---

## ⚙️ Local Setup & Installation

### Prerequisites
* Python 3.11+
* PostgreSQL (or rely on default SQLite fallback)
* Webcam (required for the Affective Computing module)

### Option A: One-Click Windows Bootstrap
For the fastest setup on Windows environments, run the initialization script from the project root:

```cmd
start.bat
```

*This script will automatically build the `.venv`, install all Python dependencies, seed the database with test users and questions, and launch the backend server.*

### Option B: Manual Setup (Linux / macOS / Windows)

1.  **Clone and enter the repository:**
    ```bash
    git clone [https://github.com/yourusername/elevate.git](https://github.com/yourusername/elevate.git)
    cd elevate
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize the Database & Seed Data:**
    ```bash
    python seed_users.py
    ```

5.  **Run the Application:**
    ```bash
    python -m backend.app
    ```
    *Access the platform at: `http://127.0.0.1:5000`*

---

## 🔑 Configuration & Environment Variables

Create a `.env` file in the project root (use `.env.example` as template).

```env
# System Config
FLASK_ENV=development
SECRET_KEY=change-me
JWT_SECRET=change-me
ADMIN_TOKEN=dev-admin-token
DATABASE_URL=sqlite:///elevate_dev.db
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# Topic AI Service Config (local or remote)
AI_TOPIC_SERVICE_URL=http://127.0.0.1:7860
AI_TOPIC_SERVICE_TIMEOUT_SECONDS=120

# Optional auth for private Hugging Face/remote AI endpoint
# AI_TOPIC_SERVICE_TOKEN=your_token
AI_TOPIC_SERVICE_AUTH_SCHEME=Bearer
# HF_TOKEN=your_hf_token
```

For standalone topic-AI service config, copy `ai/.env.example` to `ai/.env`.
The AI service now reads both `ai/.env` and root `.env`.

Render + Hugging Face Space deploy checklist:
1. Configure backend env vars in Render.
    - `AI_TOPIC_SERVICE_URL=https://<owner>-<space>.hf.space`
    - `AI_TOPIC_SERVICE_TIMEOUT_SECONDS=45`
    - If Space auth is enabled: `AI_TOPIC_SERVICE_TOKEN` and `AI_TOPIC_SERVICE_AUTH_SCHEME=Bearer`

2. Configure AI Space cache env vars.
    - `MODELS_CACHE_DIR=/data/elevate_models_cache`
    - Optional bucket restore for faster startups:
      - `HF_BUCKET_URI=hf://buckets/<username>/<bucket-name>`
      - `HF_BUCKET_CACHE_PREFIX=elevate_models_cache`

3. Confirm readiness semantics after deployment.
    - AI Space reports ready only after model preload completes.
    - `GET /health` may return `503` with `status=starting` during startup.

4. Keep database config valid for backend deployment.
    - Set `DATABASE_URL` (or `SUPABASE_POOLER_CONNECTION_STRING`) to your PostgreSQL connection string.

---

## 👥 Seed Accounts for Testing

Use these credentials to test the role-based dashboards:
* **Admin:** `admin@elevate.com` / `admin123`
* **Teacher:** `teacher@elevate.com` / `teacher123`
* **Student:** `student@elevate.com` / `student123`

---

## 📂 System Architecture Overview

The system is decoupled into strictly managed domains:
* `/backend/routes`: Feature endpoints separated by role (`/teacher`, `/student`, `/admin`).
* `/backend/models.py`: Relational schema mapping the lifecycle from Classroom -> Assignment -> Test -> TestResult -> AnswerLog.
* `/backend/question_generator.py`: The resilient LLM generation engine with strict JSON sanitization and multi-attempt top-up logic.
* `/frontend/js/api.js`: Centralized API client handling JWT injections, 401 redirects, and cache-busting.

---

## 🛠️ Troubleshooting

* **AI Generation Fails (502/Insufficient Questions):** Verify your API key quotas. The backend `question_generator.py` enforces strict limits; if the LLM shortfalls on output, it will reject the batch to preserve data integrity.
* **Classroom Creation Fails:** Ensure the teacher account is mapped to a valid `school_id`. The seed script handles this automatically for the test accounts.