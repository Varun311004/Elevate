# Elevate: AI-Powered Adaptive STEM Learning Platform

Elevate is a full-stack learning platform that helps students practise STEM topics through adaptive questions and personalized progress tracking. It combines emotion-aware learning, teacher-managed assessments, AI-assisted question generation, and learning analytics in one application.

## What Is Elevate?

Students can sign in, practise Science, Technology, Engineering, and Mathematics, take tests and assignments, and review their progress. Browser-based facial-expression analysis estimates seven learning-related emotion classes to support the adaptive-learning experience. The platform also includes progress reporting and at-risk learner insights.

Teachers can manage classrooms, students, question banks, tests, assignments, and curriculum documents. The AI service generates topic-based multiple-choice questions and supports document processing for teacher-created assessments. Admin workflows support school and user management.

The application is split into a Flask backend, a static HTML/CSS/JavaScript frontend, a separate FastAPI AI service, and a PostgreSQL-compatible database. The normal local workflow is orchestrated by the root `start.bat` script.

## Core Features

- Student, teacher, and admin sign-in with role-based workflows.
- STEM practice questions filtered by grade, subject, topic, and difficulty.
- Tests, assignments, progress tracking, reports, and teacher analytics.
- Browser-based emotion detection with seven classes: happy, bored, focused, confused, neutral, angry, and surprised.
- AI-assisted topic-based MCQ generation and scoring.
- Teacher curriculum-document ingestion for retrieval-assisted question generation.
- Bayesian Knowledge Tracing (BKT) and at-risk learner prediction.
- School-aware page routing and database migrations.

## Technology Stack

| Area | Technologies |
|---|---|
| Frontend | HTML5, CSS3, JavaScript ES modules, Bootstrap 5, Font Awesome |
| Backend API | Python 3.11, Flask, Flask-SQLAlchemy, Flask-CORS, PyJWT, Alembic |
| AI service | FastAPI, Uvicorn, Hugging Face Hub, llama.cpp, quantized Qwen 2.5 model |
| Emotion detection | TensorFlow/Keras, MobileNetV2, TensorFlow.js, face-api.js |
| Predictive analytics | scikit-learn, PyTorch, NumPy, SciPy |
| Database | PostgreSQL, including Supabase PostgreSQL |
| Document retrieval | PDF/DOCX/TXT parsing, Python cosine-similarity retrieval; optional PostgreSQL `pgvector` |
| Deployment configuration | Render for frontend/backend; Hugging Face Spaces for the AI service |

The default document-retrieval path uses custom Python cosine similarity. FAISS and ChromaDB are not part of the current implementation.

## Key Project Structure

```text
Elevate/
├── ai/                         FastAPI topic-MCQ and document-processing service
│   ├── app.py                  AI service API and readiness checks
│   ├── mcq/                    Question generation and validation logic
│   ├── models/                 Local LLM loader
│   ├── requirements.txt        AI-service dependencies
│   └── start.bat               Windows AI-service setup and launch
├── backend/
│   ├── app.py                  Flask app factory, routes registration, health endpoint
│   ├── config.py               Environment and database configuration
│   ├── models.py               SQLAlchemy data models
│   ├── routes/                 Authentication, learning, progress, teacher and admin APIs
│   ├── migrations/             Alembic database migrations
│   ├── rag_service.py          Curriculum-document ingestion and retrieval
│   └── ai_models/              Emotion model and metadata
├── frontend/
│   ├── *.html                  Sign-in, student, teacher, admin and learning pages
│   ├── css/                    Stylesheets
│   └── js/                     Frontend modules, API client and emotion detection
│       └── emotion_tfjs/       Browser emotion-model files and weights
├── scripts/
│   ├── bootstrap/              Local service orchestration and slug-aware frontend server
│   ├── database/               Database migrations and development-user seeding
│   ├── datasets/               Interaction-data preparation
│   └── startup_healthcheck.py  Startup database, data and model-artifact checks
├── .env.example                Environment-variable template (no production secrets)
├── requirements.txt            Main backend and ML dependencies
├── render.yaml                 Render deployment configuration
└── start.bat                   Recommended local startup command
```

## Run the Project Locally

The supported local startup path is **`start.bat` in the repository root**. Run it from Windows Command Prompt or PowerShell; do not start the backend or frontend separately first.

### Prerequisites

- Windows 10/11, Git, and a modern browser.
- Python 3.11 available through the Windows Python launcher (`py -3.11`).
- Windows Subsystem for Linux (WSL) with an Ubuntu distribution.
- Python 3.11 and `python3.11-venv` installed inside Ubuntu WSL.
- An internet connection for package installation and the first download of the AI model.

Install Ubuntu WSL if it is not already available (run PowerShell as Administrator, then follow the prompts):

```powershell
wsl --install -d Ubuntu
```

Open Ubuntu and install the required Python environment support:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

Confirm the required interpreters are available:

```powershell
py -3.11 --version
wsl.exe -d Ubuntu -- bash -lc "python3.11 --version"
```

### 1. Clone the Repository

```powershell
git clone https://github.com/Varun311004/Elevate.git
cd Elevate
```

### 2. Create and Configure `.env`

Create your local environment file from the template:

```powershell
copy .env.example .env
```

Open `.env` and configure the local values. For a complete local startup, use PostgreSQL because the automatic practice-question seeder explicitly requires PostgreSQL/Supabase. First create a local database named `elevate` (using `psql`, pgAdmin, or another PostgreSQL client), then set the relevant entries like this:

```env
FLASK_ENV=development
SECRET_KEY=replace-with-a-local-secret
JWT_SECRET=replace-with-a-local-jwt-secret
ADMIN_TOKEN=replace-with-a-local-admin-token
DATABASE_URL=postgresql://postgres:YOUR_LOCAL_PASSWORD@127.0.0.1:5432/elevate
CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
AI_TOPIC_SERVICE_URL=http://127.0.0.1:7860
RAG_DOCUMENT_STORAGE=local
```

Replace `YOUR_LOCAL_PASSWORD` with the password for your local PostgreSQL user. If the password contains URL-reserved characters, URL-encode them in the connection string. You can use Supabase instead, but set `DATABASE_URL` to a valid PostgreSQL DSN that the migration runner and question seeder can both read. Keep database credentials private. Although the application configuration has a SQLite option, SQLite will not work with the automated practice-question seeder used by a full fresh startup.

**Question-bank seeding on a fresh/empty database needs provider credentials.** Configure a valid `GEMINI_API_KEY` and `GEMINI_VALIDATOR_API_KEY`; `GROQ_API_KEY` can also be set when you want the seeder to use Groq generation models. The validator key is read separately, so add that variable even if it uses the same key value as your Gemini key. Provider quotas and availability can affect seeding. Never commit `.env` or real credentials to GitHub.

### 3. Start Elevate

From the repository root, run:

```powershell
start.bat
```

The startup script prepares or repairs the main Python virtual environment, installs the pinned/project dependencies, verifies the WSL TensorFlow.js conversion environment, and runs the bootstrap workflow. The bootstrap applies database migrations, checks and seeds development data when needed, verifies interaction data and model artifacts, and then launches the AI service, Flask backend, and frontend. It waits for the services to respond before opening the application in your browser.

The first run can take longer while packages and the local quantized language model are downloaded and installed. Later runs reuse healthy environments and existing artifacts where possible. Do not close the startup terminals while you are using the application.

### 4. Local URLs

| Service | URL |
|---|---|
| Frontend | `http://127.0.0.1:8000` |
| Backend health | `http://127.0.0.1:5000/health` |
| Backend API | `http://127.0.0.1:5000/api` |
| AI service health | `http://127.0.0.1:7860/health` |

The AI health endpoint may report that the service is starting while its model loads. Wait until it reports ready.

### Development Login Accounts

When the database has no users, the bootstrap process seeds these development accounts:

| Role | Email | Password |
|---|---|---|
| Admin | `admin@elevate.com` | `admin123` |
| Teacher | `teacher@elevate.com` | `teacher123` |
| Student | `student@elevate.com` | `student123` |

These are **local development credentials only**. Never use them for a public or production deployment, and change/remove them before exposing a new environment.

## Troubleshooting

- **Python/WSL check fails:** verify `py -3.11 --version`, confirm your WSL distribution is named `Ubuntu`, and install `python3.11-venv` inside WSL.
- **Database migration or connection fails:** check `DATABASE_URL`, verify PostgreSQL is running and reachable, confirm the `elevate` database exists, and make sure the URL uses the PostgreSQL DSN format.
- **Question-bank seeding fails:** check `GEMINI_API_KEY`, `GEMINI_VALIDATOR_API_KEY`, and, if used, `GROQ_API_KEY`. Provider quotas, unsupported models, or network issues can interrupt generation/validation.
- **AI service does not become ready:** allow the first model download to finish, check internet access, and review the AI-service startup output for dependency or model-loading errors.
- **A port is already in use:** stop the existing process using port `5000`, `8000`, or `7860`, then run `start.bat` again.
- **Emotion-model artifacts are missing:** the startup check expects the committed backend model/metadata and TensorFlow.js model/weight files. Restore those files from the repository before trying to regenerate artifacts.

## Deployment Overview

The deployment configuration separates the Flask backend and static frontend on Render from the FastAPI AI service on Hugging Face Spaces. Supabase PostgreSQL can be used for persistent production data. Production deployments require their own environment variables, database credentials, API keys, and secret values configured in the hosting dashboards; the local `.env` setup above is not a production configuration.

## Developed By

**Varun Joshi**  
Computer Engineering | AI and Full-Stack Development  
GitHub: [Varun311004](https://github.com/Varun311004)
