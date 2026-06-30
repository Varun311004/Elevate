@echo off
setlocal enabledelayedexpansion
title Elevate Learning Platform

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "AI_SERVICE_URL=http://127.0.0.1:7860"
set "AI_START_SCRIPT=ai\start.bat"
set "AI_LOG_FILE=%cd%\ai\topic_ai_service.log"
set "AI_HEALTH_TIMEOUT_SECONDS=45"
if not "%ELEVATE_AI_HEALTH_TIMEOUT_SECONDS%"=="" set "AI_HEALTH_TIMEOUT_SECONDS=%ELEVATE_AI_HEALTH_TIMEOUT_SECONDS%"
if not defined AI_HEALTH_TIMEOUT_SECONDS set "AI_HEALTH_TIMEOUT_SECONDS=45"
for /f "delims=0123456789" %%A in ("%AI_HEALTH_TIMEOUT_SECONDS%") do set "AI_HEALTH_TIMEOUT_SECONDS=45"
if %AI_HEALTH_TIMEOUT_SECONDS% LSS 5 set "AI_HEALTH_TIMEOUT_SECONDS=5"
if %AI_HEALTH_TIMEOUT_SECONDS% GTR 300 set "AI_HEALTH_TIMEOUT_SECONDS=300"
set /a AI_HEALTH_MAX_LOOPS=AI_HEALTH_TIMEOUT_SECONDS*2
set "AI_DO_WARMUP=0"
if /I "%ELEVATE_AI_WARMUP%"=="1" set "AI_DO_WARMUP=1"

echo.
echo  ================================================
echo   Elevate - AI Adaptive Learning Platform
echo  ================================================
echo.

:: ── Check system Python ──────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.11+ from https://python.org and add it to PATH.
    echo.
    pause
    exit /b 1
)

:: ── Validate/repair virtual environment ─────────────────────────────────────
set "NEEDS_SETUP=0"
set "NEEDS_PIP_SYNC=0"
if not exist "%VENV_PY%" (
    set "NEEDS_SETUP=1"
) else (
    "%VENV_PY%" -c "import pip._internal; from flask import Flask; import flask_sqlalchemy" >nul 2>&1
    if errorlevel 1 (
        echo  [WARN] Existing virtual environment is broken or incomplete.
        set "NEEDS_SETUP=1"
    ) else (
        "%VENV_PY%" -c "import pypdf, docx, boto3" >nul 2>&1
        if errorlevel 1 (
            echo  [WARN] Runtime dependencies are missing for document upload.
            set "NEEDS_PIP_SYNC=1"
        )
    )
)

if "%NEEDS_SETUP%"=="1" (
    echo  [SETUP] Preparing virtual environment...
    echo.

    if exist ".venv" (
        echo  [SETUP] Removing previous broken .venv...
        rmdir /s /q ".venv"
    )

    echo  [1/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )

    echo  [2/4] Bootstrapping pip and installing dependencies...
    "%VENV_PY%" -m ensurepip --upgrade >nul 2>&1
    "%VENV_PY%" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] pip bootstrap failed in .venv.
        pause
        exit /b 1
    )

    "%VENV_PY%" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 (
        echo  [ERROR] Failed to upgrade pip/setuptools/wheel.
        pause
        exit /b 1
    )

    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )

    "%VENV_PY%" -c "import pip._internal; from flask import Flask; import flask_sqlalchemy" >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] Package verification failed after installation.
        echo  [HINT] The environment has inconsistent package files. Re-run start.bat.
        pause
        exit /b 1
    )

    echo  [3/4] Verifying installed runtime...
    if not exist "scripts\startup_healthcheck.py" (
        echo  [ERROR] scripts\startup_healthcheck.py is missing.
        pause
        exit /b 1
    )

    echo.
    echo  [SETUP] Setup complete!
) else (
    echo  [INFO] Virtual environment found and healthy.
)

if "%NEEDS_SETUP%"=="0" if "%NEEDS_PIP_SYNC%"=="1" (
    echo  [SETUP] Syncing missing dependencies from requirements.txt ...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Dependency sync failed.
        pause
        exit /b 1
    )

    "%VENV_PY%" -c "import pypdf, docx, boto3" >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] Dependency verification failed after sync.
        pause
        exit /b 1
    )
)

echo  [CHECK] Running startup healthcheck...
"%VENV_PY%" scripts\startup_healthcheck.py
if errorlevel 1 (
    echo  [ERROR] Startup health check failed.
    pause
    exit /b 1
)

if /I "%ELEVATE_SKIP_AI%"=="1" (
    echo  [AI] Skipping Topic AI startup ^(ELEVATE_SKIP_AI=1^).
) else (
    set "AI_ALREADY_RUNNING=0"
    netstat -ano | findstr /I ":7860" | findstr /I "LISTENING" >nul 2>&1
    if not errorlevel 1 set "AI_ALREADY_RUNNING=1"

    if "!AI_ALREADY_RUNNING!"=="1" (
        echo  [AI] Topic AI service already running.
    ) else (
        if not exist "%AI_START_SCRIPT%" (
            echo  [WARN] %AI_START_SCRIPT% is missing.
            echo  [WARN] Skipping Topic AI startup. Teacher AI generation may be unavailable.
        ) else (
            echo  [AI] Ensuring Topic AI environment via %AI_START_SCRIPT% ...
            call "%AI_START_SCRIPT%" --ensure-env
            if errorlevel 1 (
                echo  [WARN] Topic AI environment setup failed.
                echo  [WARN] Skipping Topic AI startup. Teacher AI generation may be unavailable.
            ) else (
                echo  [AI] Starting Topic AI service at %AI_SERVICE_URL% in background ...
                if exist "%AI_LOG_FILE%" del /q "%AI_LOG_FILE%" >nul 2>&1
                start "" /B cmd /c "cd /d ""%cd%\ai"" && call start.bat --serve >> ""%AI_LOG_FILE%"" 2>&1"
                echo  [AI] Topic AI startup triggered. Check %AI_LOG_FILE% if generation fails.
            )
        )
    )
)

:: ── Launch ───────────────────────────────────────────────────────────────────
echo.
echo  ------------------------------------------------
echo   App running at:  http://127.0.0.1:5000
echo   Topic AI API :   %AI_SERVICE_URL%
echo  ------------------------------------------------
echo   Student  :  student@elevate.com  /  student123
echo   Teacher  :  teacher@elevate.com  /  teacher123
echo   Admin    :  admin@elevate.com    /  admin123
echo  ------------------------------------------------
echo   Press Ctrl+C to stop.
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:5000"

"%VENV_PY%" -m backend.run

pause
