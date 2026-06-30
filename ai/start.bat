@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "MODE=serve"
if /I "%~1"=="--ensure-env" set "MODE=ensure"
if /I "%~1"=="--serve" set "MODE=serve"

set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"

set "AI_VENV=%PROJECT_ROOT%\.venv-ai"
set "AI_PY=%AI_VENV%\Scripts\python.exe"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "AI_BOOTSTRAP_PY=python"

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "AI_BOOTSTRAP_PY=py -3.12"

set "NEEDS_CREATE=0"
if not exist "%AI_PY%" set "NEEDS_CREATE=1"

if "%NEEDS_CREATE%"=="0" (
	"%AI_PY%" --version 2>nul | findstr /R /C:"^Python 3\.11\." /C:"^Python 3\.12\." >nul
	if errorlevel 1 (
		if /I not "%AI_BOOTSTRAP_PY%"=="python" (
			echo [AI-SETUP] Rebuilding .venv-ai with Python 3.12 for llama-cpp compatibility...
			if exist "%AI_VENV%" rmdir /s /q "%AI_VENV%"
			set "NEEDS_CREATE=1"
		)
	)
)

if "%NEEDS_CREATE%"=="1" (
	echo [AI-SETUP] Creating .venv-ai ...
	if exist "%AI_VENV%" rmdir /s /q "%AI_VENV%"

	%AI_BOOTSTRAP_PY% -m venv "%AI_VENV%"
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: failed to create .venv-ai
		exit /b 1
	)

	"%AI_PY%" -m ensurepip --upgrade >nul 2>&1
)

"%AI_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
	"%AI_PY%" -m ensurepip --upgrade >nul 2>&1
)

"%AI_PY%" -c "import uvicorn, fastapi, requests, dotenv" >nul 2>&1
if errorlevel 1 (
	echo [AI-SETUP] Installing AI dependencies...
	"%AI_PY%" -m pip install --upgrade pip setuptools wheel
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: failed to bootstrap pip tooling
		exit /b 1
	)

	"%AI_PY%" -m pip install -r requirements.txt
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: failed to install ai\requirements.txt
		exit /b 1
	)
)

"%AI_PY%" -c "import uvicorn, fastapi, requests, dotenv, torch, transformers, peft, accelerate, datasets, safetensors" >nul 2>&1
if errorlevel 1 (
	echo [AI-SETUP] Installing missing ML dependencies...
	"%AI_PY%" -m pip install torch transformers peft datasets safetensors
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: failed to install required ML dependencies
		exit /b 1
	)

	"%AI_PY%" -c "import torch, transformers, peft, accelerate, datasets, safetensors" >nul 2>&1
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: AI dependency verification failed
		exit /b 1
	)
)

"%AI_PY%" -c "import llama_cpp, huggingface_hub" >nul 2>&1
if errorlevel 1 (
	"%AI_PY%" --version 2>nul | findstr /R /C:"^Python 3\.11\." /C:"^Python 3\.12\." >nul
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: Missing llama_cpp dependency in .venv-ai.
		echo [AI-SETUP] HINT: llama-cpp-python wheels are typically available for Python 3.11/3.12.
		echo [AI-SETUP] HINT: Recreate .venv-ai with Python 3.11/3.12 and run ai\start.bat --ensure-env.
		exit /b 1
	)

	echo [AI-SETUP] Installing missing llama.cpp dependencies...
	"%AI_PY%" -m pip install llama-cpp-python==0.2.77 huggingface-hub==0.23.2
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: failed to install llama-cpp-python in .venv-ai.
		echo [AI-SETUP] HINT: Use Python 3.11/3.12 and re-run ai\start.bat --ensure-env.
		exit /b 1
	)

	"%AI_PY%" -c "import llama_cpp, huggingface_hub" >nul 2>&1
	if errorlevel 1 (
		echo [AI-SETUP] ERROR: Missing llama_cpp dependency in .venv-ai after install attempt.
		echo [AI-SETUP] HINT: Use Python 3.11/3.12 and re-run ai\start.bat --ensure-env.
		exit /b 1
	)
)

if /I "%MODE%"=="ensure" (
	echo [AI-SETUP] .venv-ai ready.
	exit /b 0
)

if "%PORT%"=="" set PORT=7860

echo [AI] Running Topic AI service on http://127.0.0.1:%PORT%
"%AI_PY%" -m uvicorn app:app --host 127.0.0.1 --port %PORT%
