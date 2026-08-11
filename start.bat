@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PROJECT_ROOT=%CD%"
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "ACTIVATE=%PROJECT_ROOT%\.venv\Scripts\activate.bat"
set "REQUIREMENTS=%PROJECT_ROOT%\requirements.txt"
set "WSL_DISTRO=Ubuntu"
set "WSL_TFJS_VENV=~/.elevate-tfjs/.venv"

if not exist "%REQUIREMENTS%" (
    echo [ERROR] requirements.txt not found: %REQUIREMENTS%
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [ENV] .venv not found. Creating Python 3.11 environment...
    py -3.11 -m venv "%PROJECT_ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] Could not create .venv with Python 3.11.
        echo [ERROR] Verify: py -3.11 --version
        exit /b 1
    )
)

"%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ENV] Existing .venv is not Python 3.11. Recreating it...
    rmdir /s /q "%PROJECT_ROOT%\.venv"
    py -3.11 -m venv "%PROJECT_ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to recreate .venv with Python 3.11.
        exit /b 1
    )
)

"%PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ENV] Bootstrapping pip...
    "%PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] Could not bootstrap pip in .venv.
        exit /b 1
    )
)

"%PYTHON%" -c "import importlib.util; mods=['flask','flask_sqlalchemy','requests','tensorflow','torch','sklearn','PIL','ydf']; raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" >nul 2>&1

"%PYTHON%" -m pip check >nul 2>&1

REM Always reconcile the environment against the single source of truth.
REM pip is incremental, so already-satisfied packages are not re-downloaded.
echo [ENV] Reconciling .venv with requirements.txt...
"%PYTHON%" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 (
    echo [ERROR] Root requirements installation failed.
    exit /b 1
)

"%PYTHON%" -m pip check
if errorlevel 1 (
    echo [ERROR] .venv still has broken package requirements.
    exit /b 1
)

"%PYTHON%" -c "import tensorflow as tf; import torch; print('[ENV] Python:', __import__('sys').version.split()[0]); print('[ENV] TensorFlow:', tf.__version__); print('[ENV] Torch:', torch.__version__)"
if errorlevel 1 (
    echo [ERROR] Core .venv imports failed.
    exit /b 1
)

REM ------------------------------------------------------------------
REM WSL emotion-conversion environment.
REM This is NOT a third Windows venv. It exists only in the WSL home.
REM ------------------------------------------------------------------

wsl.exe -d %WSL_DISTRO% -- bash -lc "python3.11 --version" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL Python 3.11 is missing in %WSL_DISTRO%.
    echo [ERROR] One-time WSL setup:
    echo         sudo add-apt-repository ppa:deadsnakes/ppa
    echo         sudo apt update
    echo         sudo apt install -y python3.11 python3.11-venv
    exit /b 1
)

wsl.exe -d %WSL_DISTRO% -- bash -lc "test -x %WSL_TFJS_VENV%/bin/python" >nul 2>&1
if errorlevel 1 (
    echo [WSL] Creating dedicated TF.js conversion venv...
    wsl.exe -d %WSL_DISTRO% -- bash -lc "python3.11 -m venv %WSL_TFJS_VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create the WSL TF.js conversion environment.
        echo [ERROR] Verify that python3.11-venv is installed in WSL.
        exit /b 1
    )
)

wsl.exe -d %WSL_DISTRO% -- bash -lc "%WSL_TFJS_VENV%/bin/python -c 'import tensorflow, tensorflowjs, tensorflow_decision_forests'" >nul 2>&1
if errorlevel 1 (
    echo [WSL] Installing pinned TF.js conversion stack...
    wsl.exe -d %WSL_DISTRO% -- bash -lc "%WSL_TFJS_VENV%/bin/python -m pip install tensorflow==2.15.0 tensorflow-decision-forests==1.8.1 tensorflowjs==4.22.0"
    if errorlevel 1 (
        echo [ERROR] WSL TF.js conversion dependencies failed to install.
        exit /b 1
    )
)

wsl.exe -d %WSL_DISTRO% -- bash -lc "%WSL_TFJS_VENV%/bin/python -c 'import tensorflow, tensorflowjs, tensorflow_decision_forests; print(tensorflow.__version__)'" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL TF.js conversion environment verification failed.
    exit /b 1
)

echo [ENV] .venv and WSL TF.js conversion environment are ready.

call "%ACTIVATE%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to activate .venv.
    exit /b 1
)

echo [BOOTSTRAP] Starting Elevate...
"%PYTHON%" -m scripts.bootstrap.bootstrap %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
