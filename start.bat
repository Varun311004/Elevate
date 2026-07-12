@echo off
setlocal

REM ==================================================
REM Elevate Startup Launcher
REM Responsibility:
REM   - Activate project virtual environment
REM   - Launch Python bootstrap orchestrator
REM ==================================================

cd /d "%~dp0"

set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "ACTIVATE=%CD%\.venv\Scripts\activate.bat"

if not exist "%PYTHON%" (
    echo.
    echo [ERROR] Project virtual environment not found.
    echo [ERROR] Expected:
    echo         %PYTHON%
    echo.
    echo Create the environment using Python 3.12 before starting Elevate.
    exit /b 1
)

call "%ACTIVATE%"

"%PYTHON%" -m scripts.bootstrap.bootstrap %*

set EXIT_CODE=%ERRORLEVEL%

endlocal & exit /b %EXIT_CODE%