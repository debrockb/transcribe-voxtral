@echo off
REM Voxtral Web Application Startup Script (Windows)

echo Starting Voxtral Web Application...
echo.

REM Check if virtual environment exists
if not exist "voxtral_env\" (
    echo Error: Virtual environment not found!
    echo Please run setup first to create the environment.
    echo.
    echo Creating virtual environment...
    python -m venv voxtral_env
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Use the virtual environment's Python directly
set PYTHON_BIN=voxtral_env\Scripts\python.exe
set PIP_BIN=voxtral_env\Scripts\pip.exe

REM Check if Flask is installed
%PYTHON_BIN% -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    %PIP_BIN% install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Starting web server...
echo Access the application at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Start the Flask application
%PYTHON_BIN% app.py

REM Only pause if there was an error (allows update to proceed on clean exit)
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
