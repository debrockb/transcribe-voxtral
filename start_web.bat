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

REM Activate virtual environment
echo Activating virtual environment...
call voxtral_env\Scripts\activate.bat

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Starting web server...
echo Access the application at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Start the Flask application
python app.py

pause
