@echo off
REM Test runner script for Voxtral application (Windows)

echo ========================================
echo   Voxtral Test Runner
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "app.py" (
    echo Error: Please run this script from the VoxtralApp directory
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "voxtral_env\" (
    echo Virtual environment not found. Creating...
    python -m venv voxtral_env
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Use the virtual environment's Python
set PYTHON_BIN=voxtral_env\Scripts\python.exe
set PIP_BIN=voxtral_env\Scripts\pip.exe

echo Installing test dependencies...
%PIP_BIN% install -q --upgrade pip
%PIP_BIN% install -q -r requirements.txt

echo.
echo Running tests...
echo.

REM Parse command line argument for test type
set TEST_TYPE=%1
if "%TEST_TYPE%"=="" set TEST_TYPE=all

if "%TEST_TYPE%"=="unit" (
    echo Running unit tests only...
    %PYTHON_BIN% -m pytest -m unit -v
) else if "%TEST_TYPE%"=="api" (
    echo Running API tests only...
    %PYTHON_BIN% -m pytest -m api -v
) else if "%TEST_TYPE%"=="integration" (
    echo Running integration tests only...
    %PYTHON_BIN% -m pytest -m integration -v
) else if "%TEST_TYPE%"=="fast" (
    echo Running fast tests only (skipping slow tests)...
    %PYTHON_BIN% -m pytest -m "not slow" -v
) else if "%TEST_TYPE%"=="coverage" (
    echo Running all tests with coverage report...
    %PYTHON_BIN% -m pytest --cov=. --cov-report=html --cov-report=term-missing -v
    echo.
    echo Coverage report generated in htmlcov\index.html
) else (
    echo Running all tests...
    %PYTHON_BIN% -m pytest -v
)

echo.
echo Tests complete!
pause
