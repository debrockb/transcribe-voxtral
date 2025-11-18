@echo off
REM Voxtral Web Application Launcher for Windows
REM Double-click this file to start the web application

echo ========================================
echo   Voxtral Transcription Web App
echo   Starting server...
echo ========================================
echo.
echo Browser will open automatically in 8 seconds...
echo.

REM Change to the VoxtralApp directory
cd /d "%~dp0\VoxtralApp"

REM Open browser after 8 seconds (in background)
start "" cmd /c "timeout /t 8 /nobreak >nul && start http://localhost:8000"

REM Run the startup script
call start_web.bat
