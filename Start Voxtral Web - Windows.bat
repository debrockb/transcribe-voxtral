@echo off
REM Voxtral Web Application Launcher for Windows
REM Double-click this file to start the web application

echo ========================================
echo   Voxtral Transcription Web App
echo   Starting server...
echo ========================================
echo.
echo Browser will open with startup page...
echo.

REM Change to the VoxtralApp directory
cd /d "%~dp0\VoxtralApp"

REM Open startup page immediately (shows loading state while app starts)
start "" "%~dp0\VoxtralApp\static\startup.html"

REM Run the startup script
call start_web.bat
