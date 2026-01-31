@echo off
REM Voxtral Web Application Launcher for Windows
REM Double-click this file to start the web application

echo ========================================
echo   Voxtral Transcription Web App
echo   Starting server...
echo ========================================
echo.
echo Browser will open automatically when server is ready...
echo.

REM Change to the VoxtralApp directory
cd /d "%~dp0\VoxtralApp"

REM Run the startup script (browser will be opened by Python after port is determined)
call start_web.bat
