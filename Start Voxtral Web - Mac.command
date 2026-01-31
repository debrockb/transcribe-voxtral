#!/bin/bash
# Voxtral Web Application Launcher for macOS
# This file can be double-clicked in Finder

# Get the directory where this script is located
cd "$(dirname "$0")/VoxtralApp"

echo "🎙️  Starting Voxtral Web Application..."
echo "🌐 Browser will open automatically when server is ready..."
echo ""

# Run the startup script (browser will be opened by Python after port is determined)
./start_web.sh
