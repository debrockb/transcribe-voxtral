#!/bin/bash
# Voxtral Web Application Launcher for macOS
# This file can be double-clicked in Finder

# Get the directory where this script is located
cd "$(dirname "$0")/VoxtralApp"

echo "🎙️  Starting Voxtral Web Application..."
echo "🌐 Browser will open automatically in 8 seconds..."
echo ""

# Open browser after delay (in background)
(sleep 8 && open http://localhost:8000) &

# Run the startup script
./start_web.sh
