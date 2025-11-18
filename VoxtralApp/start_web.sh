#!/bin/bash
# Voxtral Web Application Startup Script (macOS/Linux)

echo "🎙️  Starting Voxtral Web Application..."
echo ""

# Check if virtual environment exists
if [ ! -d "voxtral_env" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

# Use the virtual environment's Python directly
PYTHON_BIN="voxtral_env/bin/python3"

# Check if required packages are installed
echo "Checking dependencies..."
$PYTHON_BIN -c "import flask, flask_socketio, flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required packages not installed!"
    echo "Installing dependencies..."
    voxtral_env/bin/pip3 install -r requirements.txt
fi

echo ""
echo "✅ Starting web server..."
echo "📱 Access the application at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the Flask application using venv's python3
$PYTHON_BIN app.py
