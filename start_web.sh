#!/bin/bash
# Voxtral Web Application Startup Script (macOS/Linux)

echo "🎙️  Starting Voxtral Web Application..."
echo ""

# Activate virtual environment
if [ -d "voxtral_env" ]; then
    echo "Activating virtual environment..."
    source voxtral_env/bin/activate
else
    echo "❌ Error: Virtual environment not found!"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import flask, flask_socketio, flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required packages not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Starting web server..."
echo "📱 Access the application at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the Flask application
python app.py
