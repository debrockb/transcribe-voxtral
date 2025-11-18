#!/bin/bash
# Test runner script for Voxtral application

set -e

echo "🧪 Voxtral Test Runner"
echo "====================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the VoxtralApp directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "voxtral_env" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv voxtral_env
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source voxtral_env/bin/activate

# Install/update test dependencies
echo "📥 Installing test dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Parse command line arguments
TEST_TYPE=${1:-all}
COVERAGE=${2:-yes}

echo ""
echo "🏃 Running tests..."
echo ""

case $TEST_TYPE in
    unit)
        echo "Running unit tests only..."
        pytest -m unit -v
        ;;
    api)
        echo "Running API tests only..."
        pytest -m api -v
        ;;
    integration)
        echo "Running integration tests only..."
        pytest -m integration -v
        ;;
    fast)
        echo "Running fast tests only (skipping slow tests)..."
        pytest -m "not slow" -v
        ;;
    coverage)
        echo "Running all tests with coverage report..."
        pytest --cov=. --cov-report=html --cov-report=term-missing -v
        echo ""
        echo "📊 Coverage report generated in htmlcov/index.html"
        ;;
    all|*)
        if [ "$COVERAGE" = "yes" ]; then
            echo "Running all tests with coverage..."
            pytest --cov=. --cov-report=term-missing -v
        else
            echo "Running all tests..."
            pytest -v
        fi
        ;;
esac

echo ""
echo "✅ Tests complete!"
