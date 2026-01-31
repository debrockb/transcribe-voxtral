#!/bin/bash
# Voxtral Setup Script for macOS

echo "🚀 Starting project setup..."

if ! command -v brew &> /dev/null; then
    echo "🍺 Homebrew not found. Installing now..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew is already installed."
fi

echo "🐍 Installing Python 3.11 and CMake (Homebrew will skip if already installed)..."
brew install python@3.11 cmake

if [ ! -d "voxtral_env" ]; then
    echo "🛠️ Creating Python virtual environment..."
    python3.11 -m venv voxtral_env
else
    echo "✅ Python virtual environment already exists."
fi

echo "📦 Activating environment and installing packages from requirements.txt..."
source voxtral_env/bin/activate
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete! You can now run the application with ./start.sh"