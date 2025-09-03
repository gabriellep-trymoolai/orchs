#!/bin/bash

# Script to install requirements with fallback options
echo "Installing orchestrator requirements..."

# Option 1: Try with pre-built wheels only
echo "Attempting installation with pre-built wheels only..."
pip install --only-binary=all -r requirements.txt 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Successfully installed with pre-built wheels!"
    exit 0
fi

echo "❌ Pre-built wheels failed, trying with build dependencies..."

# Option 2: Install build dependencies if needed (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing build dependencies for macOS..."
    if command -v brew &> /dev/null; then
        brew install rust postgresql
        export PATH="/opt/homebrew/opt/postgresql@14/bin:$PATH"
    else
        echo "Homebrew not found. Please install Homebrew first."
        exit 1
    fi
fi

# Option 3: Try normal installation
echo "Attempting normal installation..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Successfully installed with build from source!"
else
    echo "❌ Installation failed. Please check the error messages above."
    echo ""
    echo "Manual solutions:"
    echo "1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "2. Install PostgreSQL dev libraries: brew install postgresql"
    echo "3. Try installing packages individually:"
    echo "   pip install pydantic==2.4.2"
    echo "   pip install asyncpg==0.28.0"
    echo "   pip install -r requirements.txt"
    exit 1
fi

