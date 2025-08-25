#!/bin/bash

# Falling Blocks Viewer Launcher Script (Linux/Mac)
# Activates virtual environment and runs the falling blocks image viewer

echo "===================================================="
echo "       Falling Blocks Image Viewer Launcher"
echo "===================================================="
echo ""

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found at .venv/"
    echo "Please run setup first or create virtual environment"
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated: $(pwd)/.venv"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo ""
echo "Checking for required dependencies..."

# Check for pygame
python -c "import pygame; print('✓ pygame available')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ pygame not found. Installing..."
    pip install pygame==2.6.0
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install pygame"
        exit 1
    fi
    echo "✓ pygame installed successfully"
fi

# Check for other dependencies
python -c "import PIL; print('✓ Pillow available')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Pillow not found. Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✓ All dependencies available"
echo ""
echo "Starting Falling Blocks Image Viewer..."
echo "Press Ctrl+C to exit"
echo ""

# Run the falling blocks viewer
python falling_blocks_viewer.py

echo ""
echo "Falling Blocks Image Viewer closed."
