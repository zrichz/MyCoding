#!/bin/bash
# Image Painter Launcher for Linux
# Uses .venv environment with PIL and numpy

cd "$(dirname "$0")"

echo "Starting Image Painter GUI..."

# Try to activate virtual environment, fallback to system python if not found
if [ -f ".venv/bin/activate" ]; then
    echo "Using virtual environment..."
    source .venv/bin/activate
    python image_painter.py
    deactivate
elif [ -f "../.venv/bin/activate" ]; then
    echo "Using parent directory virtual environment..."
    source ../.venv/bin/activate
    python image_painter.py
    deactivate
else
    echo "No virtual environment found, using system Python..."
    echo "Installing required packages if needed..."
    pip3 install --user numpy pillow 2>/dev/null || echo "Packages already installed or install failed"
    python3 image_painter.py
fi

echo "Image Painter closed."
echo "Press Enter to close this window..."
read
