#!/bin/bash
# Image Section Matcher Launcher for Linux
# Uses .venv environment with PIL and numpy

cd "$(dirname "$0")"

echo "Starting Image Section Matcher..."

# Use the local .venv in image_processors directory
if [ -f ".venv/bin/activate" ]; then
    echo "Using local virtual environment..."
    source .venv/bin/activate
    python image_section_matcher.py
    deactivate
else
    echo "Virtual environment not found at .venv/"
    echo "Creating virtual environment and installing packages..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy pillow
    python image_section_matcher.py
    deactivate
fi

echo "Image Section Matcher closed."
echo "Press Enter to close this window..."
read
