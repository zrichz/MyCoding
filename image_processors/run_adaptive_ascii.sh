#!/bin/bash
# Adaptive ASCII Art Generator Launcher
# Uses existing virtual environment to run the adaptive ASCII art tool

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -f "../../.venv/bin/python" ]; then
    echo "Using existing virtual environment..."
    ../../.venv/bin/python adaptive_ascii_art.py
elif [ -f "../.venv/bin/python" ]; then
    echo "Using existing virtual environment..."
    ../.venv/bin/python adaptive_ascii_art.py
else
    echo "Virtual environment not found. Please ensure .venv exists."
    echo "Trying system Python..."
    python3 adaptive_ascii_art.py || python adaptive_ascii_art.py
fi

if [ $? -ne 0 ]; then
    echo
    echo "Error running the application."
    echo "Make sure the virtual environment is set up and contains the required packages:"
    echo "- numpy"
    echo "- pillow"
    echo
    read -p "Press Enter to continue..."
fi