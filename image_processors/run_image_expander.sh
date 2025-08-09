#!/bin/bash

# Image Expander 720x1600 - Easy Launch Script for Linux/Mac
# This shell script activates the virtual environment and runs the image expander application

echo "===================================================="
echo "       Image Expander 720x1600 Launcher"
echo "===================================================="
echo

# Change to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please make sure the .venv folder exists in the current directory."
    echo
    echo "To create a virtual environment, run:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ".venv/bin/activate"

# Check if activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Failed to activate virtual environment!"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"
echo

# Check if required dependencies are installed
echo "Checking for required dependencies..."
if ! python -c "import PIL, numpy, scipy; print('All dependencies available')" 2>/dev/null; then
    echo "WARNING: Required dependencies not found! Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies!"
        echo
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Run the image expander
echo
echo "Starting Image Expander 720x1600..."
echo "Press Ctrl+C to exit"
echo
python image_expander_720x1600.py

# Deactivate virtual environment when done
deactivate

echo
echo "Image Expander 720x1600 has closed."
read -p "Press Enter to exit..."
