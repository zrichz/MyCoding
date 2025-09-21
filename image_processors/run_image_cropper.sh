#!/bin/bash

# Interactive Image Cropper - Easy Launch Script for Linux/Mac
# This shell script activates the virtual environment and runs the cropper application

echo "===================================================="
echo "       Interactive Image Cropper Launcher"
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

# Check if Pillow is installed
echo "Checking for required dependencies..."
if ! python -c "import PIL; print('Pillow version:', PIL.__version__)" 2>/dev/null; then
    echo "WARNING: Required dependencies not found! Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies!"
        echo
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Run the image cropper
echo
echo "Starting Interactive Image Cropper..."
echo "Press Ctrl+C to exit"
echo
python interactive_image_cropper.py

# Deactivate virtual environment when done
deactivate

echo
echo "Interactive Image Cropper has closed."
read -p "Press Enter to exit..."
