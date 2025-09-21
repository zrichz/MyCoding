#!/bin/bash

# Interactive Image Cropper - Easy Launch Script
# This shell script activates the virtual environment and runs the cropper application

echo "Starting Interactive Image Cropper..."
echo

# Check if virtual environment exists
if [ ! -f "../.venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please make sure the .venv folder exists in the parent directory."
    echo "You may need to create it first by running:"
    echo "  python -m venv ../.venv"
    echo "  source ../.venv/bin/activate"
    echo "  pip install Pillow"
    echo
    read -p "Press Enter to continue..."
    exit 1
fi

# Check if the main script exists
if [ ! -f "interactive_image_cropper.py" ]; then
    echo "ERROR: interactive_image_cropper.py not found!"
    echo "Please make sure you're running this script from the correct directory."
    echo
    read -p "Press Enter to continue..."
    exit 1
fi

# Activate virtual environment and run the application
echo "Activating virtual environment..."
source ../.venv/bin/activate

echo "Running Interactive Image Cropper GUI..."
echo
python interactive_image_cropper.py

# Check if there was an error
if [ $? -ne 0 ]; then
    echo
    echo "An error occurred while running the application."
    read -p "Press Enter to continue..."
fi

echo
echo "Application closed."
