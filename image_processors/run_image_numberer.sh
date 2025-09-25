#!/bin/bash

# Image Numberer 1440x3200 - Easy Launch Script for Linux/Mac
# This shell script activates the virtual environment and runs the image numberer application

echo "===================================================="
echo "       Image Numberer 1440x3200 Launcher"
echo "===================================================="
echo

# Change to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f "../.venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please make sure the .venv folder exists in the parent directory."
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
source "../.venv/bin/activate"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment!"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Virtual environment activated successfully."
echo

# Check if the Python script exists
if [ ! -f "image_numberer_1440x3200.py" ]; then
    echo "ERROR: image_numberer_1440x3200.py not found!"
    echo "Please make sure the script is in the current directory."
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the image numberer application
echo "Starting Image Numberer 1440x3200..."
echo
echo "This application will:"
echo "- Process 1440x3200 pixel images only"
echo "- Add sequential numbers (0001, 0002, etc.) to each corner"
echo "- Save as highest quality JPEG"
echo "- Skip images that are not exactly 1440x3200 pixels"
echo

python3 image_numberer_1440x3200.py

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: The image numberer application encountered an error!"
    echo
else
    echo
    echo "Image numberer application completed successfully."
fi

echo
read -p "Press Enter to exit..."