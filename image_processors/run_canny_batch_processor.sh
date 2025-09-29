#!/bin/bash
# Canny Batch Processor Launcher for Linux
# Uses .venv environment with OpenCV, PIL and numpy

cd "$(dirname "$0")"

echo "Starting Canny Batch Processor..."

# Use the local .venv in image_processors directory
if [ -f ".venv/bin/activate" ]; then
    echo "Using local virtual environment..."
    source .venv/bin/activate
    
    # Install opencv-python if not already installed
    python -c "import cv2" 2>/dev/null || {
        echo "Installing opencv-python in virtual environment..."
        pip install opencv-python
    }
    
    python canny_batch_processor.py
    deactivate
else
    echo "Virtual environment not found at .venv/"
    echo "Creating virtual environment and installing packages..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install opencv-python numpy pillow
    python canny_batch_processor.py
    deactivate
fi

echo "Canny Batch Processor closed."
echo "Press Enter to close this window..."
read
