#!/bin/bash

# Face Keypoint Detection Launcher
# Activates the MyCoding virtual environment and launches the Gradio app

echo "Face Keypoint Detection Launcher"
echo "================================"

# Get the script directory and navigate to MyCoding root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYCODING_DIR="$(dirname "$SCRIPT_DIR")"

echo "Script directory: $SCRIPT_DIR"
echo "MyCoding directory: $MYCODING_DIR"

# Check for virtual environment
VENV_PATH="$MYCODING_DIR/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Using system Python..."
fi

# Navigate to the script directory
cd "$SCRIPT_DIR"

# Check if the script exists
SCRIPT_NAME="face_keypoint_detector.py"
if [ -f "$SCRIPT_NAME" ]; then
    echo "Launching Face Keypoint Detection app..."
    echo "----------------------------------------"
    
    # Install required packages if missing
    echo "Checking dependencies..."
    python -c "import gradio, cv2, mediapipe, PIL, numpy" 2>/dev/null || {
        echo "Installing missing dependencies..."
        pip install gradio opencv-python mediapipe pillow numpy
    }
    
    # Run the script
    python "$SCRIPT_NAME"
    
    echo "----------------------------------------"
    echo "Application closed."
else
    echo "Error: $SCRIPT_NAME not found in $SCRIPT_DIR"
    exit 1
fi

# Keep terminal open if run from file manager
read -p "Press Enter to exit..."
