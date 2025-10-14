#!/bin/bash
# Run Iterative Midpoint Polygon Generator with image_processors virtual environment

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the image_processors virtual environment
VENV_PATH="$SCRIPT_DIR/../../image_processors/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first or install dependencies globally."
    exit 1
fi

# Activate virtual environment and run the script
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Running Iterative Midpoint Polygon Generator..."
python "$SCRIPT_DIR/../iterative_midpoint_polygon.py"

echo "Done."
