#!/bin/bash
# Run BLIP Image Captioner with venv_mycoding virtual environment

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the venv_mycoding virtual environment
VENV_PATH="$SCRIPT_DIR/venv_mycoding"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first or install dependencies globally."
    exit 1
fi

# Activate virtual environment and run the script
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Running BLIP Image Captioner..."
python "$SCRIPT_DIR/BLIP_image_captioner.py"

echo "Done."
