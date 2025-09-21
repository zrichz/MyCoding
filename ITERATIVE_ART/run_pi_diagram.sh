#!/bin/bash
# Shell script to run pi_diagram.py
echo "Running Pi Diagram Script..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the Python script using the virtual environment
/home/rich/MyCoding/image_processors/.venv/bin/python3 pi_diagram.py

echo "Script completed."
