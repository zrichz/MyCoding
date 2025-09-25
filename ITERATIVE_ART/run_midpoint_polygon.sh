#!/bin/bash
# Shell script to run iterative_midpoint_polygon.py
echo "Running Iterative Midpoint Polygon Generator..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the Python script using the virtual environment
/home/rich/MyCoding/image_processors/.venv/bin/python3 iterative_midpoint_polygon.py

echo "Script completed."
