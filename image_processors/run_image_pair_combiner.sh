#!/bin/bash
# Image Pair Combiner Launcher for Linux
# Uses .venv environment

cd "$(dirname "$0")"

echo "Starting Image Pair Combiner GUI..."

# Activate virtual environment and run
source .venv/bin/activate
python image_pair_combiner.py
deactivate

echo "Image Pair Combiner closed."
echo "Press Enter to close this window..."
read
