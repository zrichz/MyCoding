#!/bin/bash
# Image Interleaver Launcher for Linux
# Uses .venv environment

cd "$(dirname "$0")"

echo "Starting Image Interleaver GUI..."

# Activate virtual environment and run
source .venv/bin/activate
python image_interleaver.py
deactivate

echo "Image Interleaver closed."
echo "Press Enter to close this window..."
read
