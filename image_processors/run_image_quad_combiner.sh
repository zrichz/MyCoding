#!/bin/bash
# Image Quad Combiner Launcher for Linux
# Creates horizontal combinations of 4 images scaled to fit 2560x1440

cd "$(dirname "$0")"

echo "Starting Image Quad Combiner GUI..."
echo ""
echo "Process: 4x 720x1600 images -> 2880x1600 combo -> 2560x1422 scaled output"
echo ""

# Activate virtual environment and run
source .venv/bin/activate
python image_quad_combiner.py
deactivate

echo "Image Quad Combiner closed."
echo "Press Enter to close this window..."
read
