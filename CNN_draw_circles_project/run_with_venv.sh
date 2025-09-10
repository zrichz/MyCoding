#!/bin/bash
# Script to run CNN_drawCircles.py with the appropriate virtual environment

echo "Activating virtual environment..."
source /home/rich/MyCoding/textual_inversions/.venv/bin/activate

echo "Running CNN_drawCircles.py..."
cd /home/rich/MyCoding/CNN_draw_circles_project
python3 CNN_drawCircles.py

echo "Script completed!"
