#!/bin/bash
# Run Image Quarteriser with virtual environment

echo "Starting Image Quarteriser..."
cd /home/rich/MyCoding/image_processors
source .venv/bin/activate
python3 image_quarteriser.py
echo "Image Quarteriser closed."
