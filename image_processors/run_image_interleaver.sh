#!/bin/bash
# Image Interleaver Launcher for Linux
# PIL-only version - no external dependencies required

cd "$(dirname "$0")"

echo "Starting Image Interleaver GUI..."

python3 image_interleaver.py

echo "Image Interleaver closed."
echo "Press Enter to close this window..."
read
