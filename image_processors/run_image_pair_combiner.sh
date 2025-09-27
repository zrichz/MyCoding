#!/bin/bash
# Image Pair Combiner Launcher for Linux
# PIL-only version - no external dependencies required

cd "$(dirname "$0")"

echo "Starting Image Pair Combiner GUI..."

python3 image_pair_combiner.py

echo "Image Pair Combiner closed."
echo "Press Enter to close this window..."
read
