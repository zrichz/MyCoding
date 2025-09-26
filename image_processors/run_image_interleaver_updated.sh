#!/bin/bash
# Image Interleaver Launcher for Linux
# This script will try the PIL-only version first (no dependencies), 
# then fall back to the numpy version if needed

cd "$(dirname "$0")"

echo "Starting Image Interleaver GUI..."

# Try PIL-only version first
if python3 image_interleaver_pil_only.py; then
    echo "Image Interleaver completed successfully"
else
    echo "PIL-only version failed, trying numpy version..."
    python3 image_interleaver.py
fi

echo "Press Enter to close this window..."
read
