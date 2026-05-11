#!/bin/bash
# Pixel Shifter Launcher

cd "$(dirname "$0")"

# Activate workspace virtual environment
source /home/rich/MyCoding/venvmycoding313/bin/activate
hash -r

# Run the app
echo "Launching Pixel Shifter..."
python pixel_shifter_gradio.py
