#!/bin/bash
# RGB Channel Mixer Launcher

cd "$(dirname "$0")"

# Activate workspace virtual environment
source /home/rich/MyCoding/venvmycoding313/bin/activate
hash -r

# Run the app
echo "Launching RGB Channel Mixer..."
python rgb_channel_mixer_gradio.py
