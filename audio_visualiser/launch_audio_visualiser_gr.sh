#!/bin/bash
echo "Starting Audio Band Visualiser..."
source /home/rich/MyCoding/venvMyCoding/bin/activate
cd "$(dirname "$0")"
python audio_visualiser_gr.py
