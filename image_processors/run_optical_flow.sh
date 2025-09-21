#!/bin/bash
echo "Starting Video Optical Flow Visualizer..."
cd "$(dirname "$0")/video_optical_flow_openCV"
source "../.venv/bin/activate"
python optical_flow_visualizer.py
