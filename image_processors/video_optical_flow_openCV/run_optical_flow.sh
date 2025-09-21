#!/bin/bash
echo "Starting Video Optical Flow Visualizer..."
cd "$(dirname "$0")"
source "../.venv/bin/activate"
python optical_flow_visualizer.py
