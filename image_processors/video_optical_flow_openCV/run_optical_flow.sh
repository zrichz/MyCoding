#!/bin/bash
echo "Starting Video Optical Flow Visualizer..."
cd "$(dirname "$0")"
source "../../venvmycoding313/bin/activate"
python optical_flow_visualizer.py
