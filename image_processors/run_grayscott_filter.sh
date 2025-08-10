#!/bin/bash
echo "Starting Gray-Scott Filter..."
cd "$(dirname "$0")/GrayScott_filter/src"
source "../.venv/bin/activate"
python GrayScott_filter.py
