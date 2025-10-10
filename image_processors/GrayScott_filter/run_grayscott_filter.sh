#!/bin/bash
echo "Starting Gray-Scott Filter..."
cd "$(dirname "$0")"
source "../.venv/bin/activate"
python GrayScott_filter.py
