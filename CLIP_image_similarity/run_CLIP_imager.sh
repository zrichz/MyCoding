#!/bin/bash
# Activate venv and run CLIP_imager.py with user-supplied image directory
cd "$(dirname "$0")"
source ../.venv/bin/activate
python CLIP_imager.py --input_dir "$1"
