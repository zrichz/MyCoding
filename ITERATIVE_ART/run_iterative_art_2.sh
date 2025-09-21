#!/bin/bash
# Activate the image_processors .venv and run iterative_art_2.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYCODING_DIR="$(dirname "$SCRIPT_DIR")"
source "$MYCODING_DIR/image_processors/.venv/bin/activate"
python "$SCRIPT_DIR/iterative_art_2.py"
