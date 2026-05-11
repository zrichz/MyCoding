#!/bin/bash
# Activate virtual environment
source /home/rich/MyCoding/.venv/bin/activate

# Clear Python hash table to ensure correct version
hash -r

# Run the style transfer Gradio app
python /home/rich/MyCoding/image_processors/image_style_transfer_gradio.py
