#!/bin/bash

# Change to the script directory
cd "$(dirname "$0")"

# Activate virtual environment
source /home/rich/MyCoding/venvmycoding313/bin/activate

# Refresh hash table for python command
hash -r

# Run the TI Changer SDXL Gradio script
python TI_Changer_SDXL_gradio.py
