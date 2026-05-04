#!/bin/bash

# Image Hash Encoder/Decoder - Launcher Script
# Activates Python 3.13 virtual environment and runs the Gradio application

# Activate virtual environment
source /home/rich/MyCoding/venvmycoding313/bin/activate

# Refresh hash table for python command
hash -r

# Navigate to the image_hash_project directory
cd "$(dirname "$0")"

# Run the Gradio application
python image_hash_gradio.py
