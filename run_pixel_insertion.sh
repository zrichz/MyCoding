#!/bin/bash

# Activate the Python 3.12 virtual environment
source /home/rich/MyVenvs/myVenv312/bin/activate

# Clear the hash table to ensure correct Python version
hash -r

# Run the pixel insertion Gradio app
python pixel_insertion_gradio.py
