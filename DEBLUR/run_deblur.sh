#!/bin/bash
# Simple launcher script for the DEBLUR application

# Set the working directory to the script location
cd "$(dirname "$0")"

# Set PYTHONPATH and run the application
export PYTHONPATH="/home/rich/MyCoding/DEBLUR"
/home/rich/MyCoding/DEBLUR/.venv/bin/python main.py "$@"
