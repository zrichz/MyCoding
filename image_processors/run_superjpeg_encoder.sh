#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

echo "Starting SuperJPEG Encoder/Decoder..."
echo

# Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found at .venv/bin/activate"
    echo "Please ensure the virtual environment is set up correctly."
    exit 1
fi

# Run the SuperJPEG encoder
python superjpeg_encoder.py

# Deactivate virtual environment
deactivate

echo
echo "SuperJPEG Encoder/Decoder closed."
