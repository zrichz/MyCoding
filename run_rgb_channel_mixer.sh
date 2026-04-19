#!/bin/bash

# RGB Channel Mixer Launcher

cd "$(dirname "$0")"

# Check if venv exists, create if not
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found."
    read -p "Create virtual environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    else
        echo "Cannot proceed without virtual environment. Exiting."
        exit 1
    fi
fi

# Activate venv
source .venv/bin/activate

# Install dependencies if needed
if ! python -c "import gradio" 2>/dev/null; then
    echo "Required dependencies not found."
    read -p "Install dependencies (gradio, numpy)? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing dependencies..."
        pip install gradio numpy
    else
        echo "Cannot run without dependencies. Exiting."
        exit 1
    fi
fi

# Run the app
echo "Launching RGB Channel Mixer..."
python rgb_channel_mixer_gradio.py
