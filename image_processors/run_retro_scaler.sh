#!/bin/bash

echo "Starting Retro Scaler - Intelligent Montage Creator..."
echo

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Python is not installed or not in PATH."
        echo "Please install Python 3.7+ and try again."
        read -p "Press Enter to continue..."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if virtual environment exists
if [ -f "../../.venv/bin/python" ]; then
    echo "Using virtual environment..."
    "../../.venv/bin/python" retro_scaler_720x1600.py
elif [ -f "../../venv/bin/python" ]; then
    echo "Using virtual environment (venv)..."
    "../../venv/bin/python" retro_scaler_720x1600.py
else
    echo "Using system Python..."
    $PYTHON_CMD retro_scaler_720x1600.py
fi

if [ $? -ne 0 ]; then
    echo
    echo "Script encountered an error."
    read -p "Press Enter to continue..."
fi