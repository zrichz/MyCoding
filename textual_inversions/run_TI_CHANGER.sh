#!/bin/bash

# Linux/Ubuntu launcher for TI Changer Multiple
# This script runs the TI Changer with the appropriate Python environment

echo "Running TI Changer Multiple..."
echo "Operating System: $(uname -s)"
echo ""

# Check if we're on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    
    # Try to find Python 3 installation
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "Using: $(which python3)"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo "Using: $(which python)"
    else
        echo "Error: Python not found. Please install Python 3."
        exit 1
    fi
    
    # Check for virtual environment
    # First check if there's a local venv
    if [ -d ".venv" ]; then
        echo "Found local virtual environment"
        source .venv/bin/activate
        PYTHON_CMD="python"
    elif [ -d "../image_processors/.venv" ]; then
        echo "Found image_processors virtual environment"
        source ../image_processors/.venv/bin/activate
        PYTHON_CMD="python"
    else
        echo "No virtual environment found, using system Python"
        # Install required packages if not available
        echo "Checking for required packages..."
        $PYTHON_CMD -c "import torch, matplotlib, numpy, tkinter" 2>/dev/null || {
            echo "Missing required packages. Installing..."
            $PYTHON_CMD -m pip install torch matplotlib numpy --user
        }
        
        # Check for scikit-learn for new clustering and PCA features
        $PYTHON_CMD -c "import sklearn" 2>/dev/null || {
            echo "Installing scikit-learn for clustering and PCA features..."
            $PYTHON_CMD -m pip install scikit-learn --user
        }
    fi
    
else
    echo "Error: This script is designed for Linux systems"
    echo "On Windows, use: run_TI_CHANGER.bat"
    exit 1
fi

echo ""
echo "Starting TI Changer Multiple..."
echo "----------------------------------------"

# Run the Python script
$PYTHON_CMD "TI_CHANGER_MULTIPLE_2024_10_22.py"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "TI Changer completed successfully!"
else
    echo ""
    echo "TI Changer encountered an error."
fi

echo ""
echo "Press Enter to continue..."
read
