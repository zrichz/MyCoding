#!/bin/bash

echo "Starting Neural Cellular Automata..."
echo

# Check if we're in the correct directory
if [ ! -f "NCA_baseline.py" ]; then
    echo "Error: NCA_baseline.py not found in current directory"
    echo "Please run this script from the Neural_Cellular_Automata folder"
    read -p "Press Enter to exit..."
    exit 1
fi

# Navigate to parent directory to find venv
cd ..

# Check for various virtual environment names
if [ -f "venv_mycoding/bin/activate" ]; then
    echo "Activating virtual environment: venv_mycoding"
    source venv_mycoding/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment: venv"
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment: .venv"
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found"
    echo "Continuing with system Python..."
fi

echo
echo "Running Neural Cellular Automata GUI..."
echo "Press Ctrl+C to stop the application"
echo

# Run the script from the Neural_Cellular_Automata directory
python Neural_Cellular_Automata/NCA_baseline.py

echo
echo "Neural Cellular Automata has closed."
read -p "Press Enter to exit..."
