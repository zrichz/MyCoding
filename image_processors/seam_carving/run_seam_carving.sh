#!/bin/bash

# Seam Carving Width Adjuster - Easy Launch Script
# This shell script activates the virtual environment and runs the GUI application
# Supports both width reduction (50-99%) and expansion (100-150%)

echo "Starting Seam Carving Width Adjuster..."
echo

# Check if virtual environment exists
if [ ! -f "../.venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please make sure the .venv folder exists in the parent directory."
    echo "You may need to create it first by running:"
    echo "  python -m venv ../.venv"
    echo "  source ../.venv/bin/activate"
    echo "  pip install -r seam_carving_requirements.txt"
    echo
    read -p "Press Enter to continue..."
    exit 1
fi

# Check if the main script exists
if [ ! -f "seam_carving_width_reducer.py" ]; then
    echo "ERROR: seam_carving_width_reducer.py not found!"
    echo "Please make sure you're running this script from the correct directory."
    echo
    read -p "Press Enter to continue..."
    exit 1
fi

# Activate virtual environment and run the application
echo "Activating virtual environment..."
source ../.venv/bin/activate

echo "Running Seam Carving Width Adjuster GUI..."
echo
python seam_carving_width_reducer.py

# Check if there was an error
if [ $? -ne 0 ]; then
    echo
    echo "An error occurred while running the application."
    read -p "Press Enter to continue..."
fi

echo
echo "Application closed."
