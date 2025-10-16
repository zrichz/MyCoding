#!/bin/bash

echo "Activating MyCoding virtual environment..."
echo "Location: $(pwd)/.venv"
echo

# Change to the MyCoding directory (if not already there)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at .venv"
    echo "Please make sure the virtual environment is properly set up."
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

echo
echo "Virtual environment activated!"
echo "Python executable: $(pwd)/.venv/bin/python"
echo "Current directory: $(pwd)"
echo
echo "You can now run any Python scripts in this environment."
echo "To deactivate, simply type: deactivate"
echo

# Keep the shell session active
exec bash --rcfile <(echo "PS1='(MyCoding-venv) \u@\h:\w\$ '")