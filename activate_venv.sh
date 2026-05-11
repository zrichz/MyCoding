#!/bin/bash
# Activation script for venvmycoding313
# Usage: source activate_venv.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venvmycoding313/bin/activate"

# Clear Python path hash
hash -r

echo "Virtual environment venvmycoding313 activated"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
