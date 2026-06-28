#!/bin/bash
# Activation script for venvMyCoding
# Usage: source activate_venv.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venvMyCoding/bin/activate"

# Clear Python path hash
hash -r

echo "Virtual environment venvMyCoding activated"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
