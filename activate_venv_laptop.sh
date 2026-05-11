#!/bin/bash
# Activation script for .venv laptop May 2026

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/.venv/bin/activate"

# Clear Python path hash
hash -r

echo "Virtual environment .venv activated"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
