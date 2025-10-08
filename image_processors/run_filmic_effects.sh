#!/bin/bash

# Filmic Effects Processor Launcher for Ubuntu/Linux
# Based on the Windows batch file structure

set -e  # Exit on any error

echo "=============================================="
echo "🎬 Filmic Effects Processor"
echo "=============================================="
echo "Starting Filmic Effects Processor..."
echo

# Navigate to the image_processors directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $(pwd)"

# Check if virtual environment exists in current directory
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ ERROR: Virtual environment not found at .venv/bin/"
    echo "Please ensure the .venv is properly set up in the image_processors directory."
    echo
    echo "To create the virtual environment, run:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if script exists
if [ ! -f "filmic_effects_processor.py" ]; then
    echo "❌ ERROR: filmic_effects_processor.py not found in image_processors directory"
    echo "Current directory: $(pwd)"
    echo "Available Python files:"
    ls -la *.py 2>/dev/null || echo "  No Python files found"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "
import sys
missing = []
try:
    import numpy
    print('✅ numpy: OK')
except ImportError:
    missing.append('numpy')
    print('❌ numpy: MISSING')

try:
    import PIL
    print('✅ pillow: OK')  
except ImportError:
    missing.append('pillow')
    print('❌ pillow: MISSING')

try:
    import cv2
    print('✅ opencv-python: OK')
except ImportError:
    missing.append('opencv-python')
    print('❌ opencv-python: MISSING')

try:
    import matplotlib
    print('✅ matplotlib: OK')
except ImportError:
    missing.append('matplotlib')
    print('❌ matplotlib: MISSING')

if missing:
    print(f'\\n❌ Missing packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('\\n✅ All dependencies satisfied!')
"

if [ $? -ne 0 ]; then
    echo
    echo "Installing missing dependencies..."
    pip install numpy pillow opencv-python matplotlib
    echo
fi

# Run the filmic effects processor
echo
echo "🚀 Running Filmic Effects Processor..."
echo "=============================================="
python filmic_effects_processor.py

# Capture exit code
EXIT_CODE=$?

echo
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Filmic Effects Processor completed successfully!"
else
    echo "❌ Filmic Effects Processor exited with error code: $EXIT_CODE"
fi

# Deactivate virtual environment
echo "🔧 Deactivating virtual environment..."
deactivate

echo
echo "Press Enter to exit..."
read
