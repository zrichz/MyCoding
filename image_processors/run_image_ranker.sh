#!/bin/bash
# Image Ranker Launcher for Unix/Linux/macOS
# This script launches the image ranker with optimized 2560x1440 layout

echo ""
echo "================================================"
echo "    Image Ranker - Tournament Style Ranking"
echo "================================================"
echo ""
echo "Features:"
echo "- Optimized for 2560x1440 resolution"
echo "- Side-by-side image comparison"
echo "- Tournament-style pairwise ranking"
echo "- Automatic file renaming with rank prefixes"
echo "- No popup interruptions"
echo ""

# Change to the directory containing this script
cd "$(dirname "$0")"

# Check if Python virtual environment exists
if [ -f "../venv_mycoding/bin/python" ]; then
    echo "Using virtual environment..."
    ../venv_mycoding/bin/python image_ranker.py
elif [ -f "../.venv/bin/python" ]; then
    echo "Using .venv virtual environment..."
    ../.venv/bin/python image_ranker.py
elif command -v python3 &> /dev/null; then
    echo "Using system Python3..."
    python3 image_ranker.py
elif command -v python &> /dev/null; then
    echo "Using system Python..."
    python image_ranker.py
else
    echo "Error: Python not found"
    echo "Please install Python or activate your virtual environment"
    exit 1
fi

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to launch Image Ranker"
    echo "Please check that Python and required packages are installed"
    echo ""
    read -p "Press Enter to continue..."
fi