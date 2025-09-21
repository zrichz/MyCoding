#!/bin/bash
# Script to run 32x32 RGB Ring Autoencoder with the appropriate virtual environment

echo "=== 32x32 RGB Ring Autoencoder ==="
echo "Choose an option:"
echo "1) Train new model (train_32x32.py)"
echo "2) Test existing model (test_32x32.py)"
echo ""
read -p "Enter your choice (1-2): " choice

echo "Activating virtual environment..."
source /home/rich/MyCoding/textual_inversions/.venv/bin/activate

cd /home/rich/MyCoding/CNN_draw_circles_project

case $choice in
    1)
        echo "Training new 32x32 model..."
        python3 train_32x32.py
        ;;
    2)
        echo "Testing existing 32x32 model..."
        python3 test_32x32.py
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "Script completed!"
