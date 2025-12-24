#!/bin/bash

# Music Visualizer Launcher
# Uses the existing .venv in MyCoding directory

echo "🎵 Music Visualizer Launcher"
echo "=============================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to the virtual environment
VENV_PATH="$SCRIPT_DIR/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "Please ensure the .venv directory exists in the MyCoding folder."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check and install required packages
echo "📦 Checking required packages..."

# Function to check if package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Function to install package
install_package() {
    echo "   Installing $1..."
    pip install $1
}

# Required packages
packages=(
    "librosa"
    "matplotlib" 
    "opencv-python"
    "numpy"
    "scipy"
    "soundfile"
    "ffmpeg-python"
)

missing_packages=()

for package in "${packages[@]}"; do
    # Handle package name differences
    import_name=$package
    if [ "$package" = "opencv-python" ]; then
        import_name="cv2"
    elif [ "$package" = "ffmpeg-python" ]; then
        import_name="ffmpeg"
    fi
    
    if ! check_package $import_name; then
        missing_packages+=($package)
    else
        echo "   ✅ $package"
    fi
done

# Install missing packages
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo ""
    echo "📥 Installing missing packages: ${missing_packages[*]}"
    pip install "${missing_packages[@]}"
    echo ""
fi

# Verify librosa specifically (it's the most important one)
if check_package librosa; then
    echo "✅ All packages ready!"
else
    echo "❌ Error: librosa installation failed. Trying alternative installation..."
    pip install librosa --no-cache-dir
fi

echo ""
echo "🚀 Starting Music Visualizer..."
python "$SCRIPT_DIR/music_visualizer.py"

echo "Done."