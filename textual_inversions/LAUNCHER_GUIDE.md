# TI Changer Multiple - Launcher Guide

## Available Launchers

### 🪟 Windows
- **`run_TI_CHANGER.bat`** - Windows batch file
  - Double-click to run
  - Uses the configured virtual environment path
  - Windows Command Prompt compatible

### 🐧 Linux/Ubuntu
- **`run_TI_CHANGER.sh`** - Bash script for Linux systems
  - Make executable: `chmod +x run_TI_CHANGER.sh`
  - Run with: `./run_TI_CHANGER.sh` or `bash run_TI_CHANGER.sh`
  - Automatically detects Python installation
  - Handles virtual environments
  - Installs missing packages if needed

### 🌐 Cross-Platform (Recommended)
- **`run_TI_CHANGER.py`** - Python launcher script
  - Works on Windows, Linux, macOS
  - Run with: `python run_TI_CHANGER.py` or `python3 run_TI_CHANGER.py`
  - Automatically detects system and Python installation
  - Checks and installs dependencies
  - Provides detailed system information

## Quick Start

### On Ubuntu/Linux:
```bash
# Make the script executable
chmod +x run_TI_CHANGER.sh

# Run the script
./run_TI_CHANGER.sh
```

### Cross-Platform:
```bash
# Works on any system with Python
python run_TI_CHANGER.py
```

### Manual Python Execution:
```bash
# If you have a specific Python environment
python3 TI_CHANGER_MULTIPLE_2024_10_22.py
```

## Dependencies for Linux

The launchers will automatically try to install these if missing:
- `torch` (PyTorch)
- `matplotlib`
- `numpy`
- `tkinter` (usually included with Python)

### Manual Installation on Ubuntu:
```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-tkinter

# Install Python packages
pip3 install torch matplotlib numpy
```

## Virtual Environment Setup (Optional)

For Linux systems, you can create a local virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install packages
pip install torch matplotlib numpy

# Run the script
python TI_CHANGER_MULTIPLE_2024_10_22.py
```

## Troubleshooting

### Linux Issues:
- **Permission denied**: Run `chmod +x run_TI_CHANGER.sh`
- **tkinter not found**: Install with `sudo apt install python3-tkinter`
- **Python not found**: Install with `sudo apt install python3`

### General Issues:
- Use the cross-platform launcher (`run_TI_CHANGER.py`) for best compatibility
- Check that all required packages are installed
- Ensure you're in the correct directory with the script files
