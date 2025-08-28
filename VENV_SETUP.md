# MyCoding Virtual Environment Setup

## Overview
This directory now has a dedicated virtual environment for all Python scripts.

## Virtual Environment Location
- **Path:** `C:\MyPythonCoding\MyCoding\venv_mycoding`
- **Python Version:** 3.10.6
- **Activation Script:** `activate_venv.bat`

## Quick Start

### Activate the Virtual Environment
```bash
# Method 1: Use the activation script
activate_venv.bat

# Method 2: Manual activation
venv_mycoding\Scripts\activate
```

### Install Additional Packages
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Install individual packages
pip install package_name
```

### Deactivate
```bash
deactivate
```

## Updated Scripts

The following scripts have been updated to use the new virtual environment:

1. **Textual Inversions:** `textual_inversions\run_TI_CHANGER.bat`
2. **Focus Stacker:** `.vscode\tasks.json` (Focus Stacker GUI task)

## Python Executable Path
When creating new batch files or scripts, use:
```
C:\MyPythonCoding\MyCoding\venv_mycoding\Scripts\python.exe
```

## Benefits
- ✅ Isolated environment for all MyCoding scripts
- ✅ Consistent Python version across all projects
- ✅ Easy package management
- ✅ No conflicts with other Python installations
- ✅ All scripts now use the same, working environment
