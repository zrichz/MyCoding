# TI Changer Multiple - Enhanced Version

## Overview
This script allows you to load and manipulate PyTorch textual inversion (.pt) files, with various processing options including a new individual vector extraction feature.

## New Feature: Individual Vector Extraction (Option 6)

### What it does:
- Loads a multi-vector textual inversion file
- Shows the number of vectors present in the file
- Extracts each vector as a separate, individual .pt file
- Names each file with a numbered suffix for easy identification

### Example Usage:
If you have a TI file called `my_style.pt` containing 8 vectors, selecting Option 6 will create:
```
my_style_vector_01.pt
my_style_vector_02.pt
my_style_vector_03.pt
my_style_vector_04.pt
my_style_vector_05.pt
my_style_vector_06.pt
my_style_vector_07.pt
my_style_vector_08.pt
```

### How to use:
1. Run the script using one of these methods:
   - **Windows**: Double-click `run_TI_CHANGER.bat`
   - **Linux/Ubuntu**: Run `./run_TI_CHANGER.sh` or `bash run_TI_CHANGER.sh`
   - **Cross-platform**: Run `python run_TI_CHANGER.py` (works on Windows, Linux, macOS)
   - **Direct**: Use the full Python path with your virtual environment

2. When prompted for options, enter `6`

3. The script will show you:
   - Number of vectors in the source file
   - List of files that will be created
   - Confirmation prompt

4. Type `y` to proceed with extraction

5. Each vector will be saved as a fully functional, individual TI file

### Benefits:
- **Flexibility**: Use individual vectors separately in your AI applications
- **Organization**: Better file management for complex multi-vector embeddings
- **Compatibility**: Each extracted file is a complete, standalone TI file
- **Safety**: Original file remains unchanged
- **Clarity**: Clear naming convention with numbered suffixes

## All Available Options:
1. **Smoothing**: Apply smoothing filter to all vectors
2. **Mean**: Create a single averaged vector from all vectors  
3. **Decimation**: Zero out nth elements (placeholder implementation)
4. **Division**: Divide all vectors by a scalar value
5. **Rolling**: Shift vector elements by specified amount
6. **⭐ NEW: Individual Extraction**: Extract each vector to separate files

## Requirements:
- Python 3.10+
- PyTorch
- Matplotlib
- NumPy
- Virtual environment at: `C:\MyPythonCoding\MyCoding\image_processors\.venv`

## Test File:
A test file `TEST_4vectors.pt` has been created with 4 sample vectors. You can use this to test the extraction feature by changing the filename in the script to `'TEST_4vectors.pt'`.
