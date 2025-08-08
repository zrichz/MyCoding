# Seam Carving Width Reducer

A Python GUI application that reduces image width using the seam carving algorithm. The algorithm is applied specifically to the first 25% and last 25% of the image width, preserving the middle content.

## Features

- **GUI Interface**: Easy-to-use tkinter-based interface
- **Selective Seam Carving**: Applies seam carving only to the outer 50% of the image (first and last 25%)
- **Configurable Reduction**: User can specify reduction percentage (50-99%)
- **Auto-save**: Processed images are automatically saved with "_reduced_width" suffix
- **Progress Tracking**: Real-time progress bar and status updates

## Requirements

- Python 3.7+
- Pillow (PIL)
- OpenCV
- NumPy
- tkinter (usually comes with Python)

## Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv seam_carving_env
   source seam_carving_env/bin/activate  # On Linux/Mac
   # or
   seam_carving_env\Scripts\activate     # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r seam_carving_requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python seam_carving_width_reducer.py
   ```

2. **Select Image**: Click "Browse" to select an image file (JPG, PNG, BMP, TIFF supported)

3. **Set Reduction Percentage**: Enter a value between 50-99% for width reduction

4. **Process**: Click "Process Image" to apply seam carving

5. **Result**: The processed image will be displayed and automatically saved in the same directory as the original with "_reduced_width" suffix

## How Seam Carving Works

Seam carving is a content-aware image resizing technique that:

1. **Energy Calculation**: Uses gradient magnitude to identify important image features
2. **Seam Detection**: Finds minimum-energy vertical paths through the image
3. **Seam Removal**: Removes these low-energy seams to reduce width

This implementation applies the algorithm selectively to:
- **First 25%** of image width (left side)
- **Last 25%** of image width (right side)
- **Preserves** the middle 50% unchanged

## Example

If you have an image that's 1000 pixels wide and choose 70% reduction:
- Target width: 700 pixels (300 pixels removed)
- First 250 pixels: ~150 seams removed
- Middle 500 pixels: unchanged
- Last 250 pixels: ~150 seams removed

## Notes

- Processing time depends on image size and number of seams to remove
- Best results with images that have less important content on the sides
- The algorithm preserves important vertical structures in the middle of the image
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
