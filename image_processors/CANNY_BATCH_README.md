# Canny Batch Processor

A Python GUI application that applies Canny edge detection to all images in a selected directory.

## Features

- **Directory Selection**: Choose any directory containing images
- **Batch Processing**: Processes all supported image formats automatically  
- **Customizable Parameters**: Adjust Canny edge detection settings:
  - Low Threshold (1-255, default: 50)
  - High Threshold (1-255, default: 150)
  - Aperture Size (3, 5, or 7, default: 3)
- **Progress Tracking**: Real-time progress bar and status updates
- **Background Processing**: Non-blocking UI with stop functionality
- **Automatic Output**: Creates 'canny' subdirectory with results

## Supported Formats

Input: JPG, JPEG, PNG, BMP, TIFF, GIF, WEBP
Output: PNG files with "_canny" suffix

## Requirements

- Python 3.6+
- OpenCV (`opencv-python`)
- NumPy
- Pillow (PIL)
- Tkinter (usually included with Python)

## Installation

The launcher scripts will automatically attempt to install required packages:

### Linux/Mac:
```bash
chmod +x run_canny_batch_processor.sh
./run_canny_batch_processor.sh
```

### Windows:
```cmd
run_canny_batch_processor.bat
```

Or manually install dependencies:
```bash
pip install opencv-python numpy pillow
python canny_batch_processor.py
```

## Usage

1. **Launch** the application using the launcher script
2. **Browse** to select a directory containing images
3. **Adjust** Canny parameters if needed (optional)
4. **Click "Process Images"** to start batch processing
5. **Results** are saved in `[input_directory]/canny/`

## Output

- Each image gets processed with Canny edge detection
- Output files are named: `[original_name]_canny.png`
- Black background with white edges
- Grayscale PNG format for optimal edge visibility

## Canny Parameters

- **Low Threshold**: Lower values detect more edges (more sensitive)
- **High Threshold**: Should be 2-3x the low threshold
- **Aperture Size**: Sobel kernel size (3, 5, or 7) - larger = smoother edges

## Example

Input: `photo.jpg`
Output: `canny/photo_canny.png`

The Canny edge detector finds edges by looking for rapid changes in brightness, producing clean line drawings that highlight the structure and contours in your images.
