# Gray-Scott Filter App

This project is a Gray-Scott diffusion filter application built using Python's Tkinter library. It allows users to load an image, apply a specified number of iterations of Gray-Scott diffusion processing, and save the processed image.

## Features

- Load an image from your file system.
- Specify the number of iterations for Gray-Scott diffusion processing.
- Preview the processed image in the GUI.
- Save the processed image to your file system.
- Automatic image resizing for optimal processing (max 1024x1024 pixels).

## Requirements

To run this application, you need to have Python installed along with the following packages:

- Pillow
- tkinter (usually included with Python)

You can install the required packages using pip. Make sure to include the following in your `requirements.txt`:

```
Pillow
```

## Installation

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

### Windows
1. Run the batch file:
   ```
   run_grayscott_filter.bat
   ```

### Linux/Mac
1. Make the script executable:
   ```
   chmod +x run_grayscott_filter.sh
   ```
2. Run the shell script:
   ```
   ./run_grayscott_filter.sh
   ```

### Manual Launch
1. Run the application directly:
   ```
   python src/GrayScott_filter.py
   ```

2. Use the GUI to load an image.
3. Specify the number of iterations for Gray-Scott diffusion processing.
4. Click the "Process Image" button to apply the effects.
5. Preview the processed image in the application.
6. Click the "Save Image" button to save the processed image.

## Processing Details

The Gray-Scott filter applies iterative sharpening and blurring operations:
- Converts images to grayscale for optimal processing
- Applies dual sharpening passes per iteration
- Follows with Gaussian blur for diffusion effect
- Results in unique texture and pattern generation

## Supported Formats

- Input: Most common image formats (JPEG, PNG, BMP, TIFF, etc.)
- Output: PNG (default), JPEG, and other PIL-supported formats
