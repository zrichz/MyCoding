<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# DEBLUR Project Instructions

This is a Python image deblurring project focused on implementing various deconvolution and filtering techniques to remove different types of blur from images.

## Key Technologies and Libraries
- OpenCV for image processing operations
- NumPy and SciPy for numerical computations
- scikit-image for advanced image processing algorithms
- Pillow for image I/O operations
- Matplotlib for visualization and results display

## Project Focus Areas
- Gaussian blur removal using Richardson-Lucy deconvolution
- Motion blur correction with Wiener filtering
- Edge-preserving deblurring techniques
- Point Spread Function (PSF) estimation
- Noise-robust deblurring algorithms

## Code Style Guidelines
- Use clear, descriptive function names that indicate the deblurring method
- Include comprehensive docstrings with parameter descriptions and usage examples
- Implement proper error handling for invalid image inputs
- Add performance timing for algorithm comparison
- Use type hints for better code documentation

## Implementation Preferences
- Prefer vectorized NumPy operations over loops when possible
- Include both batch processing and single image processing functions
- Implement visualization functions to show before/after comparisons
- Add parameter validation and sensible defaults
- Create modular code that allows mixing different deblurring techniques
