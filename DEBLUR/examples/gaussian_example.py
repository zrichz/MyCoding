"""
Example script demonstrating Gaussian deblurring.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.image_utils import load_image, save_image, show_comparison
from src.deblur.gaussian_deblur import GaussianDeblur


def create_test_image():
    """Create a simple test image with text."""
    # Create a simple test image
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Add some text-like patterns
    image[50:60, 50:250] = [0, 0, 0]  # Horizontal line
    image[80:140, 100:110] = [0, 0, 0]  # Vertical line
    image[120:130, 150:200] = [0, 0, 0]  # Another horizontal line
    
    return image


def blur_image(image, kernel_size=15, sigma=None):
    """Add Gaussian blur to an image."""
    if sigma is None:
        sigma = kernel_size / 6.0
    
    from scipy import ndimage
    
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for i in range(3):
            blurred[:, :, i] = ndimage.gaussian_filter(image[:, :, i], sigma)
    else:
        blurred = ndimage.gaussian_filter(image, sigma)
    
    return blurred.astype(np.uint8)


def main():
    """Main example function."""
    print("Gaussian Deblurring Example")
    print("=" * 30)
    
    # Create or load test image
    if len(sys.argv) > 1:
        print(f"Loading image: {sys.argv[1]}")
        original = load_image(sys.argv[1])
    else:
        print("Creating test image...")
        original = create_test_image()
    
    # Add blur
    print("Adding Gaussian blur...")
    kernel_size = 15
    blurred = blur_image(original, kernel_size=kernel_size)
    
    # Save blurred image
    save_image(blurred, 'blurred_example.png')
    print("Saved blurred image as 'blurred_example.png'")
    
    # Initialize deblurrer
    deblurrer = GaussianDeblur()
    
    # Test different methods
    methods = ['richardson_lucy', 'wiener']
    
    for method in methods:
        print(f"\nTesting {method} deconvolution...")
        
        # Deblur image
        if method == 'richardson_lucy':
            result = deblurrer.deblur_image(blurred, kernel_size=kernel_size, 
                                          iterations=30, method=method, 
                                          auto_downsample=True, show_progress=True)
        else:
            result = deblurrer.deblur_image(blurred, kernel_size=kernel_size, 
                                          method=method, auto_downsample=True, 
                                          show_progress=True)
        
        # Save result
        output_name = f'deblurred_{method}.png'
        save_image(result, output_name)
        print(f"Saved result as '{output_name}'")
        
        # Show comparison
        show_comparison(blurred, result, 
                       titles=(f"Blurred (kernel size: {kernel_size})", 
                              f"Deblurred ({method})"))
    
    print("\nExample completed!")
    print("Check the saved images:")
    print("- blurred_example.png")
    for method in methods:
        print(f"- deblurred_{method}.png")


if __name__ == "__main__":
    main()
