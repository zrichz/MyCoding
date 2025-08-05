"""
Example script demonstrating motion deblurring.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.image_utils import load_image, save_image, show_comparison
from src.deblur.motion_deblur import MotionDeblur


def create_test_image():
    """Create a simple test image with geometric shapes."""
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Add some shapes
    # Rectangle
    image[50:100, 50:150] = [100, 100, 100]
    
    # Circle (approximation)
    center_y, center_x = 100, 200
    radius = 30
    y, x = np.ogrid[:200, :300]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = [150, 150, 150]
    
    # Diagonal line
    for i in range(50):
        if 0 <= 150 + i < 200 and 0 <= 180 + i < 300:
            image[150 + i, 180 + i] = [0, 0, 0]
    
    return image


def add_motion_blur(image, angle, length):
    """Add motion blur to an image."""
    from scipy import ndimage
    
    # Create motion kernel
    kernel_size = length * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    center = kernel_size // 2
    
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    for i in range(-length, length + 1):
        x = int(center + i * cos_angle)
        y = int(center + i * sin_angle)
        
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()
    
    # Apply motion blur
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for i in range(3):
            blurred[:, :, i] = ndimage.convolve(image[:, :, i].astype(np.float64), 
                                              kernel, mode='reflect')
    else:
        blurred = ndimage.convolve(image.astype(np.float64), kernel, mode='reflect')
    
    return np.clip(blurred, 0, 255).astype(np.uint8)


def main():
    """Main example function."""
    print("Motion Deblurring Example")
    print("=" * 30)
    
    # Create or load test image
    if len(sys.argv) > 1:
        print(f"Loading image: {sys.argv[1]}")
        original = load_image(sys.argv[1])
    else:
        print("Creating test image...")
        original = create_test_image()
    
    # Add motion blur
    angle = 30  # degrees
    length = 15  # pixels
    print(f"Adding motion blur (angle: {angle}°, length: {length}px)...")
    blurred = add_motion_blur(original, angle, length)
    
    # Save blurred image
    save_image(blurred, 'motion_blurred_example.png')
    print("Saved blurred image as 'motion_blurred_example.png'")
    
    # Initialize deblurrer
    deblurrer = MotionDeblur()
    
    # Test different methods
    methods = ['wiener', 'inverse', 'lucy_richardson']
    
    for method in methods:
        print(f"\nTesting {method} deconvolution...")
        
        # Deblur image
        if method == 'lucy_richardson':
            result = deblurrer.remove_motion_blur(blurred, angle=angle, length=length,
                                                method=method, iterations=30)
        else:
            result = deblurrer.remove_motion_blur(blurred, angle=angle, length=length,
                                                method=method)
        
        # Save result
        output_name = f'motion_deblurred_{method}.png'
        save_image(result, output_name)
        print(f"Saved result as '{output_name}'")
        
        # Show comparison
        show_comparison(blurred, result, 
                       titles=(f"Motion Blurred ({angle}°, {length}px)", 
                              f"Deblurred ({method})"))
    
    # Test parameter estimation
    print("\nTesting parameter estimation...")
    estimated_angle, estimated_length = deblurrer.estimate_motion_parameters(blurred)
    print(f"Estimated parameters: angle={estimated_angle:.1f}°, length={estimated_length}px")
    print(f"Actual parameters: angle={angle}°, length={length}px")
    
    print("\nExample completed!")
    print("Check the saved images:")
    print("- motion_blurred_example.png")
    for method in methods:
        print(f"- motion_deblurred_{method}.png")


if __name__ == "__main__":
    main()
