#!/usr/bin/env python3
"""
Demonstrate the gradient smoothing improvement for noise reduction
"""

import cv2
import numpy as np
from focus_stacking_algorithms import FocusStackingAlgorithms

def create_noisy_test_images():
    """Create test images that will show gradient noise without smoothing."""
    # Create more complex test pattern
    base = np.ones((300, 300, 3), dtype=np.uint8) * 50  # Dark background
    
    # Add noise to make gradient selection more challenging
    noise = np.random.normal(0, 10, base.shape).astype(np.int16)
    
    # Image 1: Focus in center-left with fine details
    img1 = base.copy().astype(np.int16) + noise
    # Fine detail pattern in focus region
    for i in range(100, 200, 10):
        for j in range(80, 150, 10):
            img1[i:i+5, j:j+5] = [200, 100, 50]
    
    # Blur the right side
    img1_clean = np.clip(img1, 0, 255).astype(np.uint8)
    img1_clean[:, 180:] = cv2.GaussianBlur(img1_clean[:, 180:], (21, 21), 8)
    
    # Image 2: Focus in center-right with fine details  
    img2 = base.copy().astype(np.int16) + noise
    # Fine detail pattern in focus region
    for i in range(100, 200, 10):
        for j in range(180, 250, 10):
            img2[i:i+5, j:j+5] = [50, 200, 100]
    
    # Blur the left side
    img2_clean = np.clip(img2, 0, 255).astype(np.uint8)
    img2_clean[:, :150] = cv2.GaussianBlur(img2_clean[:, :150], (21, 21), 8)
    
    return [img1_clean, img2_clean]

def demonstrate_smoothing():
    """Demonstrate the noise reduction with gradient smoothing."""
    print("Creating noisy test images...")
    images = create_noisy_test_images()
    
    print("\n=== Testing Gradient Stacking ===")
    
    # Test without smoothing (old method - noisy)
    print("1. Without smoothing (noisy result)...")
    result_noisy = FocusStackingAlgorithms.gradient_based_stack(
        images, smooth_radius=0, blend_sigma=0.0)
    
    # Test with light smoothing
    print("2. With light smoothing (radius=2)...")
    result_light = FocusStackingAlgorithms.gradient_based_stack(
        images, smooth_radius=2, blend_sigma=1.0)
    
    # Test with medium smoothing  
    print("3. With medium smoothing (radius=4)...")
    result_medium = FocusStackingAlgorithms.gradient_based_stack(
        images, smooth_radius=4, blend_sigma=1.5)
    
    # Save all results for comparison
    cv2.imwrite('demo_input1.png', images[0])
    cv2.imwrite('demo_input2.png', images[1])
    cv2.imwrite('demo_gradient_noisy.png', result_noisy)
    cv2.imwrite('demo_gradient_light.png', result_light)
    cv2.imwrite('demo_gradient_medium.png', result_medium)
    
    print("\n=== Results Saved ===")
    print("- demo_input1.png (focus left)")
    print("- demo_input2.png (focus right)")
    print("- demo_gradient_noisy.png (old method - noisy)")
    print("- demo_gradient_light.png (light smoothing)")
    print("- demo_gradient_medium.png (medium smoothing)")
    
    print("\n=== Analysis ===")
    # Calculate some basic quality metrics
    def calculate_variance(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return np.var(gray)
    
    var_noisy = calculate_variance(result_noisy)
    var_light = calculate_variance(result_light)
    var_medium = calculate_variance(result_medium)
    
    print(f"Variance (noise indicator):")
    print(f"  Noisy result:  {var_noisy:.1f}")
    print(f"  Light smooth:  {var_light:.1f} ({var_light/var_noisy:.2f}x)")
    print(f"  Medium smooth: {var_medium:.1f} ({var_medium/var_noisy:.2f}x)")
    
    print(f"\nRecommendation: Use smooth_radius=3-5 for best noise reduction")
    print(f"while preserving detail.")

if __name__ == "__main__":
    demonstrate_smoothing()
