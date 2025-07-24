#!/usr/bin/env python3
"""
Test script for gradient-based stacking with smoothing
"""

import cv2
import numpy as np
from focus_stacking_algorithms import FocusStackingAlgorithms

def create_test_images():
    """Create simple test images with different focus regions."""
    # Create base image
    base = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Image 1: Sharp left, blurry right
    img1 = base.copy()
    img1[50:150, 50:100] = [255, 0, 0]  # Red rectangle (sharp)
    img1[50:150, 120:170] = [0, 255, 0]  # Green rectangle (will be blurred)
    # Blur the right side
    img1[:, 120:] = cv2.GaussianBlur(img1[:, 120:], (15, 15), 5)
    
    # Image 2: Blurry left, sharp right  
    img2 = base.copy()
    img2[50:150, 50:100] = [255, 0, 0]  # Red rectangle (will be blurred)
    img2[50:150, 120:170] = [0, 255, 0]  # Green rectangle (sharp)
    # Blur the left side
    img2[:, :100] = cv2.GaussianBlur(img2[:, :100], (15, 15), 5)
    
    return [img1, img2]

def test_gradient_stacking():
    """Test gradient stacking with and without smoothing."""
    print("Creating test images...")
    images = create_test_images()
    
    print("Testing gradient stacking without smoothing...")
    result_sharp = FocusStackingAlgorithms.gradient_based_stack(
        images, smooth_radius=0, blend_sigma=1.0)
    
    print("Testing gradient stacking with smoothing...")
    result_smooth = FocusStackingAlgorithms.gradient_based_stack(
        images, smooth_radius=3, blend_sigma=1.0)
    
    # Save results
    cv2.imwrite('test_gradient_sharp.png', result_sharp)
    cv2.imwrite('test_gradient_smooth.png', result_smooth)
    cv2.imwrite('test_img1.png', images[0])
    cv2.imwrite('test_img2.png', images[1])
    
    print("Results saved:")
    print("- test_img1.png (left sharp, right blurry)")
    print("- test_img2.png (left blurry, right sharp)")  
    print("- test_gradient_sharp.png (hard selection)")
    print("- test_gradient_smooth.png (smooth blending)")
    
    return result_sharp, result_smooth

if __name__ == "__main__":
    test_gradient_stacking()
