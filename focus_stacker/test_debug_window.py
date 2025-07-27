#!/usr/bin/env python3
"""
Test the gradient debug window functionality
"""

import cv2
import numpy as np
from gradient_debug_window import GradientStackingDebugWindow
import customtkinter as ctk

def create_test_images():
    """Create test images with different focus regions."""
    base = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Image 1: Sharp left, blurry right
    img1 = base.copy()
    img1[100:200, 50:150] = [255, 100, 100]  # Red rectangle (sharp)
    img1[100:200, 250:350] = [100, 255, 100]  # Green rectangle (will be blurred)
    # Blur the right side
    img1[:, 200:] = cv2.GaussianBlur(img1[:, 200:], (15, 15), 5)
    
    # Image 2: Sharp right, blurry left
    img2 = base.copy()
    img2[100:200, 50:150] = [255, 100, 100]  # Red rectangle (will be blurred)
    img2[100:200, 250:350] = [100, 255, 100]  # Green rectangle (sharp)
    # Blur the left side
    img2[:, :200] = cv2.GaussianBlur(img2[:, :200], (15, 15), 5)
    
    # Image 3: Sharp center
    img3 = base.copy()
    img3[100:200, 150:250] = [100, 100, 255]  # Blue rectangle (sharp)
    # Blur the edges
    img3[:, :100] = cv2.GaussianBlur(img3[:, :100], (15, 15), 5)
    img3[:, 300:] = cv2.GaussianBlur(img3[:, 300:], (15, 15), 5)
    
    # Image 4: Sharp top-bottom, blurry middle
    img4 = base.copy()
    img4[50:100, 150:250] = [255, 255, 100]   # Top yellow (sharp)
    img4[200:250, 150:250] = [255, 100, 255]  # Bottom magenta (sharp)
    img4[100:200, 150:250] = [200, 200, 200]  # Middle gray (will be blurred)
    # Blur the middle
    img4[125:175, :] = cv2.GaussianBlur(img4[125:175, :], (15, 15), 5)
    
    return [img1, img2, img3, img4]

def test_debug_window():
    """Test the debug window with sample images."""
    print("Creating test images...")
    images = create_test_images()
    
    # Save images for reference
    for i, img in enumerate(images):
        cv2.imwrite(f'debug_test_img_{i+1}.png', img)
    
    print("Test images saved as debug_test_img_*.png")
    print("\nTo test the debug window:")
    print("1. Run the main GUI application")
    print("2. Load these test images")
    print("3. Select 'Gradient-based' stacking method")
    print("4. Click the 'Debug Process 🔍' button")
    print("5. Navigate through images to see (in 400x400 views):")
    print("   - Original image")
    print("   - Gradient map (edge detection)")
    print("   - Weight map (focus confidence)")
    print("   - Smooth weights (after blending)")

if __name__ == "__main__":
    test_debug_window()
