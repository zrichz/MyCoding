#!/usr/bin/env python3
"""Quick test for focus stacking algorithms."""

import cv2
import numpy as np
from focus_stacking_algorithms import FocusStackingAlgorithms

def test_laplacian():
    print('Testing Laplacian Pyramid...')
    
    # Create simple test images
    img1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    images = [img1, img2]
    
    try:
        result = FocusStackingAlgorithms.laplacian_pyramid_stack(images, levels=3, sigma=1.0)
        print(f'✓ Success! Result shape: {result.shape}')
        cv2.imwrite('test_laplacian_result.png', result)
        print('✓ Saved test result')
        return True
    except Exception as e:
        print(f'✗ Error: {e}')
        return False

def test_gradient():
    print('Testing Gradient-based stacking...')
    
    # Create simple test images
    img1 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    images = [img1, img2]
    
    def progress_cb(msg):
        print(f"  {msg}")
    
    try:
        result = FocusStackingAlgorithms.gradient_based_stack(images, progress_callback=progress_cb)
        print(f'✓ Success! Result shape: {result.shape}')
        cv2.imwrite('test_gradient_result.png', result)
        print('✓ Saved test result')
        return True
    except Exception as e:
        print(f'✗ Error: {e}')
        return False

if __name__ == "__main__":
    test_laplacian()
    print()
    test_gradient()
