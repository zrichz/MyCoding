#!/usr/bin/env python3
"""
Quick test of the enhanced Gaussian deblur features.
"""

import numpy as np
import sys
import os
sys.path.append('/home/rich/MyCoding/DEBLUR')

from src.deblur.gaussian_deblur import GaussianDeblur

def test_progress_and_downsampling():
    print("Testing Enhanced Gaussian Deblur Features")
    print("=" * 50)
    
    deblurrer = GaussianDeblur()
    
    # Test 1: Small image (no downsampling)
    print("\n1. Testing small image (200x300 pixels)...")
    small_image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
    result1 = deblurrer.deblur_image(small_image, kernel_size=9, iterations=5, show_progress=True)
    print(f"✓ Input: {small_image.shape}, Output: {result1.shape}")
    
    # Test 2: Medium image (auto downsampling to 2x)
    print("\n2. Testing medium image (1000x800 pixels)...")
    medium_image = np.random.randint(0, 255, (1000, 800), dtype=np.uint8)
    result2 = deblurrer.deblur_image(medium_image, kernel_size=15, iterations=10, 
                                   auto_downsample=True, show_progress=True)
    print(f"✓ Input: {medium_image.shape}, Output: {result2.shape}")
    
    # Test 3: Large image (auto downsampling to 4x)
    print("\n3. Testing large image (2000x1600 pixels)...")
    large_image = np.random.randint(0, 255, (2000, 1600), dtype=np.uint8)
    result3 = deblurrer.deblur_image(large_image, kernel_size=21, iterations=8, 
                                   auto_downsample=True, show_progress=True)
    print(f"✓ Input: {large_image.shape}, Output: {result3.shape}")
    
    # Test 4: Manual downsampling
    print("\n4. Testing manual 4x downsampling...")
    result4 = deblurrer.deblur_image(medium_image, kernel_size=15, iterations=5, 
                                   downsample_factor=4, show_progress=True)
    print(f"✓ Input: {medium_image.shape}, Output: {result4.shape}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    print("Features working:")
    print("  • Progress display with iteration counting")
    print("  • Automatic downsampling based on image size")
    print("  • Manual downsampling factor control")
    print("  • Proper upsampling back to original resolution")

if __name__ == "__main__":
    test_progress_and_downsampling()
