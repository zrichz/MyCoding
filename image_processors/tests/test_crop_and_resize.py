#!/usr/bin/env python3
"""
Test the complete crop and resize functionality
"""

from PIL import Image
import sys
import os
from pathlib import Path

def test_crop_and_resize():
    """Test cropping and resizing together"""
    
    # Import the ImageCropper class
    sys.path.append('.')
    from interactive_image_cropper import ImageCropper
    
    # Create a dummy root and cropper instance
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    cropper = ImageCropper(root)
    
    print("Testing Complete Crop and Resize Functionality...")
    print("=" * 55)
    
    # Create test images with different sizes
    test_cases = [
        # (original_size, crop_box, expected_behavior)
        ((2000, 1500), (0, 0, 1000, 1000), "Large image cropped, then resized down"),
        ((800, 600), (100, 100, 600, 500), "Medium image cropped, no resize needed"),
        ((400, 300), (50, 50, 350, 250), "Small image cropped, then resized up"),
        ((1000, 400), (200, 100, 800, 300), "Wide image cropped, then resized down"),
        ((300, 2000), (50, 400, 250, 1600), "Tall image cropped, then resized down"),
    ]
    
    for i, ((orig_w, orig_h), (x1, y1, x2, y2), description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        
        # Create test image
        test_image = Image.new('RGB', (orig_w, orig_h), color=(255, 100, 100))
        print(f"  Original: {orig_w}×{orig_h}")
        
        # Simulate crop
        cropped = test_image.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size
        print(f"  Cropped: {crop_w}×{crop_h}")
        
        # Apply resizing rules
        final_image = cropper.apply_resizing_rules(cropped)
        final_w, final_h = final_image.size
        print(f"  Final: {final_w}×{final_h}")
        
        # Check if resizing was applied
        resize_applied = hasattr(cropper, '_resize_applied') and cropper._resize_applied
        print(f"  Resized: {'Yes' if resize_applied else 'No'}")
        
        # Verify constraints
        if final_w <= 720 and final_h <= 1600:
            print(f"  ✅ Within size constraints")
        else:
            print(f"  ❌ Exceeds size constraints: {final_w}×{final_h}")
    
    root.destroy()
    print(f"\n✅ Complete crop and resize test completed!")

if __name__ == "__main__":
    try:
        test_crop_and_resize()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
