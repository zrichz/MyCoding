#!/usr/bin/env python3
"""
Simple test script to verify the Interactive Image Cropper can start
"""

import sys
import os
sys.path.insert(0, '/home/rich/MyCoding/image_processors')

def test_cropper_import():
    try:
        from interactive_image_cropper import ImageCropper
        print("✓ ImageCropper imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dependencies():
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from PIL import Image, ImageTk
        from pathlib import Path
        print("✓ All dependencies available")
        return True
    except Exception as e:
        print(f"✗ Dependencies missing: {e}")
        return False

def main():
    print("Testing Interactive Image Cropper...")
    print()
    
    deps_ok = test_dependencies()
    import_ok = test_cropper_import()
    
    if deps_ok and import_ok:
        print()
        print("✓ All tests passed!")
        print("✓ Ready to run: python interactive_image_cropper.py")
        print("✓ Test images available in: test_images/")
        
        # Check if test images exist
        test_dir = '/home/rich/MyCoding/image_processors/test_images'
        if os.path.exists(test_dir):
            image_count = len([f for f in os.listdir(test_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.gif'))])
            print(f"✓ {image_count} test images found")
        else:
            print("! Run create_test_images.py to generate test images")
    else:
        print()
        print("✗ Some tests failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
