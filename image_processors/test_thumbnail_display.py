#!/usr/bin/env python3
"""
Test script to verify image loading and thumbnail creation
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk

def test_image_loading():
    """Test basic image loading functionality"""
    test_dir = Path('/home/rich/MyCoding/image_processors/test_images')
    
    if not test_dir.exists():
        print("✗ Test images directory not found")
        print("Run: python create_test_images.py")
        return False
    
    # Find image files
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    image_files = []
    
    for file_path in test_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(file_path)
    
    if not image_files:
        print("✗ No image files found in test directory")
        return False
    
    print(f"✓ Found {len(image_files)} test images")
    
    # Test loading first image
    test_image_path = image_files[0]
    print(f"Testing with: {test_image_path.name}")
    
    try:
        # Load original image
        original_image = Image.open(test_image_path)
        print(f"✓ Loaded original image: {original_image.size}")
        
        # Create thumbnail
        thumbnail = original_image.copy()
        thumbnail.thumbnail((800, 800), Image.Resampling.LANCZOS)
        print(f"✓ Created thumbnail: {thumbnail.size}")
        
        # Test PhotoImage creation (requires tkinter)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        photo = ImageTk.PhotoImage(thumbnail)
        print(f"✓ Created PhotoImage successfully")
        
        root.destroy()
        
        return True
        
    except Exception as e:
        print(f"✗ Error during image processing: {e}")
        return False

def test_gui_components():
    """Test basic GUI component creation"""
    try:
        root = tk.Tk()
        root.withdraw()
        
        # Test canvas creation
        canvas = tk.Canvas(root, bg="white", width=800, height=600)
        print("✓ Canvas created successfully")
        
        # Test image loading in GUI context
        test_dir = Path('/home/rich/MyCoding/image_processors/test_images')
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        if image_files:
            image = Image.open(image_files[0])
            thumbnail = image.copy()
            thumbnail.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumbnail)
            
            # Test placing image on canvas
            canvas.create_image(10, 10, anchor="nw", image=photo)
            print("✓ Image placed on canvas successfully")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ GUI component test failed: {e}")
        return False

def main():
    print("Testing Image Cropper Components...")
    print()
    
    image_test = test_image_loading()
    gui_test = test_gui_components()
    
    print()
    if image_test and gui_test:
        print("✓ All tests passed!")
        print("✓ Image thumbnails should work properly")
        print()
        print("If thumbnails still don't appear in the GUI:")
        print("1. Check console output when running the cropper")
        print("2. Try selecting the test_images directory")
        print("3. Look for any error messages")
    else:
        print("✗ Some tests failed")
        print("Issues need to be resolved before thumbnails will work")
    
    return image_test and gui_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
