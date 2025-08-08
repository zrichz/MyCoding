#!/usr/bin/env python3
"""
Quick verification that thumbnails appear in the Interactive Image Cropper
"""

import sys
import os
from pathlib import Path

def main():
    print("🔍 Verifying Interactive Image Cropper...")
    
    # Check if test images exist
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("❌ test_images directory not found")
        return False
    
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not images:
        print("❌ No test images found")
        return False
    
    print(f"✓ Found {len(images)} test images")
    
    # Test ImageTk import
    try:
        from PIL import Image, ImageTk
        print("✓ PIL and ImageTk imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test basic image loading and thumbnail creation
    try:
        import tkinter as tk
        
        test_image = images[0]
        img = Image.open(test_image)
        print(f"✓ Loaded test image: {test_image.name} ({img.size})")
        
        # Create thumbnail
        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        print(f"✓ Created thumbnail: {img.size}")
        
        # Test PhotoImage creation (needs tkinter root)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        photo = ImageTk.PhotoImage(img)
        print("✓ PhotoImage created successfully")
        root.destroy()
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return False
    
    print("\n🎉 All checks passed!")
    print("\n📝 To test the full application:")
    print("1. Run: python3 interactive_image_cropper.py")
    print("2. Click 'Select Directory'")
    print("3. Choose the 'test_images' folder")
    print("4. Thumbnails should now appear on the canvas")
    print("5. Click and drag to select crop area")
    print("6. Click 'Crop Current Image' to save the crop")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
