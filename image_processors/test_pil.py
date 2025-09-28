#!/usr/bin/env python3
"""
Minimal PIL/ImageTk test
"""

try:
    from PIL import Image, ImageTk
    import tkinter as tk
    import numpy as np
    
    print("✓ All imports successful")
    
    # Test creating a simple image
    test_img = Image.new('RGB', (256, 256), color=(255, 0, 0))
    print(f"✓ Created test image: {test_img.size}")
    
    # Test numpy conversion
    img_array = np.array(test_img)
    print(f"✓ Numpy conversion: {img_array.shape}")
    
    # Test thumbnail
    test_img.thumbnail((128, 128), Image.Resampling.LANCZOS)
    print(f"✓ Thumbnail created: {test_img.size}")
    
    # Test ImageTk conversion
    photo = ImageTk.PhotoImage(test_img)
    print("✓ ImageTk conversion successful")
    
    print("All tests passed! PIL/ImageTk/NumPy working correctly.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
