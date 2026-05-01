#!/usr/bin/env python3
"""Quick test with image mode conversion"""
from PIL import Image
import numpy as np
from image_hash_gradio import encode_image, decode_image

# Test with different image modes
for mode in ['RGB', 'RGBA', 'L']:
    print(f"\n=== Testing with {mode} image ===")
    
    # Create test image
    if mode == 'RGB':
        img = Image.new(mode, (30, 30), color=(255, 0, 0))
    elif mode == 'RGBA':
        img = Image.new(mode, (30, 30), color=(255, 0, 0, 255))
    else:  # L (grayscale)
        img = Image.new(mode, (30, 30), color=128)
    
    print(f"Original mode: {img.mode}")
    
    # Encode
    encoded, msg = encode_image(img, "test123")
    print(f"Encoded mode: {encoded.mode if encoded else 'None'}")
    
    # Decode
    decoded, msg = decode_image(encoded, "test123")
    print(f"Decoded mode: {decoded.mode if decoded else 'None'}")
    
    # Compare (convert original to RGB for comparison)
    orig_rgb = img.convert('RGB')
    orig_arr = np.array(orig_rgb)
    decoded_arr = np.array(decoded)
    
    matches = np.array_equal(orig_arr, decoded_arr)
    print(f"Match: {matches}")
    
    if not matches:
        diff = np.sum(orig_arr != decoded_arr)
        print(f"  Different pixels: {diff}")

print("\n=== All tests complete ===")
