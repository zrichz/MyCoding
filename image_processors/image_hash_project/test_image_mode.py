#!/usr/bin/env python3
"""Test if image mode is preserved through encode/decode"""
from PIL import Image
import numpy as np

# Create test images in different modes
test_rgb = Image.new('RGB', (10, 10), color='red')
test_rgba = Image.new('RGBA', (10, 10), color=(255, 0, 0, 255))
test_l = Image.new('L', (10, 10), color=128)

print("Original modes:")
print(f"RGB mode: {test_rgb.mode}")
print(f"RGBA mode: {test_rgba.mode}")
print(f"L (grayscale) mode: {test_l.mode}")

# Convert to numpy and back
for img, name in [(test_rgb, 'RGB'), (test_rgba, 'RGBA'), (test_l, 'L')]:
    arr = np.array(img)
    img_back = Image.fromarray(arr)
    print(f"\n{name} -> numpy -> PIL: {img_back.mode}")
    print(f"  Array shape: {arr.shape}, dtype: {arr.dtype}")

# Test save and reload
test_rgba.save('/tmp/test_rgba.png')
reloaded = Image.open('/tmp/test_rgba.png')
print(f"\nRGBA saved and reloaded: {reloaded.mode}")
