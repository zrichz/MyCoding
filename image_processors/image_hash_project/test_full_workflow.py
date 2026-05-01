#!/usr/bin/env python3
"""Full workflow test with actual encode/decode functions"""
import sys
sys.path.insert(0, '/home/rich/MyCoding/image_processors/image_hash_project')

from PIL import Image
import numpy as np

# Import the actual functions
from image_hash_gradio import encode_image, decode_image

# Create a test image with a clear pattern
test_img = Image.new('RGB', (50, 50))
pixels = test_img.load()

# Create a recognizable pattern (gradient)
for y in range(50):
    for x in range(50):
        pixels[x, y] = (x * 5, y * 5, 128)

print("Created test image with gradient pattern")
print(f"Original top-left pixel: {pixels[0, 0]}")
print(f"Original top-right pixel: {pixels[49, 0]}")
print(f"Original bottom-left pixel: {pixels[0, 49]}")

# Test with a seed
test_seed = "mytest123"

# Encode
print(f"\nEncoding with seed: '{test_seed}'")
encoded_img, encode_msg = encode_image(test_img, test_seed)
print(encode_msg)

# Save encoded image to disk
encoded_img.save('/tmp/test_encoded.png')
print("Saved encoded image to /tmp/test_encoded.png")

# Reload from disk (simulating user workflow)
reloaded_encoded = Image.open('/tmp/test_encoded.png')
print(f"Reloaded encoded image, mode: {reloaded_encoded.mode}, size: {reloaded_encoded.size}")

# Decode using the reloaded image
print(f"\nDecoding with seed: '{test_seed}'")
decoded_img, decode_msg = decode_image(reloaded_encoded, test_seed)
print(decode_msg)

# Compare original and decoded
orig_array = np.array(test_img)
decoded_array = np.array(decoded_img)

print(f"\nOriginal array shape: {orig_array.shape}")
print(f"Decoded array shape: {decoded_array.shape}")
print(f"Arrays equal: {np.array_equal(orig_array, decoded_array)}")
print(f"Max difference: {np.max(np.abs(orig_array.astype(int) - decoded_array.astype(int)))}")

# Check specific pixels
decoded_pixels = decoded_img.load()
print(f"\nDecoded top-left pixel: {decoded_pixels[0, 0]}")
print(f"Decoded top-right pixel: {decoded_pixels[49, 0]}")
print(f"Decoded bottom-left pixel: {decoded_pixels[0, 49]}")

# Visual check
if np.array_equal(orig_array, decoded_array):
    print("\n SUCCESS: Decoded image matches original perfectly")
else:
    print("\n ERROR: Decoded image does not match original")
    print(f"Number of different pixels: {np.sum(orig_array != decoded_array)}")
