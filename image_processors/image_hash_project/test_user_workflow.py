#!/usr/bin/env python3
"""Simulate exact user workflow"""
from PIL import Image
import numpy as np
from image_hash_gradio import encode_image, decode_image

# Create a test image with recognizable content
print("Step 1: Creating original test image...")
original = Image.new('RGB', (100, 100))
pixels = original.load()
# Create a distinctive pattern (checkerboard)
for y in range(100):
    for x in range(100):
        if (x // 10 + y // 10) % 2 == 0:
            pixels[x, y] = (255, 0, 0)  # Red
        else:
            pixels[x, y] = (0, 0, 255)  # Blue

original.save('/tmp/original.png')
print("   Saved original to /tmp/original.png")

# Step 2: Encode (user uploads original to Encode tab)
print("\nStep 2: Encoding image with seed 'mypass123'...")
seed = "mypass123"
encoded, msg = encode_image(original, seed)
print(f"   {msg}")

# Step 3: Save encoded image (user clicks 'Save Scattered Image')
encoded.save('/tmp/encoded.png')
print("   Saved encoded to /tmp/encoded.png")

# Step 4: User opens Decode tab and uploads the saved encoded image
print("\nStep 4: Loading encoded image (simulating user upload to Decode tab)...")
uploaded_encoded = Image.open('/tmp/encoded.png')
print(f"   Loaded image: {uploaded_encoded.mode}, {uploaded_encoded.size}")

# Step 5: Decode with same seed
print(f"\nStep 5: Decoding with seed '{seed}'...")
decoded, msg = decode_image(uploaded_encoded, seed)
print(f"   {msg}")

# Step 6: Compare
print("\nStep 6: Comparing original vs decoded...")
original_array = np.array(original)
decoded_array = np.array(decoded)

matches = np.array_equal(original_array, decoded_array)
print(f"   Arrays identical: {matches}")

if matches:
    print("\n   SUCCESS: Decode perfectly reproduces original image")
else:
    diff_pixels = np.sum(np.any(original_array != decoded_array, axis=2))
    total_pixels = original_array.shape[0] * original_array.shape[1]
    print(f"\n   ERROR: {diff_pixels}/{total_pixels} pixels differ")
    print(f"   Max difference: {np.max(np.abs(original_array.astype(int) - decoded_array.astype(int)))}")

# Save decoded for manual inspection
decoded.save('/tmp/decoded.png')
print("\n   Saved decoded to /tmp/decoded.png for manual inspection")

# Test with wrong seed
print("\n\nBonus Test: Decoding with WRONG seed 'wrongpass'...")
wrong_decoded, msg = decode_image(uploaded_encoded, "wrongpass")
print(f"   {msg}")
wrong_matches = np.array_equal(original_array, np.array(wrong_decoded))
print(f"   Matches original: {wrong_matches} (should be False)")
