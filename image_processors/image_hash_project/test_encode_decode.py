#!/usr/bin/env python3
"""Test the encode/decode logic"""
import numpy as np
from PIL import Image
import random

def generate_lookup_table(width, height, seed):
    total_pixels = width * height
    positions = list(range(total_pixels))
    
    if isinstance(seed, str):
        seed_int = hash(seed)
    else:
        seed_int = seed
    
    rng = random.Random(seed_int)
    rng.shuffle(positions)
    
    return positions

# Create a small test image (4x4)
test_array = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
    [[255, 0, 255], [0, 255, 255], [128, 128, 128], [255, 128, 0]],
    [[0, 128, 255], [128, 0, 255], [255, 255, 255], [0, 0, 0]],
    [[128, 128, 0], [0, 128, 128], [128, 0, 128], [64, 64, 64]]
], dtype=np.uint8)

print("Original image:")
print(test_array)

width, height = 4, 4
seed = "test123"

# Generate lookup table
lookup_table = generate_lookup_table(width, height, seed)
print(f"\nLookup table: {lookup_table}")

# ENCODE
encoded_array = np.zeros_like(test_array)
for px in range(width * height):
    ox = px % width
    oy = px // width
    new_pos = lookup_table[px]
    nx = new_pos % width
    ny = new_pos // width
    encoded_array[ny, nx] = test_array[oy, ox]

print("\nEncoded image:")
print(encoded_array)

# DECODE
decoded_array = np.zeros_like(encoded_array)
for px in range(width * height):
    ox = px % width
    oy = px // width
    encoded_pos = lookup_table[px]
    ex = encoded_pos % width
    ey = encoded_pos // width
    decoded_array[oy, ox] = encoded_array[ey, ex]

print("\nDecoded image:")
print(decoded_array)

print("\nAre original and decoded identical?", np.array_equal(test_array, decoded_array))
