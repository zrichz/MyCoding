#!/usr/bin/env python3
"""
Test script to create a sample image and test the seam carving functionality
"""

import numpy as np
from PIL import Image, ImageDraw
import os

def create_test_image():
    """Create a test image with patterns to see seam carving effect"""
    width, height = 800, 600
    
    # Create a new image with RGB mode
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add some vertical stripes (should be preserved in middle)
    for x in range(0, width, 40):
        draw.rectangle([x, 0, x+20, height], fill='blue')
    
    # Add horizontal lines
    for y in range(0, height, 60):
        draw.rectangle([0, y, width, y+10], fill='red')
    
    # Add some detailed patterns in the edges (first and last 25%)
    quarter = width // 4
    
    # Left edge pattern (first 25%)
    for i in range(0, quarter, 20):
        for j in range(0, height, 30):
            draw.ellipse([i, j, i+15, j+25], fill='green')
    
    # Right edge pattern (last 25%)
    start_right = width - quarter
    for i in range(start_right, width, 20):
        for j in range(0, height, 30):
            draw.ellipse([i, j, i+15, j+25], fill='purple')
    
    # Save the test image
    test_image_path = '/home/rich/MyCoding/image_processors/test_image.png'
    image.save(test_image_path)
    print(f"Test image created: {test_image_path}")
    print(f"Image size: {width}x{height}")
    
    return test_image_path

if __name__ == "__main__":
    create_test_image()
