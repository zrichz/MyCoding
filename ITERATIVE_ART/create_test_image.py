"""
Simple test script to verify image processing functionality
"""
from PIL import Image, ImageDraw
import os

# Create a simple test image
def create_test_image():
    # Create a 400x200 test image (2:1 ratio)
    img = Image.new('RGB', (400, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw some colorful patterns
    draw.rectangle([0, 0, 200, 200], fill=(255, 100, 100))  # Red left square
    draw.rectangle([200, 0, 400, 200], fill=(100, 255, 100))  # Green right square
    
    # Add some details
    draw.ellipse([50, 50, 150, 150], fill=(255, 255, 100))  # Yellow circle in left
    draw.rectangle([250, 50, 350, 150], fill=(100, 100, 255))  # Blue rect in right
    
    return img

if __name__ == "__main__":
    # Create test image
    test_img = create_test_image()
    test_path = "test_image_2to1.png"
    test_img.save(test_path)
    print(f"✓ Created test image: {test_path}")
    print(f"  Size: {test_img.size[0]}x{test_img.size[1]} (ratio: {test_img.size[0]/test_img.size[1]:.1f}:1)")
    print(f"  Use this image to test the interactive L-System fern generator!")
    print(f"  1. Run: python iterative_L_System_fern.py")
    print(f"  2. Click 'Load Image' and select '{test_path}'")
    print(f"  3. Switch to 'Image Processing' mode")
    print(f"  4. Adjust parameters and see the L-System effect!")