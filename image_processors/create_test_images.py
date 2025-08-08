#!/usr/bin/env python3
"""
Create sample images for testing the Interactive Image Cropper
"""

from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_test_images():
    """Create a variety of test images for the cropper"""
    test_dir = Path("/home/rich/MyCoding/image_processors/test_images")
    test_dir.mkdir(exist_ok=True)
    
    print(f"Creating test images in: {test_dir}")
    
    # Image 1: Landscape with pattern
    img1 = Image.new('RGB', (1200, 800), 'lightblue')
    draw1 = ImageDraw.Draw(img1)
    
    # Draw some patterns
    for i in range(0, 1200, 100):
        draw1.rectangle([i, 200, i+50, 600], fill='darkblue')
    
    for i in range(0, 800, 100):
        draw1.rectangle([300, i, 900, i+50], fill='red')
    
    # Add some circles
    for x in range(200, 1000, 200):
        for y in range(150, 650, 200):
            draw1.ellipse([x-30, y-30, x+30, y+30], fill='yellow')
    
    # Add text
    try:
        draw1.text((50, 50), "Test Image 1 - Landscape", fill='black')
        draw1.text((50, 750), "Bottom Text", fill='black')
    except:
        pass  # Font might not be available
    
    img1.save(test_dir / "test_landscape.jpg")
    print("✓ Created test_landscape.jpg (1200x800)")
    
    # Image 2: Portrait with gradient
    img2 = Image.new('RGB', (600, 1000), 'white')
    draw2 = ImageDraw.Draw(img2)
    
    # Create gradient effect
    for y in range(1000):
        color_val = int(255 * (y / 1000))
        draw2.line([(0, y), (600, y)], fill=(color_val, 0, 255-color_val))
    
    # Add some geometric shapes
    draw2.rectangle([100, 200, 500, 400], outline='white', width=5)
    draw2.ellipse([150, 600, 450, 800], fill='orange')
    
    try:
        draw2.text((50, 50), "Test Image 2 - Portrait", fill='white')
    except:
        pass
    
    img2.save(test_dir / "test_portrait.png")
    print("✓ Created test_portrait.png (600x1000)")
    
    # Image 3: Square with grid
    img3 = Image.new('RGB', (800, 800), 'lightgreen')
    draw3 = ImageDraw.Draw(img3)
    
    # Draw grid
    for i in range(0, 800, 50):
        draw3.line([(i, 0), (i, 800)], fill='darkgreen', width=2)
        draw3.line([(0, i), (800, i)], fill='darkgreen', width=2)
    
    # Add colored squares
    colors = ['red', 'blue', 'yellow', 'purple', 'orange']
    for i, color in enumerate(colors):
        x = (i % 3) * 200 + 100
        y = (i // 3) * 200 + 100
        draw3.rectangle([x, y, x+100, y+100], fill=color)
    
    try:
        draw3.text((50, 750), "Test Image 3 - Square Grid", fill='black')
    except:
        pass
    
    img3.save(test_dir / "test_square.jpg")
    print("✓ Created test_square.jpg (800x800)")
    
    # Image 4: Wide panoramic
    img4 = Image.new('RGB', (1600, 400), 'skyblue')
    draw4 = ImageDraw.Draw(img4)
    
    # Sky and ground
    draw4.rectangle([0, 200, 1600, 400], fill='green')
    
    # Add some "buildings"
    building_positions = [200, 400, 800, 1200, 1400]
    for pos in building_positions:
        height = pos % 100 + 50
        draw4.rectangle([pos-40, 200-height, pos+40, 200], fill='gray')
        draw4.rectangle([pos-35, 200-height+10, pos+35, 200-10], fill='lightgray')
    
    # Add sun
    draw4.ellipse([1450, 50, 1550, 150], fill='yellow')
    
    try:
        draw4.text((50, 350), "Test Image 4 - Panoramic", fill='black')
    except:
        pass
    
    img4.save(test_dir / "test_panoramic.png")
    print("✓ Created test_panoramic.png (1600x400)")
    
    # Image 5: Small image
    img5 = Image.new('RGB', (300, 200), 'pink')
    draw5 = ImageDraw.Draw(img5)
    
    # Simple pattern
    draw5.ellipse([50, 25, 250, 175], fill='purple')
    draw5.rectangle([100, 75, 200, 125], fill='white')
    
    try:
        draw5.text((10, 10), "Small Image", fill='black')
    except:
        pass
    
    img5.save(test_dir / "test_small.jpg")
    print("✓ Created test_small.jpg (300x200)")
    
    print(f"\nCreated 5 test images in {test_dir}")
    print("You can now test the Interactive Image Cropper with these images!")
    
    return test_dir


if __name__ == "__main__":
    test_dir = create_test_images()
    print(f"\nTo test the cropper, run:")
    print(f"python interactive_image_cropper.py")
    print(f"Then select the directory: {test_dir}")
