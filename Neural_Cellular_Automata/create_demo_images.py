"""
Create a simple demo image for testing the Neural Cellular Automata
"""
import numpy as np
from PIL import Image
import os

def create_demo_images():
    """Create simple test images for the NCA"""
    
    # Create output directory
    os.makedirs("demo_images", exist_ok=True)
    
    # Simple circle
    size = 64
    center = size // 2
    radius = size // 4
    
    circle_img = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist <= radius:
                circle_img[y, x] = [255, 0, 0, 255]  # Red circle
            else:
                circle_img[y, x] = [0, 0, 0, 0]  # Transparent background
    
    Image.fromarray(circle_img).save("demo_images/red_circle.png")
    print("Created: demo_images/red_circle.png")
    
    # Simple square
    square_img = np.zeros((size, size, 4), dtype=np.uint8)
    start = size // 4
    end = 3 * size // 4
    square_img[start:end, start:end] = [0, 255, 0, 255]  # Green square
    
    Image.fromarray(square_img).save("demo_images/green_square.png")
    print("Created: demo_images/green_square.png")
    
    # Simple cross
    cross_img = np.zeros((size, size, 4), dtype=np.uint8)
    thickness = 8
    center_start = center - thickness // 2
    center_end = center + thickness // 2
    
    # Horizontal bar
    cross_img[center_start:center_end, :] = [0, 0, 255, 255]  # Blue
    # Vertical bar
    cross_img[:, center_start:center_end] = [0, 0, 255, 255]  # Blue
    
    Image.fromarray(cross_img).save("demo_images/blue_cross.png")
    print("Created: demo_images/blue_cross.png")
    
    # Gradient
    gradient_img = np.zeros((size, size, 4), dtype=np.uint8)
    for x in range(size):
        intensity = int(255 * x / size)
        gradient_img[:, x] = [intensity, intensity, intensity, 255]
    
    Image.fromarray(gradient_img).save("demo_images/gradient.png")
    print("Created: demo_images/gradient.png")

if __name__ == "__main__":
    create_demo_images()
