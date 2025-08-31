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
    size = 128  # Larger size for better detail
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
    
    # Gradient with more detail
    gradient_img = np.zeros((size, size, 4), dtype=np.uint8)
    for x in range(size):
        for y in range(size):
            # Create a more complex gradient pattern
            intensity_x = int(255 * x / size)
            intensity_y = int(255 * y / size)
            gradient_img[y, x] = [intensity_x, intensity_y, 128, 255]
    
    Image.fromarray(gradient_img).save("demo_images/gradient.png")
    print("Created: demo_images/gradient.png")
    
    # Complex pattern for testing 256x256 capability
    complex_size = 256
    complex_img = np.zeros((complex_size, complex_size, 4), dtype=np.uint8)
    center_x, center_y = complex_size // 2, complex_size // 2
    
    # Create concentric circles with different colors
    for y in range(complex_size):
        for x in range(complex_size):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= 30:
                complex_img[y, x] = [255, 0, 0, 255]  # Red center
            elif dist <= 60:
                complex_img[y, x] = [255, 255, 0, 255]  # Yellow ring
            elif dist <= 90:
                complex_img[y, x] = [0, 255, 0, 255]  # Green ring
            elif dist <= 120:
                complex_img[y, x] = [0, 255, 255, 255]  # Cyan ring
            else:
                complex_img[y, x] = [0, 0, 0, 0]  # Transparent background
    
    Image.fromarray(complex_img).save("demo_images/complex_pattern.png")
    print("Created: demo_images/complex_pattern.png")

if __name__ == "__main__":
    create_demo_images()
