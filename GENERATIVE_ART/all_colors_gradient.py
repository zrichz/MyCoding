"""
All Colors Gradient Generator

Generates a 1024x1024 image containing smooth gradients across color space.
Uses HSV color space to create visually appealing smooth transitions.
"""

import numpy as np
from PIL import Image
import colorsys
import os


def generate_hsv_gradient(width=1024, height=1024, hue_cycles=1.0, mode='circular'):
    """
    Generate an image with smooth color gradients using HSV color space
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        hue_cycles: Number of complete hue cycles (1.0 = full rainbow)
        mode: Gradient mode - 'circular', 'rectangular', 'diagonal', or 'radial'
    
    Returns:
        PIL Image
    """
    # Create coordinate arrays
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Normalize coordinates to 0-1 range
    x_norm = x_coords / (width - 1)
    y_norm = y_coords / (height - 1)
    
    if mode == 'circular':
        # Circular gradient: hue varies with angle, saturation with radius
        center_x = width / 2
        center_y = height / 2
        
        # Calculate angle (hue) and distance (saturation)
        dx = x_coords - center_x
        dy = y_coords - center_y
        
        angle = np.arctan2(dy, dx)  # -pi to pi
        hue = (angle / (2 * np.pi) + 0.5) * hue_cycles  # Normalize to 0-1
        
        distance = np.sqrt(dx**2 + dy**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        saturation = np.clip(distance / max_distance, 0, 1)
        
        value = np.ones_like(hue)  # Full brightness
        
    elif mode == 'rectangular':
        # Rectangular gradient: hue on x-axis, saturation on y-axis
        hue = x_norm * hue_cycles
        saturation = y_norm
        value = np.ones_like(hue)
        
    elif mode == 'diagonal':
        # Diagonal gradient: smooth transition from corner to corner
        hue = (x_norm + y_norm) / 2 * hue_cycles
        saturation = np.abs(x_norm - y_norm)
        value = np.ones_like(hue)
        
    elif mode == 'radial':
        # Radial gradient: hue varies with angle, value with radius
        center_x = width / 2
        center_y = height / 2
        
        dx = x_coords - center_x
        dy = y_coords - center_y
        
        angle = np.arctan2(dy, dx)
        hue = (angle / (2 * np.pi) + 0.5) * hue_cycles
        
        distance = np.sqrt(dx**2 + dy**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        value = 1.0 - np.clip(distance / max_distance, 0, 1)
        
        saturation = np.ones_like(hue)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Wrap hue to 0-1 range
    hue = hue % 1.0
    
    # Convert HSV to RGB
    # Vectorized conversion for efficiency
    h = hue.flatten()
    s = saturation.flatten()
    v = value.flatten()
    
    # Convert each HSV triplet to RGB
    rgb_pixels = np.array([colorsys.hsv_to_rgb(h[i], s[i], v[i]) 
                           for i in range(len(h))])
    
    # Reshape back to image dimensions
    rgb_image = rgb_pixels.reshape(height, width, 3)
    
    # Convert to 8-bit integer values
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return Image.fromarray(rgb_image)


def generate_multi_gradient(width=1024, height=1024):
    """
    Generate a complex multi-dimensional gradient
    Combines multiple color spaces for maximum variety
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Create three independent gradients for R, G, B
    # Using sine waves for smooth periodic transitions
    x_norm = x_coords / width * 2 * np.pi
    y_norm = y_coords / height * 2 * np.pi
    
    # Red channel: horizontal gradient with sine modulation
    red = (np.sin(x_norm) * 0.5 + 0.5) * (np.cos(y_norm * 0.5) * 0.5 + 0.5)
    
    # Green channel: vertical gradient with sine modulation
    green = (np.sin(y_norm) * 0.5 + 0.5) * (np.cos(x_norm * 0.5) * 0.5 + 0.5)
    
    # Blue channel: diagonal gradient
    blue = (np.sin(x_norm + y_norm) * 0.5 + 0.5)
    
    # Combine into RGB image
    rgb_image = np.stack([red, green, blue], axis=2)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return Image.fromarray(rgb_image)


def generate_hilbert_colors(width=1024, height=1024):
    """
    Generate colors arranged along a Hilbert space-filling curve
    This creates a smooth path through color space
    """
    def hilbert_index_to_xy(index, order):
        """Convert Hilbert curve index to x,y coordinates"""
        x = y = 0
        s = 1
        while s < (1 << order):
            rx = 1 & (index // 2)
            ry = 1 & (index ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            index //= 4
            s *= 2
        return x, y
    
    # Determine Hilbert curve order (must be power of 2)
    order = int(np.log2(width))
    size = 2 ** order
    
    # Create image array
    img_array = np.zeros((size, size, 3), dtype=np.uint8)
    
    total_pixels = size * size
    
    # Map colors along Hilbert curve
    for i in range(total_pixels):
        x, y = hilbert_index_to_xy(i, order)
        
        # Map position along curve to hue
        hue = i / total_pixels
        saturation = 0.8 + 0.2 * np.sin(i / total_pixels * 4 * np.pi)
        value = 0.8 + 0.2 * np.cos(i / total_pixels * 4 * np.pi)
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        img_array[y, x] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return Image.fromarray(img_array)


def main():
    """Generate and save various gradient images"""
    
    output_dir = "/home/rich/MyCoding/fractal_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating color gradient images...")
    
    # Generate circular gradient (hue wheel)
    print("1. Generating circular gradient (color wheel)...")
    img_circular = generate_hsv_gradient(1024, 1024, mode='circular')
    img_circular.save(os.path.join(output_dir, "colors_circular_gradient.png"))
    print("   Saved: colors_circular_gradient.png")
    
    # Generate rectangular gradient
    print("2. Generating rectangular gradient...")
    img_rect = generate_hsv_gradient(1024, 1024, mode='rectangular')
    img_rect.save(os.path.join(output_dir, "colors_rectangular_gradient.png"))
    print("   Saved: colors_rectangular_gradient.png")
    
    # Generate radial gradient
    print("3. Generating radial gradient...")
    img_radial = generate_hsv_gradient(1024, 1024, mode='radial')
    img_radial.save(os.path.join(output_dir, "colors_radial_gradient.png"))
    print("   Saved: colors_radial_gradient.png")
    
    # Generate multi-gradient
    print("4. Generating multi-dimensional gradient...")
    img_multi = generate_multi_gradient(1024, 1024)
    img_multi.save(os.path.join(output_dir, "colors_multi_gradient.png"))
    print("   Saved: colors_multi_gradient.png")
    
    # Generate Hilbert curve gradient
    print("5. Generating Hilbert curve gradient...")
    img_hilbert = generate_hilbert_colors(1024, 1024)
    img_hilbert.save(os.path.join(output_dir, "colors_hilbert_gradient.png"))
    print("   Saved: colors_hilbert_gradient.png")
    
    print("\nAll images generated successfully in fractal_outputs/")


if __name__ == "__main__":
    main()
