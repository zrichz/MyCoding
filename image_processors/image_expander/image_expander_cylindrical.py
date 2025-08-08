"""
Image Expander Script

This script processes multiple images from a directory to create square versions:
1. Converts any image to a 1:1 aspect ratio by adding padding
2. Resizes each image to 512x512 pixels
3. Fills the padding areas with random white noise
4. Preserves the original image content in its original aspect ratio
5. Saves the resulting images as PNGs in a "512x512_images" subdirectory

Usage:
- Run the script
- Select a directory containing images when prompted
- All processed images will be saved in a "512x512_images" subdirectory
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
import os
import glob

def create_square_image(image, target_size):
    """Create a square image by adding black padding."""
    width, height = image.size
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    
    if width > height:
        # Center the image vertically
        top_padding = (target_size - height) // 2
        new_image.paste(image, (0, top_padding))
    else:
        # Center the image horizontally
        left_padding = (target_size - width) // 2
        new_image.paste(image, (left_padding, 0))
    
    return new_image

def apply_inverse_cylindrical_transform(original_image, square_image):
    """
    Transform the original image to fit a square aspect ratio using inverse cylindrical transformation.
    This simulates unwrapping an image from a cylinder to create a square image.
    """
    width, height = original_image.size
    sq_width, sq_height = square_image.size
    is_landscape = width > height
    
    # Create a new square image
    result = Image.new("RGB", (sq_width, sq_height), (0, 0, 0))
    
    if is_landscape:
        # For landscape images: wrap around vertical cylinder and expand vertically
        top_padding = (sq_height - height) // 2
        
        # Create source image with the original content centered
        source_img = Image.new("RGB", (sq_width, sq_height), (0, 0, 0))
        source_img.paste(original_image, (0, top_padding))
        
        # Number of output strips
        num_strips = 200
        strip_height = max(1, sq_height // num_strips)
        
        # Calculate the angle range for mapping
        max_angle = np.pi / 2  # 90 degrees in radians
        
        for i in range(num_strips):
            y_start = i * strip_height
            y_end = min((i + 1) * strip_height, sq_height)
            
            # Calculate position in normalized coordinates (-1 to 1) from center
            y_mid = (y_start + y_end) / 2
            norm_pos = 2 * (y_mid / sq_height - 0.5)  # -1 at top, 0 at center, 1 at bottom
            
            # Calculate angle on cylinder
            angle = norm_pos * max_angle  # Ranges from -π/2 to +π/2
            
            # Calculate stretch factor using proper cylindrical projection formula
            # cos(angle) gives the compression factor when projecting onto cylinder
            # 1/cos(angle) gives the stretching factor when unwrapping
            stretch_factor = 1.0 / max(0.001, np.cos(angle))  # Limit to avoid infinite stretch
            
            # Source height needs to be smaller by the stretch factor
            source_height = int(strip_height / stretch_factor)
            
            # Center the sampling around the corresponding point on the cylinder
            source_center = sq_height / 2
            source_y_mid = source_center + (y_mid - source_center) / stretch_factor
            source_y_start = int(source_y_mid - source_height / 2)
            source_y_end = source_y_start + source_height
            
            # Ensure source coordinates are within bounds
            source_y_start = max(0, min(source_y_start, sq_height - source_height))
            source_y_end = min(sq_height, source_y_start + source_height)
            
            # Extract strip from source
            if source_y_end > source_y_start:
                strip = source_img.crop((0, source_y_start, sq_width, source_y_end))
                
                # Resize to destination size and paste
                strip = strip.resize((sq_width, y_end - y_start), Image.LANCZOS)
                result.paste(strip, (0, y_start))
    
    else:
        # For portrait images: wrap around horizontal cylinder and expand horizontally
        left_padding = (sq_width - width) // 2
        
        # Create source image with the original content centered
        source_img = Image.new("RGB", (sq_width, sq_height), (0, 0, 0))
        source_img.paste(original_image, (left_padding, 0))
        
        # Number of output strips
        num_strips = 200
        strip_width = max(1, sq_width // num_strips)
        
        # Calculate the angle range for mapping
        max_angle = np.pi / 2  # 90 degrees in radians
        
        for i in range(num_strips):
            x_start = i * strip_width
            x_end = min((i + 1) * strip_width, sq_width)
            
            # Calculate position in normalized coordinates (-1 to 1) from center
            x_mid = (x_start + x_end) / 2
            norm_pos = 2 * (x_mid / sq_width - 0.5)  # -1 at left, 0 at center, 1 at right
            
            # Calculate angle on cylinder
            angle = norm_pos * max_angle  # Ranges from -π/2 to +π/2
            
            # Calculate stretch factor using proper cylindrical projection formula
            stretch_factor = 1.0 / max(0.001, np.cos(angle))  # Limit to avoid infinite stretch
            
            # Source width needs to be smaller by the stretch factor
            source_width = int(strip_width / stretch_factor)
            
            # Center the sampling around the corresponding point on the cylinder
            source_center = sq_width / 2
            source_x_mid = source_center + (x_mid - source_center) / stretch_factor
            source_x_start = int(source_x_mid - source_width / 2)
            source_x_end = source_x_start + source_width
            
            # Ensure source coordinates are within bounds
            source_x_start = max(0, min(source_x_start, sq_width - source_width))
            source_x_end = min(sq_width, source_x_start + source_width)
            
            # Extract strip from source
            if source_x_end > source_x_start:
                strip = source_img.crop((source_x_start, 0, source_x_end, sq_height))
                
                # Resize to destination size and paste
                strip = strip.resize((x_end - x_start, sq_height), Image.LANCZOS)
                result.paste(strip, (x_start, 0))
    
    return result

def process_image(image_path, output_dir):
    """Process a single image and save to the output directory."""
    try:
        # Load the image
        original_image = Image.open(image_path)
        width, height = original_image.size
        
        # Determine the largest dimension
        largest_dimension = max(width, height)
        
        # Create a 1:1 image with black padding
        square_image = create_square_image(original_image, largest_dimension)
        
        # Resize the image to 512x512 pixels
        resized_image = square_image.resize((512, 512), Image.LANCZOS)
        
        # Calculate the new dimensions of the original content after resizing
        if width > height:
            new_height = int((height / width) * 512)
            resized_original = original_image.resize((512, new_height), Image.LANCZOS)
        else:
            new_width = int((width / height) * 512)
            resized_original = original_image.resize((new_width, 512), Image.LANCZOS)
        
        # Apply inverse cylindrical transformation instead of adding white noise
        final_image = apply_inverse_cylindrical_transform(resized_original, resized_image)
        
        # Save the new image to the output directory
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_sq.png")
        final_image.save(save_path)
        print(f"Processed: {filename} -> {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    # Create a Tkinter GUI to select a directory
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    input_dir = filedialog.askdirectory(title="Select Directory with Images")
    
    if not input_dir:
        print("No directory selected.")
        return
    
    # Create output directory
    output_dir = os.path.join(input_dir, "512x512_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files (common formats)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process all images
    success_count = 0
    for image_path in image_files:
        if process_image(image_path, output_dir):
            success_count += 1
    
    print(f"Processing complete: {success_count} of {len(image_files)} images successfully processed")
    print(f"Results saved in: {output_dir}")
    
if __name__ == "__main__":
    main()