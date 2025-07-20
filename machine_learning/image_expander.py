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

def add_white_noise_to_padding(original_image, square_image):
    """Add white noise only to the padding areas."""
    width, height = original_image.size
    sq_width, sq_height = square_image.size
    
    # Create a mask where white areas will be the padding
    mask = Image.new("L", (sq_width, sq_height), 0)  # Start with black
    
    # Calculate position of the original image
    if width > height:
        top_padding = (sq_height - height) // 2
        mask.paste(255, (0, 0, sq_width, top_padding))  # Top padding
        mask.paste(255, (0, top_padding + height, sq_width, sq_height))  # Bottom padding
    else:
        left_padding = (sq_width - width) // 2
        mask.paste(255, (0, 0, left_padding, sq_height))  # Left padding
        mask.paste(255, (left_padding + width, 0, sq_width, sq_height))  # Right padding
    
    # Generate noise for the entire image
    noise = np.random.randint(0, 256, (sq_height, sq_width, 3), dtype=np.uint8)
    noise_image = Image.fromarray(noise, mode="RGB")
    
    # Combine the original image and noise using the mask
    result = Image.composite(noise_image, square_image, mask)
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
        
        # Add white noise to the padded areas
        final_image = add_white_noise_to_padding(resized_original, resized_image)
        
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