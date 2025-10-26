"""
Image Montage Creator
Processes all images in a directory, resizes them to 1024x1024, 
and combines 4 images at a time into 2048x2048 montages.
"""

import os
from PIL import Image
from pathlib import Path
import argparse


def get_image_files(directory):
    """Get all image files from the specified directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def resize_image(image_path, target_size=(1024, 1024)):
    """Resize an image to the target size while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize while maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create a new image with the exact target size and paste the resized image
            resized_img = Image.new('RGB', target_size, (255, 255, 255))  # White background
            
            # Calculate position to center the image
            x_offset = (target_size[0] - img.width) // 2
            y_offset = (target_size[1] - img.height) // 2
            
            resized_img.paste(img, (x_offset, y_offset))
            return resized_img
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def create_montage(images, montage_size=(2048, 2048)):
    """Create a 2x2 montage from 4 images"""
    if len(images) != 4:
        raise ValueError("Exactly 4 images are required for a 2x2 montage")
    
    # Create the montage canvas
    montage = Image.new('RGB', montage_size, (255, 255, 255))
    
    # Position each image in the 2x2 grid
    positions = [
        (0, 0),        # Top-left
        (1024, 0),     # Top-right
        (0, 1024),     # Bottom-left
        (1024, 1024)   # Bottom-right
    ]
    
    for i, img in enumerate(images):
        if img is not None:
            montage.paste(img, positions[i])
    
    return montage


def process_images(input_directory, output_directory=None):
    """Main function to process all images and create montages"""
    
    # Set output directory
    if output_directory is None:
        output_directory = Path(input_directory) / "montages"
    else:
        output_directory = Path(output_directory)
    
    # Create output directory if it doesn't exist
    output_directory.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(input_directory)
    
    if not image_files:
        print(f"No image files found in {input_directory}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Output directory: {output_directory}")
    
    # Process images in groups of 4
    montage_count = 0
    resized_images = []
    
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
        
        # Resize the image
        resized_img = resize_image(image_path)
        
        if resized_img is not None:
            resized_images.append(resized_img)
        else:
            # If image failed to process, add a white placeholder
            placeholder = Image.new('RGB', (1024, 1024), (255, 255, 255))
            resized_images.append(placeholder)
        
        # Create montage when we have 4 images
        if len(resized_images) == 4:
            montage_count += 1
            montage = create_montage(resized_images)
            
            # Save the montage
            montage_filename = f"montage_{montage_count:03d}.jpg"
            montage_path = output_directory / montage_filename
            montage.save(montage_path, quality=90, optimize=True)
            
            print(f"✅ Created montage: {montage_filename}")
            
            # Reset for next group
            resized_images = []
    
    # Handle remaining images (less than 4)
    if resized_images:
        # Fill remaining slots with white images
        while len(resized_images) < 4:
            placeholder = Image.new('RGB', (1024, 1024), (255, 255, 255))
            resized_images.append(placeholder)
        
        montage_count += 1
        montage = create_montage(resized_images)
        
        montage_filename = f"montage_{montage_count:03d}_partial.jpg"
        montage_path = output_directory / montage_filename
        montage.save(montage_path, quality=90, optimize=True)
        
        print(f"✅ Created partial montage: {montage_filename}")
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Summary:")
    print(f"   - Total images processed: {len(image_files)}")
    print(f"   - Montages created: {montage_count}")
    print(f"   - Output location: {output_directory}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Create 2048x2048 montages from images")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("-o", "--output", help="Output directory for montages (default: input_dir/montages)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Process images
    process_images(args.input_dir, args.output)


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) == 1:
        # No command line arguments - use current directory
        print("No directory specified. Using current directory...")
        print("Usage: python image_montage_creator.py <input_directory> [-o output_directory]")
        print("\nProcessing current directory as example...")
        
        current_dir = Path.cwd()
        process_images(current_dir)
    else:
        main()
