#!/home/rich/MyCoding/image_processors/.venv/bin/python3
"""
FUNCTIONALITY SUMMARY:
======================

Primary Function - Image Splitting:
- Takes any image and splits it into 4 equal quadrants
- Creates quarters labeled: _TL (top-left), _TR (top-right), _BL (bottom-left), _BR (bottom-right)
- Adjusts dims to even numbers if necessary
- Saves quarters to a 'quarters' subdirectory
- Example: 'photo.jpg' becomes: photo_TL.jpg, photo_TR.jpg, photo_BL.jpg, photo_BR.jpg

Secondary Function - Rescale & Join (Optional):
- Rescales all quarter images by 0.9 (90% of original size)
- Groups quarters into sets of 4 and joins horizontally
- Creates 2560x1440 widescreen images (640px per quarter)
- Saves to 'joined_2560x1440' subdirectory
- Files named: joined_group_001.jpg, joined_group_002.jpg, etc.

"""

import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import sys

class ImageQuarteriser:
    def __init__(self):
        # Create a hidden root window for the file dialog only
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        
        # Variables
        self.total_images = 0
        self.processed_images = 0
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
        
    def select_directory(self):
        """Open directory selection dialog and return selected directory"""
        print("Opening directory selection dialog...")
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if directory:
            print(f"Selected directory: {directory}")
            return directory
        else:
            print("No directory selected.")
            return None
    
    def scan_directory(self, directory):
        """Scan selected directory for images and return list of image files"""
        if not directory or not os.path.exists(directory):
            return []
            
        image_files = []
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                image_files.append(filename)
        
        return image_files
    
    def print_progress_bar(self, current, total, bar_length=50):
        """Print a terminal progress bar"""
        percent = float(current) / total
        arrow = '█' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(arrow))
        
        sys.stdout.write(f'\r[{arrow}{spaces}] {current}/{total} ({percent:.1%})')
        sys.stdout.flush()
    
    def rescale_and_join_images(self, quarters_dir):
        """Rescale images by 0.9 and join four horizontally to create 2560x1440 images"""
        print("\n=== Rescaling and Joining Images ===")
        
        # Get all images from quarters directory
        image_files = self.scan_directory(quarters_dir)
        if not image_files:
            print("No images found in quarters directory.")
            return
        
        # Create output directory for joined images
        joined_dir = os.path.join(os.path.dirname(quarters_dir), "joined_2560x1440")
        os.makedirs(joined_dir, exist_ok=True)
        print(f"Created joined images directory: {joined_dir}")
        
        # Process images in groups of 4
        num_groups = (len(image_files) + 3) // 4  # Ceiling division
        print(f"Processing {len(image_files)} images in {num_groups} groups of 4...")
        
        for group_idx in range(num_groups):
            try:
                start_idx = group_idx * 4
                end_idx = min(start_idx + 4, len(image_files))
                group_files = image_files[start_idx:end_idx]
                
                print(f"\nProcessing group {group_idx + 1}/{num_groups} ({len(group_files)} images)...")
                
                # Load and rescale images
                rescaled_images = []
                for img_file in group_files:
                    img_path = os.path.join(quarters_dir, img_file)
                    with Image.open(img_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Rescale by 0.9
                        original_size = img.size
                        new_width = int(original_size[0] * 0.9)
                        new_height = int(original_size[1] * 0.9)
                        rescaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        rescaled_images.append(rescaled_img.copy())
                        print(f"  Rescaled {img_file}: {original_size} -> {new_width}x{new_height}")
                
                # If we have fewer than 4 images, create blank ones to fill
                target_height = 1440
                single_width = 2560 // 4  # 640 pixels per image
                
                while len(rescaled_images) < 4:
                    # Create a black image to fill empty slots
                    blank_img = Image.new('RGB', (single_width, target_height), (0, 0, 0))
                    rescaled_images.append(blank_img)
                    print(f"  Added blank image to fill group")
                
                # Resize all images to fit exactly in 2560x1440 when joined
                final_images = []
                for img in rescaled_images:
                    # Resize to fit in the 640x1440 slot
                    resized_img = img.resize((single_width, target_height), Image.Resampling.LANCZOS)
                    final_images.append(resized_img)
                
                # Join four images horizontally
                joined_img = Image.new('RGB', (2560, 1440))
                x_offset = 0
                for img in final_images:
                    joined_img.paste(img, (x_offset, 0))
                    x_offset += single_width
                
                # Save the joined image
                output_filename = f"joined_group_{group_idx + 1:03d}.jpg"
                output_path = os.path.join(joined_dir, output_filename)
                joined_img.save(output_path, quality=95)
                
                print(f"  ✓ Created {output_filename} (2560x1440)")
                
            except Exception as e:
                print(f"  ✗ Error processing group {group_idx + 1}: {str(e)}")
        
        print(f"\n✓ Rescaling and joining completed!")
        print(f"  Created {num_groups} joined images (2560x1440)")
        print(f"  Output directory: {joined_dir}")

    def process_images(self, directory):
        """Process all images in the selected directory"""
        quarters_dir = os.path.join(directory, "quarters")
        
        try:
            # Create quarters subdirectory
            os.makedirs(quarters_dir, exist_ok=True)
            print(f"Created quarters directory: {quarters_dir}")
            
            # Get list of image files
            image_files = self.scan_directory(directory)
            
            if not image_files:
                print("No supported image files found in the selected directory.")
                return
            
            self.total_images = len(image_files)
            self.processed_images = 0
            
            print(f"Found {self.total_images} image files to process...")
            print()
            
            # Process each image
            for i, filename in enumerate(image_files):
                try:
                    print(f"Processing {i+1}/{self.total_images}: {filename}")
                    
                    # Load image
                    input_path = os.path.join(directory, filename)
                    with Image.open(input_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        original_width, original_height = img.size
                        
                        # Ensure dimensions are divisible by 2
                        width = original_width if original_width % 2 == 0 else original_width - 1
                        height = original_height if original_height % 2 == 0 else original_height - 1
                        
                        if width != original_width or height != original_height:
                            img = img.resize((width, height), Image.Resampling.LANCZOS)
                            print(f"  Resized from {original_width}x{original_height} to {width}x{height}")
                        
                        # Calculate quarter dimensions
                        quarter_width = width // 2
                        quarter_height = height // 2
                        
                        print(f"  Splitting into quarters ({quarter_width}x{quarter_height} each)...")
                        
                        # Extract quarters
                        quarters = {
                            '_TL': img.crop((0, 0, quarter_width, quarter_height)),  # Top-left
                            '_TR': img.crop((quarter_width, 0, width, quarter_height)),  # Top-right
                            '_BL': img.crop((0, quarter_height, quarter_width, height)),  # Bottom-left
                            '_BR': img.crop((quarter_width, quarter_height, width, height))  # Bottom-right
                        }
                        
                        # Save quarters
                        base_name = os.path.splitext(filename)[0]
                        extension = os.path.splitext(filename)[1]
                        
                        saved_count = 0
                        for suffix, quarter_img in quarters.items():
                            quarter_filename = f"{base_name}{suffix}{extension}"
                            quarter_path = os.path.join(quarters_dir, quarter_filename)
                            quarter_img.save(quarter_path, quality=95 if extension.lower() in ['.jpg', '.jpeg'] else None)
                            saved_count += 1
                        
                        if saved_count == 4:
                            print(f"  ✓ Successfully split into 4 quarters")
                        else:
                            print(f"  ⚠ Partially processed ({saved_count}/4 quarters saved)")
                
                except Exception as e:
                    print(f"  ✗ Error processing {filename}: {str(e)}")
                
                # Update progress and show progress bar
                self.processed_images += 1
                self.print_progress_bar(self.processed_images, self.total_images)
            
            # Final status
            print("\n")  # New line after progress bar
            print(f"✓ Processing completed successfully!")
            print(f"  Total images processed: {self.processed_images}")
            print(f"  Total quarters created: {self.processed_images * 4}")
            print(f"  Output directory: {quarters_dir}")
            
            # Ask user if they want to rescale and join images
            print("\n" + "="*50)
            while True:
                choice = input("Do you want to rescale images by 0.9 and join them into 2560x1440 images? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    self.rescale_and_join_images(quarters_dir)
                    break
                elif choice in ['n', 'no']:
                    print("Skipping rescale and join operation.")
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
        
        except Exception as e:
            print(f"✗ Fatal error: {str(e)}")
    
    def run(self):
        """Start the application"""
        # Select directory
        directory = self.select_directory()
        
        if not directory:
            print("Exiting - no directory selected.")
            return
        
        # Process the images
        self.process_images(directory)
        
        # Clean up the hidden tkinter window
        self.root.destroy()

def main():
    """Main function"""
    print("=== Image Quarteriser ===")
    print("This tool splits images into four equal quarters.")
    print()
    
    app = ImageQuarteriser()
    app.run()

if __name__ == "__main__":
    main()
