#!/home/rich/MyCoding/image_processors/.venv/bin/python3
"""
Image Quarteriser - Split images into four equal quarters
Uses tkinter for directory selection only, with terminal-based progress
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
