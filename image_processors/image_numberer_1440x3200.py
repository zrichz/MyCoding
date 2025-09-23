"""
Image Numberer - Batch Processing GUI Application

This script provides a GUI for batch processing 1440x3200 pixel images by adding
unique padded numbers to the four corners of each image.

Key Features:
- Select directory containing 1440x3200 pixel images
- Adds sequential numbered labels to each corner (0001, 0002, 0003, 0004 for first image, etc.)
- Small black text with white border for visibility
- Automatically saves as highest quality JPEG (Q=1/100%)
- Generates unique timestamped filenames with conflict resolution
"""

from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from datetime import datetime

class ImageNumberer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Numberer - Batch Processor (1440x3200)")
        self.root.geometry("900x600")
        
        self.input_directory = None
        
        # Target dimensions for validation
        self.target_width = 1440
        self.target_height = 3200
        
        # Counter for numbering across all images
        self.global_counter = 1
        
        # Text settings
        self.font_size = 24
        self.text_color = (0, 0, 0)  # Black
        self.border_color = (255, 255, 255)  # White
        self.border_width = 2
        self.margin = 20  # Distance from corner
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(main_frame, text="Image Numberer - Batch Processor (1440x3200)", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Settings info frame
        info_frame = tk.LabelFrame(main_frame, text="Processing Settings", 
                                  font=("Arial", 10, "bold"))
        info_frame.pack(pady=(0, 20), fill='x')
        
        # Settings info
        settings_text = ("Target: 1440x3200 pixels only\n"
                        "Numbers: 4 per image (corners), padded format (0001, 0002, etc.)\n"
                        "Text: Black with white border, saved as highest quality JPEG")
        settings_label = tk.Label(info_frame, text=settings_text, 
                                 font=("Arial", 9), justify='center')
        settings_label.pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(0, 20))
        
        # Select directory button
        select_button = tk.Button(button_frame, text="Select Image Directory", 
                                 command=self.select_directory, bg="#4CAF50", fg="white",
                                 font=("Arial", 10), padx=20, pady=10)
        select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        self.process_button = tk.Button(button_frame, text="Process All Images", 
                                       command=self.process_all_images, bg="#2196F3", fg="white",
                                       font=("Arial", 10), padx=20, pady=10, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT)
        
        # Reset counter button
        reset_button = tk.Button(button_frame, text="Reset Counter", 
                                command=self.reset_counter, bg="#FF9800", fg="white",
                                font=("Arial", 10), padx=20, pady=10)
        reset_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress frame
        progress_frame = tk.LabelFrame(main_frame, text="Progress", 
                                      font=("Arial", 10, "bold"))
        progress_frame.pack(pady=(0, 20), fill='x')
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(pady=10)
        
        # Progress label
        self.progress_label = tk.Label(progress_frame, text="No directory selected", 
                                      font=("Arial", 9))
        self.progress_label.pack()
        
        # Counter display
        self.counter_label = tk.Label(progress_frame, text=f"Next number: {self.global_counter:04d}", 
                                     font=("Arial", 9, "bold"), fg="blue")
        self.counter_label.pack(pady=(5, 0))
        
        # Status frame
        status_frame = tk.Frame(main_frame)
        status_frame.pack(expand=True, fill='both')
        
        # Status text area
        self.status_text = tk.Text(status_frame, height=15, wrap=tk.WORD,
                                  font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, expand=True, fill='both')
        scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Initial status
        self.log_message("Ready to process 1440x3200 images. Select a directory to begin.")
    
    def log_message(self, message):
        """Add a message to the status text area"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def reset_counter(self):
        """Reset the global counter to 1"""
        self.global_counter = 1
        self.counter_label.config(text=f"Next number: {self.global_counter:04d}")
        self.log_message("Counter reset to 0001")
    
    def select_directory(self):
        """Select input directory"""
        directory = filedialog.askdirectory(title="Select Directory with 1440x3200 Images")
        if directory:
            self.input_directory = directory
            self.process_button.config(state=tk.NORMAL)
            self.progress_label.config(text=f"Directory selected: {os.path.basename(directory)}")
            self.log_message(f"Selected directory: {directory}")
            
            # Count valid images
            image_files = self.get_valid_image_files(directory)
            self.log_message(f"Found {len(image_files)} valid image files")
    
    def get_valid_image_files(self, directory):
        """Get list of valid image files that are exactly 1440x3200"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                try:
                    # Check image dimensions
                    filepath = os.path.join(directory, filename)
                    with Image.open(filepath) as img:
                        if img.size == (self.target_width, self.target_height):
                            image_files.append(filename)
                        else:
                            self.log_message(f"Skipped {filename}: {img.size[0]}x{img.size[1]} (not 1440x3200)")
                except Exception as e:
                    self.log_message(f"Error checking {filename}: {str(e)}")
        
        return sorted(image_files)
    
    def get_font(self):
        """Get the font for text rendering"""
        try:
            # Try to use a system font
            return ImageFont.truetype("arial.ttf", self.font_size)
        except:
            try:
                return ImageFont.truetype("Arial.ttf", self.font_size)
            except:
                try:
                    # Fallback to default font
                    return ImageFont.load_default()
                except:
                    # Ultimate fallback
                    return None
    
    def draw_text_with_border(self, draw, position, text, font):
        """Draw text with white border"""
        x, y = position
        
        # Draw border by drawing text in multiple positions
        for dx in range(-self.border_width, self.border_width + 1):
            for dy in range(-self.border_width, self.border_width + 1):
                if dx != 0 or dy != 0:  # Don't draw on the center position yet
                    draw.text((x + dx, y + dy), text, font=font, fill=self.border_color)
        
        # Draw the main text on top
        draw.text(position, text, font=font, fill=self.text_color)
    
    def process_all_images(self):
        """Process all images in the selected directory"""
        if not self.input_directory:
            return
        
        try:
            # Disable button during processing
            self.process_button.config(state=tk.DISABLED)
            
            # Get valid image files
            image_files = self.get_valid_image_files(self.input_directory)
            total_files = len(image_files)
            
            if total_files == 0:
                self.log_message("No valid 1440x3200 image files to process")
                return
            
            # Create output directory
            output_dir = os.path.join(self.input_directory, "numbered")
            os.makedirs(output_dir, exist_ok=True)
            self.log_message(f"Created output directory: {output_dir}")
            
            # Process each image
            successful = 0
            failed = 0
            
            for i, filename in enumerate(image_files):
                try:
                    # Update progress
                    progress = (i / total_files) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(text=f"Processing {i+1}/{total_files}: {filename}")
                    
                    # Process the image
                    input_path = os.path.join(self.input_directory, filename)
                    
                    # Generate output filename
                    output_filename, output_path = self.generate_timestamp_filename(output_dir)
                    
                    # Get the four numbers for this image
                    numbers = [f"{self.global_counter + j:04d}" for j in range(4)]
                    
                    if self.process_single_image(input_path, output_path, numbers):
                        successful += 1
                        self.log_message(f"✓ Processed: {filename} -> {output_filename} (numbers: {', '.join(numbers)})")
                        # Increment counter by 4 for next image
                        self.global_counter += 4
                        self.counter_label.config(text=f"Next number: {self.global_counter:04d}")
                    else:
                        failed += 1
                        self.log_message(f"✗ Failed: {filename}")
                        
                except Exception as e:
                    failed += 1
                    self.log_message(f"✗ Error processing {filename}: {str(e)}")
            
            # Complete
            self.progress_var.set(100)
            self.progress_label.config(text=f"Complete: {successful} processed, {failed} failed")
            self.log_message(f"\nBatch processing complete!")
            self.log_message(f"Successfully processed: {successful} images")
            self.log_message(f"Failed: {failed} images")
            self.log_message(f"Output directory: {output_dir}")
            self.log_message(f"Next counter value: {self.global_counter:04d}")
            
            if successful > 0:
                messagebox.showinfo("Processing Complete", 
                                   f"Successfully processed {successful} images!\n"
                                   f"Output saved to: {output_dir}\n"
                                   f"Next counter: {self.global_counter:04d}")
            
        except Exception as e:
            self.log_message(f"Error during batch processing: {str(e)}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
        
        finally:
            # Re-enable button
            self.process_button.config(state=tk.NORMAL)
    
    def generate_timestamp_filename(self, output_dir):
        """Generate a timestamp-based filename with conflict resolution"""
        # Get current timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        
        # Start with counter 001
        counter = 1
        
        while True:
            # Format counter as 3-digit number
            counter_str = f"{counter:03d}"
            filename = f"{timestamp}_{counter_str}_numbered_1440x3200.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Check if file exists
            if not os.path.exists(output_path):
                return filename, output_path
            
            # Increment counter and try again
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 999:
                raise Exception("Too many files with the same timestamp - cannot generate unique filename")
    
    def process_single_image(self, input_path, output_path, numbers):
        """Process a single image file by adding corner numbers"""
        try:
            # Load image
            image = Image.open(input_path)
            
            # Verify dimensions
            if image.size != (self.target_width, self.target_height):
                self.log_message(f"Skipping {input_path}: wrong dimensions {image.size}")
                return False
            
            # Convert to RGB if necessary (for JPEG saving)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create draw object
            draw = ImageDraw.Draw(image)
            
            # Get font
            font = self.get_font()
            
            # Define corner positions for 1440x3200 image
            # Top-left: numbers[0]
            # Top-right: numbers[1] 
            # Bottom-left: numbers[2]
            # Bottom-right: numbers[3]
            
            positions = [
                (self.margin, self.margin),  # Top-left
                (self.target_width - 100, self.margin),  # Top-right (approximate, will adjust)
                (self.margin, self.target_height - 50),  # Bottom-left (approximate, will adjust)
                (self.target_width - 100, self.target_height - 50)  # Bottom-right (approximate, will adjust)
            ]
            
            # Adjust positions based on text size if font is available
            if font:
                for i, (number, pos) in enumerate(zip(numbers, positions)):
                    # Get text bounding box
                    bbox = draw.textbbox((0, 0), number, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    x, y = pos
                    
                    # Adjust position based on corner
                    if i == 1:  # Top-right
                        x = self.target_width - text_width - self.margin
                    elif i == 2:  # Bottom-left
                        y = self.target_height - text_height - self.margin
                    elif i == 3:  # Bottom-right
                        x = self.target_width - text_width - self.margin
                        y = self.target_height - text_height - self.margin
                    
                    positions[i] = (int(x), int(y))
            
            # Draw numbers in corners
            for number, position in zip(numbers, positions):
                self.draw_text_with_border(draw, position, number, font)
            
            # Save as highest quality JPEG (quality=100, Q=1 equivalent)
            image.save(output_path, 'JPEG', quality=100, optimize=True)
            
            return True
            
        except Exception as e:
            self.log_message(f"Error processing {input_path}: {e}")
            return False
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageNumberer()
    app.run()