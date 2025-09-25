"""
Retro Scaler - Intelligent Montage Creator and Scaler

This script creates optimal montages from smaller images before scaling to 720x1600 pixels.
Perfect for combining retro game screenshots, pixel art, or other small images into organized layouts.

Key Features:
- Select directory containing small images
- Intelligent montage creation: combines images of similar sizes
- Aspect ratio validation: only creates montages that won't exceed 720px width when scaled to 1600px height
- Optimal layout calculation (1x2, 1x3, 2x2, 2x3, etc.)
- Scales final montage to exactly 720x1600 pixels while preserving aspect ratios
- Fallback to single image processing when montages aren't feasible
- Output format selection: high-quality JPEG or PNG
- Generates unique timestamped filenames
"""

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import math
from datetime import datetime

class RetroScaler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Retro Scaler - Montage Creator (720x1600)")
        self.root.geometry("900x700")
        
        self.input_directory = None
        
        # Target dimensions
        self.target_width = 720
        self.target_height = 1600
        
        # Montage settings
        self.max_images_per_montage = 6  # Maximum images to combine
        self.min_images_per_montage = 2  # Minimum images to combine
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(main_frame, text="Retro Scaler - Intelligent Montage Creator (720x1600)", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Settings info frame
        info_frame = tk.LabelFrame(main_frame, text="Montage Settings", 
                                  font=("Arial", 10, "bold"))
        info_frame.pack(pady=(0, 20), fill='x')
        
        # Montage size setting
        montage_frame = tk.Frame(info_frame)
        montage_frame.pack(pady=10)
        
        tk.Label(montage_frame, text="Max images per montage:", 
                font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # Create montage size variable and dropdown
        self.montage_size_var = tk.StringVar(value="6")
        montage_values = ["2", "3", "4", "6", "8", "9"]
        montage_dropdown = tk.OptionMenu(montage_frame, self.montage_size_var, *montage_values, 
                                        command=self.on_montage_size_change)
        montage_dropdown.config(font=("Arial", 9))
        montage_dropdown.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Label(montage_frame, text="images", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Output format setting
        format_frame = tk.Frame(info_frame)
        format_frame.pack(pady=10)
        
        self.save_as_jpg = tk.BooleanVar(value=False)  # Default to PNG
        jpg_checkbox = tk.Checkbutton(format_frame, text="Save as high-quality JPEG (Q=1, otherwise PNG)", 
                                     variable=self.save_as_jpg, font=("Arial", 9, "bold"),
                                     command=self.on_format_change)
        jpg_checkbox.pack(side=tk.LEFT)
        
        # Settings info
        settings_text = "Creates aspect-ratio compliant montages from small images\nTarget: 720x1600 pixels (width limit enforced)"
        self.settings_label = tk.Label(info_frame, text=settings_text, 
                                      font=("Arial", 9), justify='center')
        self.settings_label.pack(pady=(5, 10))
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(0, 20))
        
        # Select directory button
        select_button = tk.Button(button_frame, text="Select Image Directory", 
                                 command=self.select_directory, bg="#4CAF50", fg="white",
                                 font=("Arial", 10), padx=20, pady=10)
        select_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        self.process_button = tk.Button(button_frame, text="Create Montages", 
                                       command=self.process_all_images, bg="#2196F3", fg="white",
                                       font=("Arial", 10), padx=20, pady=10, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT)
        
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
        
        # Status frame
        status_frame = tk.LabelFrame(main_frame, text="Status Log", 
                                    font=("Arial", 10, "bold"))
        status_frame.pack(expand=True, fill='both')
        
        # Status text with scrollbar
        status_text_frame = tk.Frame(status_frame)
        status_text_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(status_text_frame)
        self.status_text = tk.Text(status_text_frame, height=8, wrap=tk.WORD, 
                                  yscrollcommand=scrollbar.set, font=("Consolas", 9))
        scrollbar.config(command=self.status_text.yview)
        
        self.status_text.pack(side=tk.LEFT, expand=True, fill='both')
        scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Initial status
        self.log_message("Ready to create montages. Select a directory to begin.")
    
    def on_montage_size_change(self, value):
        """Handle montage size dropdown change"""
        self.max_images_per_montage = int(value)
        self.log_message(f"Max images per montage changed to {self.max_images_per_montage}")
    
    def on_format_change(self):
        """Handle output format checkbox change"""
        format_type = "high-quality JPEG (Q=1)" if self.save_as_jpg.get() else "PNG"
        self.log_message(f"Output format changed to {format_type}")
    
    def log_message(self, message):
        """Add message to status log"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def select_directory(self):
        """Select input directory containing images"""
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if directory:
            self.input_directory = directory
            self.progress_label.config(text=f"Selected: {os.path.normpath(directory)}")
            self.process_button.config(state=tk.NORMAL)
            self.log_message(f"Selected directory: {os.path.normpath(directory)}")
            
            # Analyze images in directory
            self.analyze_images()
    
    def analyze_images(self):
        """Analyze images in the selected directory"""
        if not self.input_directory:
            return
        
        image_files = []
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
        
        for filename in os.listdir(self.input_directory):
            if filename.lower().endswith(supported_formats):
                image_files.append(filename)
        
        if not image_files:
            self.log_message("No supported image files found!")
            return
        
        self.log_message(f"Found {len(image_files)} image files")
        
        # Analyze image sizes
        size_groups = {}
        for filename in image_files:
            try:
                filepath = os.path.join(self.input_directory, filename)
                with Image.open(filepath) as img:
                    size = img.size
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append(filename)
            except Exception as e:
                self.log_message(f"Error analyzing {filename}: {e}")
        
        self.log_message(f"Found {len(size_groups)} different image sizes:")
        for size, files in size_groups.items():
            self.log_message(f"  {size[0]}x{size[1]}: {len(files)} images")
    
    def get_optimal_grid(self, num_images):
        """Calculate optimal grid dimensions for montage"""
        if num_images <= 1:
            return (1, 1)
        elif num_images == 2:
            return (1, 2)  # Vertical stack
        elif num_images == 3:
            return (1, 3)  # Vertical stack
        elif num_images == 4:
            return (2, 2)  # Square
        elif num_images <= 6:
            return (2, 3)  # 2 columns, 3 rows
        elif num_images <= 8:
            return (2, 4)  # 2 columns, 4 rows
        else:
            return (3, 3)  # 3x3 grid
    
    def can_create_montage(self, img_width, img_height, num_images):
        """Check if montage is feasible without exceeding 720px when scaled to 1600px height"""
        if num_images <= 1:
            return True, (1, 1)
        
        # Try different grid configurations
        possible_grids = [
            (1, 2), (1, 3), (1, 4),  # Vertical stacks
            (2, 2), (2, 3), (2, 4),  # 2-column grids
            (3, 2), (3, 3)           # 3-column grids
        ]
        
        for cols, rows in possible_grids:
            if cols * rows < num_images:
                continue
            if cols * rows > num_images + 2:  # Don't waste too much space
                continue
                
            # Calculate montage dimensions
            montage_width = img_width * cols
            montage_height = img_height * rows
            
            # Check if montage width already exceeds 720
            if montage_width > 720:
                continue
            
            # Calculate what width would be when scaled to 1600px height
            aspect_ratio = montage_width / montage_height
            scaled_width_at_1600 = int(1600 * aspect_ratio)
            
            # Check if scaled width would exceed 720
            if scaled_width_at_1600 <= 720:
                return True, (cols, rows)
        
        # No valid montage configuration found
        return False, (1, 1)
    
    def create_montage(self, image_paths, output_path):
        """Create a montage from a list of image paths"""
        try:
            # Load all images
            images = []
            for path in image_paths:
                img = Image.open(path)
                images.append(img)
            
            if not images:
                return False
            
            # Get the size of the first image (assuming all are same size)
            img_width, img_height = images[0].size
            
            # Check if montage is feasible
            can_montage, (grid_cols, grid_rows) = self.can_create_montage(img_width, img_height, len(images))
            
            if not can_montage:
                self.log_message(f"Cannot create montage: {img_width}x{img_height} images would exceed 720px width when scaled")
                return False
            
            # Create montage dimensions
            montage_width = img_width * grid_cols
            montage_height = img_height * grid_rows
            
            # Create blank montage canvas
            montage = Image.new('RGB', (montage_width, montage_height), (0, 0, 0))
            
            # Place images in grid
            for i, img in enumerate(images):
                if i >= grid_cols * grid_rows:
                    break  # Don't exceed grid capacity
                
                col = i % grid_cols
                row = i // grid_cols
                
                x = col * img_width
                y = row * img_height
                
                montage.paste(img, (x, y))
            
            # Scale montage to target dimensions (720x1600)
            scaled_montage = montage.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
            
            # Save with format-specific options
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                scaled_montage.save(output_path, 'JPEG', quality=100, optimize=True)
            else:
                scaled_montage.save(output_path, 'PNG', optimize=True)
            
            # Close images to free memory
            for img in images:
                img.close()
            montage.close()
            scaled_montage.close()
            
            return True
            
        except Exception as e:
            self.log_message(f"Error creating montage: {e}")
            return False
    
    def process_single_image(self, image_path, output_path):
        """Process a single image by scaling it to 720x1600 with aspect ratio preservation"""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Scale to fit within 720x1600 while maintaining aspect ratio
            img.thumbnail((self.target_width, self.target_height), Image.Resampling.LANCZOS)
            
            # Create a 720x1600 canvas and center the image
            canvas = Image.new('RGB', (self.target_width, self.target_height), (0, 0, 0))
            
            # Calculate position to center the image
            x = (self.target_width - img.width) // 2
            y = (self.target_height - img.height) // 2
            
            canvas.paste(img, (x, y))
            
            # Save with format-specific options
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                canvas.save(output_path, 'JPEG', quality=100, optimize=True)
            else:
                canvas.save(output_path, 'PNG', optimize=True)
            
            img.close()
            canvas.close()
            
            return True
            
        except Exception as e:
            self.log_message(f"Error processing single image: {e}")
            return False
    
    def generate_timestamp_filename(self, output_dir, prefix, ext):
        """Generate a timestamp-based filename with conflict resolution"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        
        counter = 1
        while True:
            counter_str = f"{counter:03d}"
            filename = f"{timestamp}_{counter_str}_{prefix}_720x1600{ext}"
            output_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(output_path):
                return filename, output_path
            
            counter += 1
            if counter > 999:
                raise Exception("Too many files with the same timestamp")
    
    def process_all_images(self):
        """Process all images and create montages"""
        if not self.input_directory:
            messagebox.showerror("Error", "Please select a directory first")
            return
        
        try:
            # Create output directory
            output_dir = os.path.join(self.input_directory, "montages_720x1600")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all image files
            image_files = []
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
            
            for filename in os.listdir(self.input_directory):
                if filename.lower().endswith(supported_formats):
                    image_files.append(filename)
            
            if not image_files:
                messagebox.showerror("Error", "No supported image files found")
                return
            
            # Group images by size
            size_groups = {}
            for filename in image_files:
                filepath = os.path.join(self.input_directory, filename)
                try:
                    with Image.open(filepath) as img:
                        size = img.size
                        if size not in size_groups:
                            size_groups[size] = []
                        size_groups[size].append(filepath)
                except Exception as e:
                    self.log_message(f"Error processing {filename}: {e}")
            
            # Create montages for each size group
            total_montages = 0
            successful = 0
            failed = 0
            
            for size, file_paths in size_groups.items():
                img_width, img_height = size
                
                if len(file_paths) < self.min_images_per_montage:
                    self.log_message(f"Skipping {img_width}x{img_height} group: only {len(file_paths)} images (need at least {self.min_images_per_montage})")
                    continue
                
                # Check if any montage is possible for this image size
                can_montage, _ = self.can_create_montage(img_width, img_height, self.min_images_per_montage)
                if not can_montage:
                    self.log_message(f"Skipping {img_width}x{img_height} group: would exceed 720px width when scaled to 1600px height")
                    continue
                
                # Create montages from this size group
                group_name = f"{img_width}x{img_height}"
                images_processed = 0
                
                while images_processed < len(file_paths):
                    # Determine how many images to use for this montage
                    remaining_images = len(file_paths) - images_processed
                    
                    # Find the optimal number of images for this montage
                    best_count = 1
                    for test_count in range(self.min_images_per_montage, min(self.max_images_per_montage, remaining_images) + 1):
                        can_montage, _ = self.can_create_montage(img_width, img_height, test_count)
                        if can_montage:
                            best_count = test_count
                        else:
                            break  # Stop at first invalid configuration
                    
                    # Take images for this montage
                    end_idx = images_processed + best_count
                    montage_images = file_paths[images_processed:end_idx]
                    
                    # Determine output format
                    if self.save_as_jpg.get():
                        ext = ".jpg"
                    else:
                        ext = ".png"
                    
                    # Generate output filename
                    if len(montage_images) > 1:
                        montage_prefix = f"{group_name}_montage_{len(montage_images)}imgs"
                    else:
                        montage_prefix = f"{group_name}_single"
                    output_filename, output_path = self.generate_timestamp_filename(output_dir, montage_prefix, ext)
                    
                    # Update progress
                    total_montages += 1
                    progress = (total_montages / max(len(size_groups), 1)) * 100
                    self.progress_var.set(min(progress, 100))
                    
                    if len(montage_images) > 1:
                        self.progress_label.config(text=f"Creating montage: {output_filename}")
                        # Create montage
                        if self.create_montage(montage_images, output_path):
                            successful += 1
                            image_names = [os.path.basename(p) for p in montage_images]
                            self.log_message(f"✓ Created montage: {output_filename} from {len(montage_images)} images")
                            self.log_message(f"  Source images: {', '.join(image_names[:3])}{'...' if len(image_names) > 3 else ''}")
                        else:
                            failed += 1
                            self.log_message(f"✗ Failed to create montage from {group_name} group")
                    else:
                        # Process single image
                        self.progress_label.config(text=f"Processing single: {output_filename}")
                        if self.process_single_image(montage_images[0], output_path):
                            successful += 1
                            self.log_message(f"✓ Processed single image: {output_filename}")
                        else:
                            failed += 1
                            self.log_message(f"✗ Failed to process single image")
                    
                    images_processed = end_idx
            
            # Final results
            self.progress_var.set(100)
            self.progress_label.config(text="Processing complete!")
            
            if successful > 0:
                self.log_message(f"\n✓ Successfully created {successful} montages")
                self.log_message(f"Output directory: {os.path.normpath(output_dir)}")
                messagebox.showinfo("Success", f"Created {successful} montages!\nSaved in: {os.path.normpath(output_dir)}")
            
            if failed > 0:
                self.log_message(f"✗ Failed to create {failed} montages")
            
        except Exception as e:
            self.log_message(f"Error during processing: {e}")
            messagebox.showerror("Error", f"Processing failed: {e}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = RetroScaler()
    app.run()