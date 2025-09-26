"""
Retro Scaler - Intelligent Montage Creator and Scaler

This script creates optimal montages from smaller images before scaling to 720x1600 pixels.
Perfect for combining retro game screenshots, pixel art, or other small images into organized layouts.

Key Features:
- Select directory containing small images
- Intelligent montage creation: combines images of similar sizes
- Aspect ratio validation: only creates montages that won't exceed 720px width when scaled to 1600px height
- Optimal layout calculation (1x2, 1x3, 2x2, 2x3, etc.)
- Intelligent expansion: uses blur and fade effects instead of black bars
- Fixed 160px maximum blur applied to expanded regions
- Fixed 50% luminance reduction (darkening) for natural fade effect
- Dual-axis expansion: applies effects to both horizontal and vertical expansions
- Preserves aspect ratios during montage creation and scaling
- Fallback to single image processing when montages aren't feasible
- Output format selection: high-quality JPEG or PNG
- Generates unique timestamped filenames
"""

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy import ndimage
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
        
        # Blur and fade settings (same as image_expander_720x1600.py)
        self.blur_amount = 160
        self.luminance_drop = 50
        
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
        settings_text = f"Creates aspect-ratio compliant montages with intelligent expansion\nTarget: 720x1600 pixels • Blur: {self.blur_amount}px • Fade: {self.luminance_drop}%"
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
    
    def apply_luminance_reduction(self, image_array, reduction_percent):
        """Apply luminance reduction (darkening) to an image array"""
        if reduction_percent <= 0:
            return image_array
        
        # Calculate reduction factor (0-100% becomes 0.0-1.0)
        reduction_factor = reduction_percent / 100.0
        
        # Apply reduction (darken)
        darkened = image_array.astype(np.float32) * (1.0 - reduction_factor)
        return np.clip(darkened, 0, 255).astype(np.uint8)
    
    def apply_horizontal_blur(self, line_array, blur_amount):
        """Apply horizontal-only blur to preserve colors"""
        if blur_amount <= 0:
            return line_array
        
        # Create horizontal Gaussian kernel
        kernel_size = int(blur_amount * 2) * 2 + 1  # Ensure odd size
        sigma = blur_amount / 3.0  # Convert blur_amount to sigma
        
        # Create 1D horizontal Gaussian kernel
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Apply horizontal convolution to each color channel
        if len(line_array.shape) == 2:  # Color image (width, channels)
            blurred = np.zeros_like(line_array, dtype=np.float32)
            for channel in range(line_array.shape[1]):
                # Use mode='nearest' to handle edges properly
                blurred[:, channel] = ndimage.convolve1d(
                    line_array[:, channel].astype(np.float32), 
                    kernel, 
                    axis=0, 
                    mode='nearest'
                )
            return np.clip(blurred, 0, 255).astype(np.uint8)
        else:  # Grayscale
            blurred = ndimage.convolve1d(
                line_array.astype(np.float32), 
                kernel, 
                axis=0, 
                mode='nearest'
            )
            return np.clip(blurred, 0, 255).astype(np.uint8)
    
    def apply_vertical_blur(self, column_array, blur_amount):
        """Apply vertical-only blur to preserve colors"""
        if blur_amount <= 0:
            return column_array
        
        # Create vertical Gaussian kernel
        kernel_size = int(blur_amount * 2) * 2 + 1  # Ensure odd size
        sigma = blur_amount / 3.0  # Convert blur_amount to sigma
        
        # Create 1D vertical Gaussian kernel
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Apply vertical convolution to each color channel
        if len(column_array.shape) == 2:  # Color image (height, channels)
            blurred = np.zeros_like(column_array, dtype=np.float32)
            for channel in range(column_array.shape[1]):
                # Use mode='nearest' to handle edges properly
                blurred[:, channel] = ndimage.convolve1d(
                    column_array[:, channel].astype(np.float32), 
                    kernel, 
                    axis=0, 
                    mode='nearest'
                )
            return np.clip(blurred, 0, 255).astype(np.uint8)
        else:  # Grayscale
            blurred = ndimage.convolve1d(
                column_array.astype(np.float32), 
                kernel, 
                axis=0, 
                mode='nearest'
            )
            return np.clip(blurred, 0, 255).astype(np.uint8)
    
    def validate_images_for_montage(self, image_paths):
        """Validate that images can be processed for montage creation"""
        try:
            valid_images = []
            invalid_count = 0
            
            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        # Check for reasonable dimensions
                        width, height = img.size
                        if width <= 0 or height <= 0:
                            self.log_message(f"Invalid dimensions for {os.path.basename(path)}: {width}x{height}")
                            invalid_count += 1
                            continue
                        
                        # Check for extremely large images that might cause memory issues
                        if width > 10000 or height > 10000:
                            self.log_message(f"Warning: Very large image {os.path.basename(path)}: {width}x{height}")
                        
                        # Check that image can be loaded properly
                        img.load()  # Force image loading
                        valid_images.append(path)
                        
                except Exception as e:
                    self.log_message(f"Cannot load image {os.path.basename(path)}: {e}")
                    invalid_count += 1
            
            if invalid_count > 0:
                self.log_message(f"Found {invalid_count} invalid images out of {len(image_paths)}")
            
            return valid_images
            
        except Exception as e:
            self.log_message(f"Error during image validation: {e}")
            return []

    def create_montage(self, image_paths, output_path):
        """Create a montage from a list of image paths (supports mixed sizes)"""
        try:
            # Validate images first
            valid_paths = self.validate_images_for_montage(image_paths)
            if not valid_paths:
                self.log_message("No valid images found for montage creation")
                return False
                
            if len(valid_paths) < len(image_paths):
                self.log_message(f"Using {len(valid_paths)} valid images out of {len(image_paths)} total")
            
            # Load all valid images and get their sizes
            images = []
            sizes = []
            for path in valid_paths:
                img = Image.open(path)
                images.append(img)
                sizes.append(img.size)
            
            if not images:
                return False
            
            # Determine target cell size for the montage
            # Use the median size as target to balance between too small and too large
            widths = [size[0] for size in sizes]
            heights = [size[1] for size in sizes]
            widths.sort()
            heights.sort()
            
            # Use median dimensions as target cell size
            target_cell_width = widths[len(widths) // 2]
            target_cell_height = heights[len(heights) // 2]
            
            self.log_message(f"Using target cell size: {target_cell_width}x{target_cell_height} for montage")
            
            # Check if montage is feasible with target cell size
            can_montage, (grid_cols, grid_rows) = self.can_create_montage(target_cell_width, target_cell_height, len(images))
            
            if not can_montage:
                # Try with smallest dimensions as a fallback
                min_width = min(widths)
                min_height = min(heights)
                self.log_message(f"Median size failed, trying minimum size: {min_width}x{min_height}")
                
                can_montage, (grid_cols, grid_rows) = self.can_create_montage(min_width, min_height, len(images))
                
                if can_montage:
                    target_cell_width = min_width
                    target_cell_height = min_height
                    self.log_message(f"Using minimum size approach: {target_cell_width}x{target_cell_height}")
                else:
                    self.log_message(f"Cannot create montage: even {min_width}x{min_height} cells would exceed 720px width when scaled")
                    return False
            
            # Normalize all images to target cell size while preserving aspect ratio
            normalized_images = []
            for i, img in enumerate(images):
                try:
                    # Create a black background of target cell size
                    cell_bg = Image.new('RGB', (target_cell_width, target_cell_height), (0, 0, 0))
                    
                    # Scale image to fit within cell while maintaining aspect ratio
                    img_copy = img.copy()
                    img_copy.thumbnail((target_cell_width, target_cell_height), Image.Resampling.LANCZOS)
                    
                    # Center the scaled image in the cell
                    x_offset = (target_cell_width - img_copy.width) // 2
                    y_offset = (target_cell_height - img_copy.height) // 2
                    cell_bg.paste(img_copy, (x_offset, y_offset))
                    
                    normalized_images.append(cell_bg)
                    img_copy.close()
                    
                except Exception as e:
                    self.log_message(f"Error normalizing image {i+1}: {e}")
                    # Clean up and return failure
                    for img in images:
                        img.close()
                    for norm_img in normalized_images:
                        norm_img.close()
                    return False
            
            # Create montage dimensions using normalized cell size
            montage_width = target_cell_width * grid_cols
            montage_height = target_cell_height * grid_rows
            
            # Create blank montage canvas
            montage = Image.new('RGB', (montage_width, montage_height), (0, 0, 0))
            
            # Place normalized images in grid
            for i, normalized_img in enumerate(normalized_images):
                if i >= grid_cols * grid_rows:
                    break  # Don't exceed grid capacity
                
                col = i % grid_cols
                row = i // grid_cols
                
                x = col * target_cell_width
                y = row * target_cell_height
                
                montage.paste(normalized_img, (x, y))
            
            # Scale montage to fit within 720x1600 while maintaining aspect ratio
            montage.thumbnail((self.target_width, self.target_height), Image.Resampling.LANCZOS)
            
            # Convert montage to numpy array for intelligent expansion
            montage_array = np.array(montage)
            orig_height, orig_width = montage_array.shape[:2]
            
            # Use fixed settings for blur and fade
            max_blur = self.blur_amount
            max_luminance_drop = self.luminance_drop
            
            # Calculate expansion needed
            width_expansion = max(0, self.target_width - orig_width)
            height_expansion = max(0, self.target_height - orig_height)
            
            # Calculate padding for each side
            left_pad = width_expansion // 2
            right_pad = width_expansion - left_pad
            top_pad = height_expansion // 2
            bottom_pad = height_expansion - top_pad
            
            # Create expanded image array
            if len(montage_array.shape) == 3:  # Color image
                expanded_array = np.zeros((self.target_height, self.target_width, montage_array.shape[2]), dtype=montage_array.dtype)
            else:  # Grayscale
                expanded_array = np.zeros((self.target_height, self.target_width), dtype=montage_array.dtype)
            
            # Copy original montage to center
            y_start = top_pad
            y_end = y_start + orig_height
            x_start = left_pad
            x_end = x_start + orig_width
            expanded_array[y_start:y_end, x_start:x_end] = montage_array
            
            # Process vertical expansion (top)
            if top_pad > 0:
                top_line = montage_array[0]  # First line of montage
                for i in range(top_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (top_pad - i) / top_pad if top_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply horizontal blur and luminance reduction
                    blurred_line = self.apply_horizontal_blur(top_line, blur_amount)
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                    expanded_array[i, x_start:x_end] = final_line
            
            # Process vertical expansion (bottom)
            if bottom_pad > 0:
                bottom_line = montage_array[-1]  # Last line of montage
                for i in range(bottom_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (i + 1) / bottom_pad if bottom_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply horizontal blur and luminance reduction
                    blurred_line = self.apply_horizontal_blur(bottom_line, blur_amount)
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                    expanded_array[y_end + i, x_start:x_end] = final_line
            
            # Process horizontal expansion (left)
            if left_pad > 0:
                left_column = expanded_array[:, x_start]  # First column of current image
                for i in range(left_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (left_pad - i) / left_pad if left_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply vertical blur and luminance reduction
                    blurred_column = self.apply_vertical_blur(left_column, blur_amount)
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reduction)
                    expanded_array[:, i] = final_column
            
            # Process horizontal expansion (right)
            if right_pad > 0:
                right_column = expanded_array[:, x_end - 1]  # Last column of current image
                for i in range(right_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (i + 1) / right_pad if right_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply vertical blur and luminance reduction
                    blurred_column = self.apply_vertical_blur(right_column, blur_amount)
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reduction)
                    expanded_array[:, x_end + i] = final_column
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(expanded_array.astype('uint8'))
            
            # Save with format-specific options
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                processed_image.save(output_path, 'JPEG', quality=100, optimize=True)
            else:
                processed_image.save(output_path, 'PNG', optimize=True)
            
            # Close images to free memory
            for img in images:
                img.close()
            for normalized_img in normalized_images:
                normalized_img.close()
            montage.close()
            processed_image.close()
            
            return True
            
        except Exception as e:
            self.log_message(f"Error creating montage: {e}")
            return False
    
    def process_single_image(self, image_path, output_path):
        """Process a single image using intelligent expansion with blur and fade"""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Scale to fit within 720x1600 while maintaining aspect ratio
            img.thumbnail((self.target_width, self.target_height), Image.Resampling.LANCZOS)
            
            # Convert image to numpy array for intelligent expansion
            img_array = np.array(img)
            orig_height, orig_width = img_array.shape[:2]
            
            # Use fixed settings for blur and fade
            max_blur = self.blur_amount
            max_luminance_drop = self.luminance_drop
            
            # Calculate expansion needed
            width_expansion = max(0, self.target_width - orig_width)
            height_expansion = max(0, self.target_height - orig_height)
            
            # Calculate padding for each side
            left_pad = width_expansion // 2
            right_pad = width_expansion - left_pad
            top_pad = height_expansion // 2
            bottom_pad = height_expansion - top_pad
            
            # Create expanded image array
            if len(img_array.shape) == 3:  # Color image
                expanded_array = np.zeros((self.target_height, self.target_width, img_array.shape[2]), dtype=img_array.dtype)
            else:  # Grayscale
                expanded_array = np.zeros((self.target_height, self.target_width), dtype=img_array.dtype)
            
            # Copy original image to center
            y_start = top_pad
            y_end = y_start + orig_height
            x_start = left_pad
            x_end = x_start + orig_width
            expanded_array[y_start:y_end, x_start:x_end] = img_array
            
            # Process vertical expansion (top)
            if top_pad > 0:
                top_line = img_array[0]  # First line of original image
                for i in range(top_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (top_pad - i) / top_pad if top_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply horizontal blur and luminance reduction
                    blurred_line = self.apply_horizontal_blur(top_line, blur_amount)
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                    expanded_array[i, x_start:x_end] = final_line
            
            # Process vertical expansion (bottom)
            if bottom_pad > 0:
                bottom_line = img_array[-1]  # Last line of original image
                for i in range(bottom_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (i + 1) / bottom_pad if bottom_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply horizontal blur and luminance reduction
                    blurred_line = self.apply_horizontal_blur(bottom_line, blur_amount)
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                    expanded_array[y_end + i, x_start:x_end] = final_line
            
            # Process horizontal expansion (left)
            if left_pad > 0:
                left_column = expanded_array[:, x_start]  # First column of current image
                for i in range(left_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (left_pad - i) / left_pad if left_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply vertical blur and luminance reduction
                    blurred_column = self.apply_vertical_blur(left_column, blur_amount)
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reduction)
                    expanded_array[:, i] = final_column
            
            # Process horizontal expansion (right)
            if right_pad > 0:
                right_column = expanded_array[:, x_end - 1]  # Last column of current image
                for i in range(right_pad):
                    # Calculate effects (0 to max at outermost)
                    progress = (i + 1) / right_pad if right_pad > 0 else 0
                    blur_amount = progress * max_blur
                    luminance_reduction = progress * max_luminance_drop
                    
                    # Apply vertical blur and luminance reduction
                    blurred_column = self.apply_vertical_blur(right_column, blur_amount)
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reduction)
                    expanded_array[:, x_end + i] = final_column
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(expanded_array.astype('uint8'))
            
            # Save with format-specific options
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                processed_image.save(output_path, 'JPEG', quality=100, optimize=True)
            else:
                processed_image.save(output_path, 'PNG', optimize=True)
            
            img.close()
            processed_image.close()
            
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
            
            # Analyze all images and store their info
            image_info = []
            for filename in image_files:
                filepath = os.path.join(self.input_directory, filename)
                try:
                    with Image.open(filepath) as img:
                        image_info.append({
                            'path': filepath,
                            'size': img.size,
                            'filename': filename
                        })
                except Exception as e:
                    self.log_message(f"Error processing {filename}: {e}")
            
            if not image_info:
                self.log_message("No valid images found for processing")
                return
            
            # Create montages from all available images (mixed sizes allowed)
            total_montages = 0
            successful = 0
            failed = 0
            images_processed = 0
            
            while images_processed < len(image_info):
                # Calculate remaining images
                remaining_images = len(image_info) - images_processed
                
                if remaining_images < self.min_images_per_montage:
                    self.log_message(f"Remaining {remaining_images} images are insufficient for montage (need at least {self.min_images_per_montage})")
                    break
                
                # Determine batch size for this montage (start with minimum and find best fit)
                best_count = self.min_images_per_montage
                
                # Test different batch sizes to find the largest feasible one
                for test_count in range(self.min_images_per_montage, min(self.max_images_per_montage, remaining_images) + 1):
                    # Test this batch - get sample image sizes from the batch
                    test_batch = image_info[images_processed:images_processed + test_count]
                    test_sizes = [info['size'] for info in test_batch]
                    
                    # Use the same logic as create_montage for consistency
                    test_widths = [size[0] for size in test_sizes]
                    test_heights = [size[1] for size in test_sizes]
                    test_widths.sort()
                    test_heights.sort()
                    
                    # Try median dimensions first (matches create_montage logic)
                    median_width = test_widths[len(test_widths) // 2]
                    median_height = test_heights[len(test_heights) // 2]
                    
                    can_montage, _ = self.can_create_montage(median_width, median_height, test_count)
                    
                    # If median fails, try minimum (also matches create_montage fallback)
                    if not can_montage:
                        min_width = min(test_widths)
                        min_height = min(test_heights)
                        can_montage, _ = self.can_create_montage(min_width, min_height, test_count)
                    
                    if can_montage:
                        best_count = test_count
                    else:
                        break  # Stop at first invalid configuration
                
                # Take images for this montage
                end_idx = images_processed + best_count
                batch_info = image_info[images_processed:end_idx]
                montage_paths = [info['path'] for info in batch_info]
                
                # Analyze sizes in this batch
                sizes_in_batch = [info['size'] for info in batch_info]
                unique_sizes = list(set(sizes_in_batch))
                
                # Determine output format
                if self.save_as_jpg.get():
                    ext = ".jpg"
                else:
                    ext = ".png"
                
                # Generate output filename based on batch composition
                if len(unique_sizes) > 1:
                    # Mixed sizes - use dimensions of first image and indicate mixed
                    first_size = batch_info[0]['size']
                    montage_prefix = f"{first_size[0]}x{first_size[1]}_mixed_montage_{len(montage_paths)}imgs"
                    self.log_message(f"Creating mixed-size montage with {len(montage_paths)} images:")
                    for size in unique_sizes:
                        count = sizes_in_batch.count(size)
                        self.log_message(f"  - {count} images at {size[0]}x{size[1]}")
                else:
                    # Single size
                    size = unique_sizes[0]
                    montage_prefix = f"{size[0]}x{size[1]}_montage_{len(montage_paths)}imgs"
                    self.log_message(f"Creating montage with {len(montage_paths)} images at {size[0]}x{size[1]}")
                
                output_filename, output_path = self.generate_timestamp_filename(output_dir, montage_prefix, ext)
                
                # Update progress
                total_montages += 1
                progress = (images_processed / len(image_info)) * 100
                self.progress_var.set(min(progress, 100))
                
                if len(montage_paths) > 1:
                    self.progress_label.config(text=f"Creating montage: {output_filename}")
                    # Try to create montage
                    if self.create_montage(montage_paths, output_path):
                        successful += 1
                        image_names = [os.path.basename(p) for p in montage_paths]
                        self.log_message(f"✓ Created montage: {output_filename} from {len(montage_paths)} images")
                        self.log_message(f"  Source images: {', '.join(image_names[:3])}{'...' if len(image_names) > 3 else ''}")
                    else:
                        # Montage failed - fall back to processing images individually
                        self.log_message(f"⚠ Montage creation failed - falling back to individual processing")
                        fallback_successful = 0
                        fallback_failed = 0
                        
                        for i, image_path in enumerate(montage_paths):
                            # Generate individual filename
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            if self.save_as_jpg.get():
                                fallback_ext = ".jpg"
                            else:
                                fallback_ext = ".png"
                            
                            fallback_filename, fallback_path = self.generate_timestamp_filename(
                                output_dir, f"{base_name}_scaled", fallback_ext
                            )
                            
                            self.progress_label.config(text=f"Processing fallback {i+1}/{len(montage_paths)}: {fallback_filename}")
                            
                            if self.process_single_image(image_path, fallback_path):
                                fallback_successful += 1
                                self.log_message(f"✓ Processed individual image: {fallback_filename}")
                            else:
                                fallback_failed += 1
                                self.log_message(f"✗ Failed to process individual image: {fallback_filename}")
                        
                        successful += fallback_successful
                        failed += fallback_failed
                        
                        if fallback_successful > 0:
                            self.log_message(f"✓ Fallback completed: {fallback_successful} images processed individually")
                        if fallback_failed > 0:
                            self.log_message(f"✗ Fallback failures: {fallback_failed} images failed")
                else:
                    # Process single image
                    self.progress_label.config(text=f"Processing single: {output_filename}")
                    if self.process_single_image(montage_paths[0], output_path):
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