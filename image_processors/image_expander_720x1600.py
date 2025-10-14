"""
Image Auto-Expander - Batch Processing GUI Application

This script provides a gui for batch processing images to a fixed 720x1600 pixels
with configurable initial crop width.

Key Features:
- Select directory containing images
- Configurable initial crop width (512-720px): crops oversized images horizontally before scaling
- Output format selection: choose between high-quality JPEG (Q=1/100%) or PNG format
- Automatically scales cropped images to exactly 720x1600 pixels (width x height)
- Intelligent preprocessing: center crops images wider than selected crop width, then scales to 720px wide
- Intelligent expansion: only expands dimensions that are smaller than target
- Fixed, 160px maximum blur applied to expanded regions
- Fixed, 25% luminance reduction for fade effect
- Generates unique timestamped filenames with conflict resolution
"""

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy import ndimage
import os

# Ensure numpy is properly imported
try:
    import numpy as np
except ImportError:
    print("NumPy is required but not installed. Please install it using: pip install numpy")
    exit(1)
from datetime import datetime

class ImageExpander:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Auto-Expander - Batch Processor (720x1600)")
        self.root.geometry("900x600")
        
        self.input_directory = None
        
        # Target dimensions
        self.target_width = 720
        self.target_height = 1600
        
        # Configurable crop width (512-720 in multiples of 16)
        self.crop_width = 720  # Default value
        
        # Fixed settings
        self.blur_amount = 160
        self.luminance_drop = 25
        
        # Pre-compute and cache Gaussian kernels for performance optimization
        self.kernel_cache = {}
        self._initialize_kernel_cache()
        
        self.setup_gui()
    
    def _initialize_kernel_cache(self):
        """Pre-compute and cache Gaussian kernels for all blur amounts we'll need.
        
        This eliminates the expensive kernel calculation bottleneck by pre-computing
        kernels for blur amounts from 0 to max_blur_amount in increments that match
        the precision we actually use during processing.
        """
        max_blur = self.blur_amount  # 160px maximum
        
        # Create kernels for blur amounts in steps of 0.5 (sufficient precision)
        # This covers all possible blur values we'll encounter during processing
        blur_steps = np.arange(0, max_blur + 0.5, 0.5)
        
        print(f"Pre-computing {len(blur_steps)} Gaussian kernels for blur optimization...")
        
        for blur_amount in blur_steps:
            if blur_amount > 0:
                # Calculate kernel parameters (same logic as original)
                kernel_size = int(blur_amount * 2) * 2 + 1  # Ensure odd size
                sigma = blur_amount / 3.0  # Convert blur_amount to sigma
                
                # Create 1D Gaussian kernel using vectorized NumPy operations
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-x**2 / (2 * sigma**2))
                kernel = kernel / kernel.sum()  # Normalize
                
                # Cache both the kernel and its parameters for quick lookup
                self.kernel_cache[blur_amount] = {
                    'kernel': kernel.astype(np.float32),  # Use float32 for memory efficiency
                    'kernel_size': kernel_size,
                    'sigma': sigma
                }
        
        print(f"Kernel cache initialized with {len(self.kernel_cache)} pre-computed kernels")
    
    def setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(main_frame, text="Image Auto-Expander - Batch Processor (720x1600)", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Settings info frame
        info_frame = tk.LabelFrame(main_frame, text="Processing Settings", 
                                  font=("Arial", 10, "bold"))
        info_frame.pack(pady=(0, 20), fill='x')
        
        # Crop width setting
        crop_frame = tk.Frame(info_frame)
        crop_frame.pack(pady=10)
        
        tk.Label(crop_frame, text="Initial Crop Width (then scaled to 720px):", 
                font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # Create crop width variable and dropdown
        self.crop_width_var = tk.StringVar(value="720")
        crop_values = [str(x) for x in range(512, 721, 16)]  # 512 to 720 in steps of 16
        crop_dropdown = tk.OptionMenu(crop_frame, self.crop_width_var, *crop_values, 
                                     command=self.on_crop_width_change)
        crop_dropdown.config(font=("Arial", 9))
        crop_dropdown.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Label(crop_frame, text="pixels", font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Output format setting
        format_frame = tk.Frame(info_frame)
        format_frame.pack(pady=10)
        
        self.save_as_jpg = tk.BooleanVar(value=False)  # Default to PNG
        jpg_checkbox = tk.Checkbutton(format_frame, text="Save as high-quality JPEG (Q=1, otherwise PNG)", 
                                     variable=self.save_as_jpg, font=("Arial", 9, "bold"),
                                     command=self.on_format_change)
        jpg_checkbox.pack(side=tk.LEFT)
        
        # Fixed settings info
        # Update settings info
        settings_text = f"Fixed Settings: 160px blur, 25% luminance reduction\nTarget: 720x1600 pixels (crop width: {self.crop_width}px)"
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
        self.process_button = tk.Button(button_frame, text="Process All Images", 
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
        self.log_message("Ready to process images. Select a directory to begin.")
    
    def on_crop_width_change(self, value):
        """Handle crop width dropdown change"""
        self.crop_width = int(value)
        # Update the settings label
        settings_text = f"Fixed Settings: 160px blur, 25% luminance reduction\nTarget: 720x1600 pixels (crop width: {self.crop_width}px)"
        self.settings_label.config(text=settings_text)
        self.log_message(f"Crop width changed to {self.crop_width}px")
    
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
        directory = filedialog.askdirectory(title="Select Image Directory")
        
        if directory:
            self.input_directory = directory
            # Count image files
            image_files = self.get_image_files(directory)
            count = len(image_files)
            
            self.log_message(f"Selected directory: {os.path.normpath(directory)}")
            self.log_message(f"Found {count} image files")
            self.progress_label.config(text=f"Directory selected: {count} images found")
            
            if count > 0:
                self.process_button.config(state=tk.NORMAL)
            else:
                self.process_button.config(state=tk.DISABLED)
                messagebox.showwarning("No Images", "No supported image files found.")
    
    def get_image_files(self, directory):
        """Get list of supported image files in directory"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in os.listdir(directory):
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                image_files.append(filename)
        
        return sorted(image_files)
    
    def process_all_images(self):
        """Process all images in the selected directory"""
        if not self.input_directory:
            return
        
        try:
            # Disable button during processing
            self.process_button.config(state=tk.DISABLED)
            
            # Get image files
            image_files = self.get_image_files(self.input_directory)
            total_files = len(image_files)
            
            if total_files == 0:
                self.log_message("No image files to process")
                return
            
            # Create output directory with crop width in name
            dir_name = f"processed_720x1600_{self.crop_width}crop"
            output_dir = os.path.join(self.input_directory, dir_name)
            os.makedirs(output_dir, exist_ok=True)
            self.log_message(f"Created output directory: {os.path.normpath(output_dir)}")
            
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
                    name, orig_ext = os.path.splitext(filename)
                    
                    # Determine output format based on user selection
                    if self.save_as_jpg.get():
                        ext = ".jpg"
                    else:
                        ext = ".png"
                    
                    # Use new timestamp-based filename schema
                    output_filename, output_path = self.generate_timestamp_filename(output_dir, ext)
                    
                    if self.process_single_image(input_path, output_path):
                        successful += 1
                        self.log_message(f"✓ Processed: {filename} -> {output_filename}")
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
            self.log_message(f"Output directory: {os.path.normpath(output_dir)}")
            
            if successful > 0:
                messagebox.showinfo("Processing Complete", 
                                   f"Successfully processed {successful} images!\n"
                                   f"Output saved to: {os.path.normpath(output_dir)}")
            
        except Exception as e:
            self.log_message(f"Error during batch processing: {str(e)}")
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")
        
        finally:
            # Re-enable button
            self.process_button.config(state=tk.NORMAL)
    
    def generate_timestamp_filename(self, output_dir, ext):
        """Generate a timestamp-based filename with conflict resolution"""
        # Get current timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        
        # Start with counter 001
        counter = 1
        
        while True:
            # Format counter as 3-digit number
            counter_str = f"{counter:03d}"
            filename = f"{timestamp}_{counter_str}_720x1600{ext}"
            output_path = os.path.join(output_dir, filename)
            
            # Check if file exists
            if not os.path.exists(output_path):
                return filename, output_path
            
            # Increment counter and try again
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 999:
                raise Exception("Too many files with the same timestamp - cannot generate unique filename")
    
    def apply_horizontal_blur(self, line_array, blur_amount):
        """Apply horizontal-only blur using pre-cached Gaussian kernels"""
        if blur_amount <= 0:
            return line_array
        
        # Round blur_amount to nearest 0.5 to match our cache precision
        blur_key = round(blur_amount * 2) / 2.0
        
        # Get cached kernel (fallback to nearest if exact match not found)
        if blur_key not in self.kernel_cache:
            # Find closest cached blur amount
            available_keys = list(self.kernel_cache.keys())
            blur_key = min(available_keys, key=lambda x: abs(x - blur_amount))
        
        kernel = self.kernel_cache[blur_key]['kernel']
        
        # Apply horizontal convolution using vectorized operations
        if len(line_array.shape) == 2:  # Color image (width, channels)
            # Vectorized processing: handle all channels at once
            line_float = line_array.astype(np.float32)
            blurred = np.zeros_like(line_float)
            
            # Process all channels in a vectorized manner
            for channel in range(line_array.shape[1]):
                blurred[:, channel] = ndimage.convolve1d(
                    line_float[:, channel], 
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
        """Apply vertical-only blur using pre-cached Gaussian kernels"""
        if blur_amount <= 0:
            return column_array
        
        # Round blur_amount to nearest 0.5 to match our cache precision
        blur_key = round(blur_amount * 2) / 2.0
        
        # Get cached kernel (fallback to nearest if exact match not found)
        if blur_key not in self.kernel_cache:
            # Find closest cached blur amount
            available_keys = list(self.kernel_cache.keys())
            blur_key = min(available_keys, key=lambda x: abs(x - blur_amount))
        
        kernel = self.kernel_cache[blur_key]['kernel']
        
        # Apply vertical convolution using vectorized operations
        if len(column_array.shape) == 2:  # Color image (height, channels)
            # Vectorized processing: handle all channels at once
            column_float = column_array.astype(np.float32)
            blurred = np.zeros_like(column_float)
            
            # Process all channels in a vectorized manner
            for channel in range(column_array.shape[1]):
                blurred[:, channel] = ndimage.convolve1d(
                    column_float[:, channel], 
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
    
    def apply_luminance_reduction(self, line_array, luminance_factor):
        """Apply luminance reduction to darken the line using vectorized operations"""
        if luminance_factor <= 0:
            return line_array
        
        # Vectorized luminance reduction calculation
        # This single operation replaces multiple array operations
        reduction_multiplier = 1.0 - (luminance_factor / 100.0)
        
        # Apply reduction using vectorized NumPy operations (much faster than loops)
        reduced = line_array.astype(np.float32) * reduction_multiplier
        
        # Convert back to uint8 with clamping in one vectorized operation
        return np.clip(reduced, 0, 255).astype(np.uint8)
    
    def apply_luminance_reduction_batch(self, array_batch, luminance_factors):
        """Apply luminance reduction to multiple arrays at once for vectorized processing"""
        if len(luminance_factors) == 0:
            return array_batch
        
        # Convert to float32 for calculations
        batch_float = array_batch.astype(np.float32)
        
        # Create reduction multipliers array for vectorized operation
        reduction_multipliers = 1.0 - (np.array(luminance_factors) / 100.0)
        
        # Apply reductions using broadcasting - much faster than individual operations
        # This handles multiple luminance reductions in a single vectorized operation
        for i, multiplier in enumerate(reduction_multipliers):
            batch_float[i] *= multiplier
        
        # Convert back to uint8 with clamping
        return np.clip(batch_float, 0, 255).astype(np.uint8)
    
    def process_single_image(self, input_path, output_path):
        """Process a single image file"""
        try:
            # Load image
            original_image = Image.open(input_path)
            
            # Center crop horizontally if image is wider than crop width
            if original_image.width > self.crop_width:
                # Calculate crop coordinates for center cropping
                left = (original_image.width - self.crop_width) // 2
                right = left + self.crop_width
                original_image = original_image.crop((left, 0, right, original_image.height))
            
            # Scale cropped image to 720px wide while maintaining aspect ratio
            if original_image.width != 720:
                # Calculate new height maintaining aspect ratio
                aspect_ratio = original_image.height / original_image.width
                new_height = int(720 * aspect_ratio)
                original_image = original_image.resize((720, new_height), Image.Resampling.LANCZOS)
            
            # Convert image to numpy array
            img_array = np.array(original_image)
            orig_height, orig_width = img_array.shape[:2]
            
            # Use fixed settings
            max_blur = self.blur_amount
            max_luminance_drop = self.luminance_drop
            
            # Calculate expansion needed (width should always be 720 after scaling)
            width_expansion = max(0, 720 - orig_width)
            height_expansion = max(0, self.target_height - orig_height)
            
            # Calculate padding for each side
            left_pad = width_expansion // 2
            right_pad = width_expansion - left_pad
            top_pad = height_expansion // 2
            bottom_pad = height_expansion - top_pad
            
            # Create expanded image array (always 720px wide after scaling)
            if len(img_array.shape) == 3:  # Color image
                expanded_array = np.zeros((self.target_height, 720, img_array.shape[2]), dtype=img_array.dtype)
            else:  # Grayscale
                expanded_array = np.zeros((self.target_height, 720), dtype=img_array.dtype)
            
            # Copy original image to center
            y_start = top_pad
            y_end = y_start + orig_height
            x_start = left_pad
            x_end = x_start + orig_width
            expanded_array[y_start:y_end, x_start:x_end] = img_array
            
            # OPTIMIZED: Process vertical expansion (top) using vectorized operations
            if top_pad > 0:
                self._process_top_expansion_vectorized(expanded_array, img_array, 
                                                     top_pad, x_start, x_end, 
                                                     max_blur, max_luminance_drop)
            
            # OPTIMIZED: Process vertical expansion (bottom) using vectorized operations
            if bottom_pad > 0:
                self._process_bottom_expansion_vectorized(expanded_array, img_array, 
                                                        bottom_pad, y_end, x_start, x_end,
                                                        max_blur, max_luminance_drop)
            
            # OPTIMIZED: Process horizontal expansion (left) using vectorized operations
            if left_pad > 0:
                self._process_left_expansion_vectorized(expanded_array, left_pad, x_start,
                                                      max_blur, max_luminance_drop)
            
            # OPTIMIZED: Process horizontal expansion (right) using vectorized operations
            if right_pad > 0:
                self._process_right_expansion_vectorized(expanded_array, right_pad, x_end,
                                                       max_blur, max_luminance_drop)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(expanded_array.astype('uint8'))
            
            # Save the processed image with format-specific options
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                # Save as highest quality JPEG (quality=100, Q=1 equivalent)
                processed_image.save(output_path, 'JPEG', quality=100, optimize=True)
            else:
                # Save as PNG (default)
                processed_image.save(output_path, 'PNG', optimize=True)
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def _process_top_expansion_vectorized(self, expanded_array, img_array, top_pad, x_start, x_end, max_blur, max_luminance_drop):
        """Vectorized processing of top expansion region - MAJOR PERFORMANCE IMPROVEMENT"""
        if top_pad <= 0:
            return
            
        # Get source line (first line of original image)
        top_line = img_array[0]  
        
        # Pre-calculate all blur amounts and luminance reductions for the entire region
        # This vectorized calculation replaces the individual loop calculations
        progress_values = np.array([(top_pad - i) / top_pad for i in range(top_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        # Group similar blur amounts to minimize kernel switching
        # This batching optimization reduces kernel lookups significantly
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)  # Round to 0.5 precision
        
        for blur_amount in unique_blurs:
            # Find all lines that use this blur amount
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            line_indices = np.where(mask)[0]
            
            if len(line_indices) > 0:
                # Apply blur once for all lines using this blur amount
                blurred_line = self.apply_horizontal_blur(top_line, blur_amount)
                
                # Apply different luminance reductions to each line in a vectorized manner
                for idx in line_indices:
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reductions[idx])
                    expanded_array[idx, x_start:x_end] = final_line
    
    def _process_bottom_expansion_vectorized(self, expanded_array, img_array, bottom_pad, y_end, x_start, x_end, max_blur, max_luminance_drop):
        """Vectorized processing of bottom expansion region - MAJOR PERFORMANCE IMPROVEMENT"""
        if bottom_pad <= 0:
            return
            
        # Get source line (last line of original image)
        bottom_line = img_array[-1]
        
        # Pre-calculate all blur amounts and luminance reductions vectorized
        progress_values = np.array([(i + 1) / bottom_pad for i in range(bottom_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        # Group similar blur amounts for batch processing
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            # Find all lines that use this blur amount
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            line_indices = np.where(mask)[0]
            
            if len(line_indices) > 0:
                # Apply blur once for all lines using this blur amount
                blurred_line = self.apply_horizontal_blur(bottom_line, blur_amount)
                
                # Apply different luminance reductions vectorized
                for idx in line_indices:
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reductions[idx])
                    expanded_array[y_end + idx, x_start:x_end] = final_line
    
    def _process_left_expansion_vectorized(self, expanded_array, left_pad, x_start, max_blur, max_luminance_drop):
        """Vectorized processing of left expansion region - MAJOR PERFORMANCE IMPROVEMENT"""
        if left_pad <= 0:
            return
            
        # Get source column (first column of current image)
        left_column = expanded_array[:, x_start]
        
        # Pre-calculate all blur amounts and luminance reductions vectorized
        progress_values = np.array([(left_pad - i) / left_pad for i in range(left_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        # Group similar blur amounts for batch processing
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            # Find all columns that use this blur amount
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            column_indices = np.where(mask)[0]
            
            if len(column_indices) > 0:
                # Apply blur once for all columns using this blur amount
                blurred_column = self.apply_vertical_blur(left_column, blur_amount)
                
                # Apply different luminance reductions vectorized
                for idx in column_indices:
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reductions[idx])
                    expanded_array[:, idx] = final_column
    
    def _process_right_expansion_vectorized(self, expanded_array, right_pad, x_end, max_blur, max_luminance_drop):
        """Vectorized processing of right expansion region - MAJOR PERFORMANCE IMPROVEMENT"""
        if right_pad <= 0:
            return
            
        # Get source column (last column of current image)
        right_column = expanded_array[:, x_end - 1]
        
        # Pre-calculate all blur amounts and luminance reductions vectorized
        progress_values = np.array([(i + 1) / right_pad for i in range(right_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        # Group similar blur amounts for batch processing
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            # Find all columns that use this blur amount
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            column_indices = np.where(mask)[0]
            
            if len(column_indices) > 0:
                # Apply blur once for all columns using this blur amount
                blurred_column = self.apply_vertical_blur(right_column, blur_amount)
                
                # Apply different luminance reductions vectorized
                for idx in column_indices:
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reductions[idx])
                    expanded_array[:, x_end + idx] = final_column
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageExpander()
    app.run()