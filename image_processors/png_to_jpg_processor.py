"""
PNG to JPG Batch Processor with Rotation and Scaling

This script replicates the functionality of a batch file that processes PNG images:

For PORTRAIT/SQUARE images:
1. Scales to max 1080x1440 preserving aspect ratio (using blur fill method)
2. Rotates 90° counter-clockwise
3. Stretches to 1920x1080
4. Saves as high-quality JPEG with numbered filenames

For LANDSCAPE images (width > height):
1. Scales to max 1440x1080 preserving aspect ratio (using blur fill method)
2. Skips rotation step (already in correct orientation)
3. Stretches to 1920x1080
4. Saves as high-quality JPEG with numbered filenames

Uses the image expansion technique from image_expander_720x1600.py for intelligent fill.
Optimizes scaling by using the largest possible dimension for each orientation.
"""

import os
import sys
import glob
from PIL import Image
import numpy as np
from scipy import ndimage
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

class PNGProcessor:
    def __init__(self):
        self.output_dir = "processed_files"
        self.blur_amount = 160  # Blur for fill areas
        self.luminance_drop = 50  # Darken fill areas by 50%
        self.quality = 88  # JPEG quality
        self.source_dir = None  # Directory containing PNG files
        
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
    
    def apply_luminance_reduction(self, line_array, luminance_factor):
        """Apply luminance reduction to darken the line"""
        if luminance_factor <= 0:
            return line_array
        
        # Convert to float for calculation
        line_float = line_array.astype(np.float32)
        
        # Apply luminance reduction (multiply by (1 - factor))
        reduction_multiplier = 1.0 - (luminance_factor / 100.0)
        reduced = line_float * reduction_multiplier
        
        return np.clip(reduced, 0, 255).astype(np.uint8)
    
    def scale_with_fill(self, img_array, target_width, target_height):
        """Scale image preserving aspect ratio, filling gaps with blurred edges"""
        orig_height, orig_width = img_array.shape[:2]
        
        # Calculate scaling to fit within target dimensions
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image to fit within target dimensions
        resized_image = Image.fromarray(img_array).resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_array = np.array(resized_image)
        
        # If already fits perfectly, return as-is
        if new_width == target_width and new_height == target_height:
            return resized_array
        
        # Calculate padding needed
        width_expansion = target_width - new_width
        height_expansion = target_height - new_height
        
        left_pad = width_expansion // 2
        right_pad = width_expansion - left_pad
        top_pad = height_expansion // 2
        bottom_pad = height_expansion - top_pad
        
        # Create expanded image array
        if len(resized_array.shape) == 3:  # Color image
            expanded_array = np.zeros((target_height, target_width, resized_array.shape[2]), dtype=resized_array.dtype)
        else:  # Grayscale
            expanded_array = np.zeros((target_height, target_width), dtype=resized_array.dtype)
        
        # Copy resized image to center
        y_start = top_pad
        y_end = y_start + new_height
        x_start = left_pad
        x_end = x_start + new_width
        expanded_array[y_start:y_end, x_start:x_end] = resized_array
        
        # Fill vertical gaps (top and bottom)
        if top_pad > 0:
            top_line = resized_array[0]  # First line of resized image
            for i in range(top_pad):
                # Calculate effects (0 to max at outermost)
                progress = (top_pad - i) / top_pad if top_pad > 0 else 0
                blur_amount = progress * self.blur_amount
                luminance_reduction = progress * self.luminance_drop
                
                # Apply horizontal blur and luminance reduction
                blurred_line = self.apply_horizontal_blur(top_line, blur_amount)
                final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                expanded_array[i, x_start:x_end] = final_line
        
        if bottom_pad > 0:
            bottom_line = resized_array[-1]  # Last line of resized image
            for i in range(bottom_pad):
                # Calculate effects (0 to max at outermost)
                progress = (i + 1) / bottom_pad if bottom_pad > 0 else 0
                blur_amount = progress * self.blur_amount
                luminance_reduction = progress * self.luminance_drop
                
                # Apply horizontal blur and luminance reduction
                blurred_line = self.apply_horizontal_blur(bottom_line, blur_amount)
                final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                expanded_array[y_end + i, x_start:x_end] = final_line
        
        # Fill horizontal gaps (left and right)
        if left_pad > 0:
            left_column = expanded_array[:, x_start]  # First column of image
            for i in range(left_pad):
                # Calculate effects (0 to max at outermost)
                progress = (left_pad - i) / left_pad if left_pad > 0 else 0
                blur_amount = progress * self.blur_amount
                luminance_reduction = progress * self.luminance_drop
                
                # Apply vertical blur and luminance reduction
                blurred_column = self.apply_vertical_blur(left_column, blur_amount)
                final_column = self.apply_luminance_reduction(blurred_column, luminance_reduction)
                expanded_array[:, i] = final_column
        
        if right_pad > 0:
            right_column = expanded_array[:, x_end - 1]  # Last column of image
            for i in range(right_pad):
                # Calculate effects (0 to max at outermost)
                progress = (i + 1) / right_pad if right_pad > 0 else 0
                blur_amount = progress * self.blur_amount
                luminance_reduction = progress * self.luminance_drop
                
                # Apply vertical blur and luminance reduction
                blurred_column = self.apply_vertical_blur(right_column, blur_amount)
                final_column = self.apply_luminance_reduction(blurred_column, luminance_reduction)
                expanded_array[:, x_end + i] = final_column
        
        return expanded_array
    
    def rotate_90_ccw(self, img_array):
        """Rotate image 90 degrees counter-clockwise"""
        return np.rot90(img_array, k=1)
    
    def stretch_to_target(self, img_array, target_width, target_height):
        """Stretch image to exact target dimensions"""
        image = Image.fromarray(img_array)
        stretched = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return np.array(stretched)
    
    def process_single_png(self, input_file, output_file):
        """Process a single PNG file through the complete pipeline"""
        try:
            # Load PNG
            image = Image.open(input_file)
            
            # Convert to RGB if necessary (for JPEG output)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            orig_height, orig_width = img_array.shape[:2]
            
            # Determine if image is landscape (width > height)
            is_landscape = orig_width > orig_height
            
            if is_landscape:
                # For landscape images: Scale to max 1440x1080 (no rotation needed)
                # This allows maximum scaling: 1440/width instead of 1080/width
                scaled_array = self.scale_with_fill(img_array, 1440, 1080)
                # Skip rotation step
                final_array = self.stretch_to_target(scaled_array, 1920, 1080)
            else:
                # For portrait/square images: Use original pipeline
                # Step 1: Scale to max 1080x1440 preserving aspect ratio with blur fill
                scaled_array = self.scale_with_fill(img_array, 1080, 1440)
                
                # Step 2: Rotate 90° counter-clockwise
                rotated_array = self.rotate_90_ccw(scaled_array)
                
                # Step 3: Stretch to 1920x1080
                final_array = self.stretch_to_target(rotated_array, 1920, 1080)
            
            # Step 4: Save as high-quality JPEG
            final_image = Image.fromarray(final_array)
            final_image.save(output_file, 'JPEG', 
                           quality=88, 
                           optimize=True,
                           progressive=False,  # Explicitly ensure baseline JPEG
                           subsampling=2)      # 4:2:0 chroma subsampling
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            return False
    
    def process_all_pngs(self, source_directory=None, progress_callback=None):
        """Process all PNG files in the specified directory"""
        if source_directory:
            self.source_dir = source_directory
        
        # Use current directory if no source specified
        if not self.source_dir:
            self.source_dir = os.getcwd()
        
        # Create output directory as subdirectory of source
        self.output_dir = os.path.join(self.source_dir, "processed_files")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Find all PNG files in source directory
        png_pattern = os.path.join(self.source_dir, "*.png")
        png_files = glob.glob(png_pattern)
        
        if not png_files:
            if progress_callback:
                progress_callback("No PNG files found in selected directory.", True)
            else:
                print("No PNG files found in selected directory.")
            return
        
        total_files = len(png_files)
        if progress_callback:
            progress_callback(f"Found {total_files} PNG file(s). Starting processing...", False)
        else:
            print(f"Found {total_files} PNG file(s):")
        
        count = 0
        for png_file in png_files:
            count += 1
            
            # Generate output filename: IMG00001.jpg, IMG00002.jpg, etc.
            output_filename = f"IMG{count:05d}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)
            
            filename_only = os.path.basename(png_file)
            status_msg = f"Processing {count}/{total_files}: {filename_only} → {output_filename}"
            
            if progress_callback:
                progress_callback(status_msg, False)
            else:
                print(f"Processing: {filename_only} → {output_filename}")
            
            if self.process_single_png(png_file, output_path):
                if not progress_callback:
                    print(f"  ✓ Success")
            else:
                if not progress_callback:
                    print(f"  ✗ Failed")
        
        final_msg = f"Done! Converted {count} image(s) to 1920x1080 JPEGs.\nSaved in \"{self.output_dir}\" folder."
        if progress_callback:
            progress_callback(final_msg, True)
        else:
            print(f"\nDone. Converted {count} image(s) to 1920x1080 JPEGs.")
            print(f"Saved in \"{self.output_dir}\" folder.")


class PNGProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PNG to JPG Batch Processor")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        self.processor = PNGProcessor()
        self.processing = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="PNG to JPG Batch Processor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Description
        desc_text = ("-Processes PNGs through the following pipeline:\n"
                    "1. Scales to max 1080x1440 preserving aspect ratio (with blur fill)\n"
                    "2. Rotate 90° counter-clockwise\n"
                    "3. Stretch to 1920x1080\n"
                    "4. Save as high-quality baseline JPEG files")
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.LEFT)
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20), sticky=tk.W)
        
        # Directory selection
        ttk.Label(main_frame, text="Source Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.dir_var = tk.StringVar()
        self.dir_entry = ttk.Entry(main_frame, textvariable=self.dir_var, width=50)
        self.dir_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=(10, 5))
        
        self.browse_btn = ttk.Button(main_frame, text="Browse...", command=self.browse_directory)
        self.browse_btn.grid(row=2, column=2, pady=5)
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Start Processing", 
                                     command=self.start_processing, style="Accent.TButton")
        self.process_btn.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # Status text
        self.status_text = tk.Text(main_frame, height=8, wrap=tk.WORD)
        self.status_text.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        
        # Scrollbar for status text
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=5, column=3, sticky="ns", pady=(0, 10))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure text area to expand
        main_frame.rowconfigure(5, weight=1)
    
    def browse_directory(self):
        """Open directory browser"""
        directory = filedialog.askdirectory(title="Select directory containing PNG files")
        if directory:
            self.dir_var.set(directory)
            self.update_status(f"Selected directory: {directory}\n")
    
    def update_status(self, message, is_final=False):
        """Update status text"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
        if is_final:
            self.processing = False
            self.progress.stop()
            self.process_btn.configure(text="Start Processing", state="normal")
            messagebox.showinfo("Processing Complete", message)
    
    def process_files_thread(self):
        """Process files in a separate thread"""
        try:
            # Process files (settings are already hardcoded in processor)
            self.processor.process_all_pngs(
                source_directory=self.dir_var.get(),
                progress_callback=self.update_status
            )
        except Exception as e:
            self.update_status(f"Error during processing: {str(e)}", True)
    
    def start_processing(self):
        """Start the processing workflow"""
        if self.processing:
            return
        
        source_dir = self.dir_var.get().strip()
        if not source_dir:
            messagebox.showerror("Error", "Please select a source directory first.")
            return
        
        if not os.path.exists(source_dir):
            messagebox.showerror("Error", "Selected directory does not exist.")
            return
        
        # Check for PNG files
        png_pattern = os.path.join(source_dir, "*.png")
        png_files = glob.glob(png_pattern)
        if not png_files:
            messagebox.showwarning("Warning", "No PNG files found in the selected directory.")
            return
        
        # Start processing
        self.processing = True
        self.process_btn.configure(text="Processing...", state="disabled")
        self.progress.start()
        self.status_text.delete(1.0, tk.END)
        
        # Run processing in separate thread
        processing_thread = threading.Thread(target=self.process_files_thread)
        processing_thread.daemon = True
        processing_thread.start()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    """Main function - launches GUI interface"""
    app = PNGProcessorGUI()
    app.run()


if __name__ == "__main__":
    main()
