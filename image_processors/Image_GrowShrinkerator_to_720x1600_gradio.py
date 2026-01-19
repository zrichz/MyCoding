"""
Image Auto-Expander - Batch Processing Gradio Application (Percentage-Based Cropping)

This script provides a GUI for batch processing images to a fixed 720x1600 pixels
with percentage-based width cropping.

Key Features:
- Select directory containing images using file dialog
- Accepts ANY initial image size
- Configurable width crop percentage (0-25% in 1% steps): crops from each side
- Output format selection: choose between high-quality JPEG (Q=100) or PNG format
- Automatically processes images to exactly 720x1600 pixels (width x height)
- Intelligent preprocessing: center crops width by percentage, expands height to (20/9) aspect ratio, then scales
- Fixed, 160px maximum blur applied to expanded regions
- Fixed, 25% luminance reduction for fade effect
- Generates unique timestamped filenames with conflict resolution
"""

from PIL import Image
import gradio as gr
import numpy as np
from scipy import ndimage
import os
from datetime import datetime
from tkinter import Tk, filedialog

# Ensure numpy is properly imported
try:
    import numpy as np
except ImportError:
    print("NumPy is required but not installed. Please install it using: pip install numpy")
    exit(1)


class ImageExpanderProcessor:
    def __init__(self):
        # Target dimensions
        self.target_width = 720
        self.target_height = 1600
        
        # Fixed settings
        self.blur_amount = 160
        self.luminance_drop = 25
        
        # Pre-compute and cache Gaussian kernels for performance optimization
        self.kernel_cache = {}
        self._initialize_kernel_cache()
    
    def _initialize_kernel_cache(self):
        """Pre-compute and cache Gaussian kernels for all blur amounts we'll need."""
        max_blur = self.blur_amount  # 160px maximum
        
        # Create kernels for blur amounts in steps of 0.5 (sufficient precision)
        blur_steps = np.arange(0, max_blur + 0.5, 0.5)
        
        print(f"Pre-computing {len(blur_steps)} Gaussian kernels for blur optimization...")
        
        for blur_amount in blur_steps:
            if blur_amount > 0:
                # Calculate kernel parameters
                kernel_size = int(blur_amount * 2) * 2 + 1  # Ensure odd size
                sigma = blur_amount / 3.0  # Convert blur_amount to sigma
                
                # Create 1D Gaussian kernel using vectorized NumPy operations
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-x**2 / (2 * sigma**2))
                kernel = kernel / kernel.sum()  # Normalize
                
                # Cache both the kernel and its parameters for quick lookup
                self.kernel_cache[blur_amount] = {
                    'kernel': kernel.astype(np.float32),
                    'kernel_size': kernel_size,
                    'sigma': sigma
                }
        
        print(f"Kernel cache initialized with {len(self.kernel_cache)} pre-computed kernels")
    
    def browse_folder(self):
        """Open a folder selection dialog."""
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title="Select Image Directory")
        root.destroy()
        return folder_path if folder_path else ""
    
    def get_image_files(self, directory):
        """Get list of supported image files in directory"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in os.listdir(directory):
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                image_files.append(filename)
        
        return sorted(image_files)
    
    def generate_timestamp_filename(self, output_dir, ext):
        """Generate a timestamp-based filename with conflict resolution"""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        
        counter = 1
        while True:
            counter_str = f"{counter:03d}"
            filename = f"{timestamp}_{counter_str}_720x1600{ext}"
            output_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(output_path):
                return filename, output_path
            
            counter += 1
            
            if counter > 999:
                raise Exception("Too many files with the same timestamp")
    
    def apply_horizontal_blur(self, line_array, blur_amount):
        """Apply horizontal-only blur using pre-cached Gaussian kernels"""
        if blur_amount <= 0:
            return line_array
        
        blur_key = round(blur_amount * 2) / 2.0
        
        if blur_key not in self.kernel_cache:
            available_keys = list(self.kernel_cache.keys())
            blur_key = min(available_keys, key=lambda x: abs(x - blur_amount))
        
        kernel = self.kernel_cache[blur_key]['kernel']
        
        if len(line_array.shape) == 2:  # Color image
            line_float = line_array.astype(np.float32)
            blurred = np.zeros_like(line_float)
            
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
        
        blur_key = round(blur_amount * 2) / 2.0
        
        if blur_key not in self.kernel_cache:
            available_keys = list(self.kernel_cache.keys())
            blur_key = min(available_keys, key=lambda x: abs(x - blur_amount))
        
        kernel = self.kernel_cache[blur_key]['kernel']
        
        if len(column_array.shape) == 2:  # Color image
            column_float = column_array.astype(np.float32)
            blurred = np.zeros_like(column_float)
            
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
        """Apply luminance reduction to darken the line"""
        if luminance_factor <= 0:
            return line_array
        
        reduction_multiplier = 1.0 - (luminance_factor / 100.0)
        reduced = line_array.astype(np.float32) * reduction_multiplier
        return np.clip(reduced, 0, 255).astype(np.uint8)
    
    def process_single_image(self, input_path, output_path, crop_percent_per_side):
        """Process a single image file with percentage-based cropping"""
        try:
            # Load image
            original_image = Image.open(input_path)
            
            # Step 1: Apply percentage-based center crop to width only
            if crop_percent_per_side > 0:
                # Calculate total crop percentage (both sides)
                total_crop_percent = crop_percent_per_side * 2
                # Calculate new width after cropping
                new_width = int(original_image.width * (1 - total_crop_percent / 100.0))
                # Calculate crop positions (centered)
                left = (original_image.width - new_width) // 2
                right = left + new_width
                # Crop only the width, keep full height
                original_image = original_image.crop((left, 0, right, original_image.height))
            
            # Get the cropped width (z in the requirements)
            z = original_image.width
            
            # Step 2: Calculate target height based on (20/9) aspect ratio
            # Target height should be (20/9) * z
            target_expanded_height = int((20.0 / 9.0) * z)
            
            # Convert to numpy array for processing
            img_array = np.array(original_image)
            orig_height, orig_width = img_array.shape[:2]
            
            # Calculate how much height expansion is needed
            height_expansion = max(0, target_expanded_height - orig_height)
            
            # Vertical padding: 1:2 ratio (top:bottom)
            # Top gets 1/3, bottom gets 2/3 of the expansion
            top_pad = height_expansion // 3
            bottom_pad = height_expansion - top_pad
            
            # Create expanded image array with new dimensions
            final_height = orig_height + height_expansion
            if len(img_array.shape) == 3:  # Color image
                expanded_array = np.zeros((final_height, orig_width, img_array.shape[2]), dtype=img_array.dtype)
            else:  # Grayscale
                expanded_array = np.zeros((final_height, orig_width), dtype=img_array.dtype)
            
            # Copy original image to position (top-weighted due to 1:3, 2:3 ratio)
            y_start = top_pad
            y_end = y_start + orig_height
            expanded_array[y_start:y_end, :] = img_array
            
            max_blur = self.blur_amount
            max_luminance_drop = self.luminance_drop
            
            # Process top and bottom expansions
            if top_pad > 0:
                self._process_top_expansion_vectorized(expanded_array, img_array, 
                                                     top_pad, 0, orig_width, 
                                                     max_blur, max_luminance_drop)
            
            if bottom_pad > 0:
                self._process_bottom_expansion_vectorized(expanded_array, img_array, 
                                                        bottom_pad, y_end, 0, orig_width,
                                                        max_blur, max_luminance_drop)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(expanded_array.astype('uint8'))
            
            # Step 3: Resize to exactly 720x1600 using high-quality resampling
            processed_image = processed_image.resize((self.target_width, self.target_height), 
                                                    Image.Resampling.LANCZOS)
            
            # Save the processed image
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                processed_image.save(output_path, 'JPEG', quality=90, optimize=True)
            else:
                processed_image.save(output_path, 'PNG', optimize=True)
            
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def _process_top_expansion_vectorized(self, expanded_array, img_array, top_pad, x_start, x_end, max_blur, max_luminance_drop):
        """Vectorized processing of top expansion region"""
        if top_pad <= 0:
            return
            
        top_line = img_array[0]
        
        progress_values = np.array([(top_pad - i) / top_pad for i in range(top_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            line_indices = np.where(mask)[0]
            
            if len(line_indices) > 0:
                blurred_line = self.apply_horizontal_blur(top_line, blur_amount)
                
                for idx in line_indices:
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reductions[idx])
                    expanded_array[idx, x_start:x_end] = final_line
    
    def _process_bottom_expansion_vectorized(self, expanded_array, img_array, bottom_pad, y_end, x_start, x_end, max_blur, max_luminance_drop):
        """Vectorized processing of bottom expansion region"""
        if bottom_pad <= 0:
            return
            
        bottom_line = img_array[-1]
        
        progress_values = np.array([(i + 1) / bottom_pad for i in range(bottom_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            line_indices = np.where(mask)[0]
            
            if len(line_indices) > 0:
                blurred_line = self.apply_horizontal_blur(bottom_line, blur_amount)
                
                for idx in line_indices:
                    final_line = self.apply_luminance_reduction(blurred_line, luminance_reductions[idx])
                    expanded_array[y_end + idx, x_start:x_end] = final_line
    
    def _process_left_expansion_vectorized(self, expanded_array, left_pad, x_start, max_blur, max_luminance_drop):
        """Vectorized processing of left expansion region"""
        if left_pad <= 0:
            return
            
        left_column = expanded_array[:, x_start]
        
        progress_values = np.array([(left_pad - i) / left_pad for i in range(left_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            column_indices = np.where(mask)[0]
            
            if len(column_indices) > 0:
                blurred_column = self.apply_vertical_blur(left_column, blur_amount)
                
                for idx in column_indices:
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reductions[idx])
                    expanded_array[:, idx] = final_column
    
    def _process_right_expansion_vectorized(self, expanded_array, right_pad, x_end, max_blur, max_luminance_drop):
        """Vectorized processing of right expansion region"""
        if right_pad <= 0:
            return
            
        right_column = expanded_array[:, x_end - 1]
        
        progress_values = np.array([(i + 1) / right_pad for i in range(right_pad)])
        blur_amounts = progress_values * max_blur
        luminance_reductions = progress_values * max_luminance_drop
        
        unique_blurs = np.unique(np.round(blur_amounts * 2) / 2.0)
        
        for blur_amount in unique_blurs:
            mask = np.abs(blur_amounts - blur_amount) < 0.25
            column_indices = np.where(mask)[0]
            
            if len(column_indices) > 0:
                blurred_column = self.apply_vertical_blur(right_column, blur_amount)
                
                for idx in column_indices:
                    final_column = self.apply_luminance_reduction(blurred_column, luminance_reductions[idx])
                    expanded_array[:, x_end + idx] = final_column
    
    def process_batch(self, input_folder, crop_percent_per_side, save_as_jpg, progress=gr.Progress()):
        """Process all images in the selected directory"""
        if not input_folder or not os.path.exists(input_folder):
            return "Please select a valid input folder."
        
        # Convert string to float
        crop_percent_per_side = float(crop_percent_per_side)
        
        # Get image files
        image_files = self.get_image_files(input_folder)
        total_files = len(image_files)
        
        if total_files == 0:
            return "No image files found in the selected folder."
        
        # Create output directory
        if crop_percent_per_side > 0:
            dir_name = f"processed_720x1600_crop{int(crop_percent_per_side)}pct"
        else:
            dir_name = "processed_720x1600"
        output_dir = os.path.join(input_folder, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        successful = 0
        failed = 0
        log_lines = []
        
        log_lines.append(f"Input folder: {os.path.normpath(input_folder)}")
        log_lines.append(f"Output folder: {os.path.normpath(output_dir)}")
        log_lines.append(f"Width crop: {crop_percent_per_side}% per side ({crop_percent_per_side * 2}% total)")
        log_lines.append(f"Format: {'JPEG (Q=90)' if save_as_jpg else 'PNG'}")
        log_lines.append(f"Found {total_files} images to process\n")
        
        for i, filename in enumerate(image_files):
            try:
                progress((i + 1) / total_files, desc=f"Processing {i+1}/{total_files}: {filename}")
                
                input_path = os.path.join(input_folder, filename)
                
                # Determine output format
                ext = ".jpg" if save_as_jpg else ".png"
                output_filename, output_path = self.generate_timestamp_filename(output_dir, ext)
                
                if self.process_single_image(input_path, output_path, crop_percent_per_side):
                    successful += 1
                    log_lines.append(f"✓ {filename} → {output_filename}")
                else:
                    failed += 1
                    log_lines.append(f"✗ Failed: {filename}")
                    
            except Exception as e:
                failed += 1
                log_lines.append(f"✗ Error: {filename} - {str(e)}")
        
        # Summary
        log_lines.append(f"\n{'='*60}")
        log_lines.append(f"Successfully processed: {successful} images")
        log_lines.append(f"Failed: {failed} images")
        log_lines.append(f"Output saved to: {os.path.normpath(output_dir)}")
        
        return "\n".join(log_lines)


# Initialize processor
processor = ImageExpanderProcessor()

# Build Gradio interface with custom CSS for larger text
css = """
    .gradio-container {
        font-size: 18px !important;
    }
    label {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .gr-button {
        font-size: 18px !important;
    }
    h1 {
        font-size: 32px !important;
    }
    p {
        font-size: 18px !important;
    }
    .gr-text-input, .gr-dropdown, .gr-checkbox, textarea {
        font-size: 18px !important;
    }
"""

with gr.Blocks(title="Image Auto-Expander (Percentage)", css=css) as demo:
    gr.Markdown("# Image Auto-Expander - Percentage-Based Cropping (720x1600)")
    gr.Markdown("Accepts any image size • Crops by percentage • Expands to 20:9 ratio • Scales to 720x1600")
    
    browse_btn = gr.Button("📁 Browse Folder", size="lg", scale=1)
    
    input_folder = gr.Textbox(
        label="Input Folder",
        placeholder="Click 'Browse Folder' to select...",
        interactive=True
    )
    
    crop_percent_per_side = gr.Slider(
        minimum=0,
        maximum=25,
        step=1,
        value=0,
        label="Width Crop Percentage (per side)",
        info="0% = no crop, 8% = crops 8% from left + 8% from right (16% total). Crop is applied before expansion."
    )
    
    save_as_jpg = gr.Checkbox(
        label="Save as JPEG (Q=90)",
        value=False,
        info="Otherwise saves PNG"
    )
    
    process_btn = gr.Button("Process All Images", variant="primary", size="lg")
    
    output_log = gr.Textbox(
        label="Processing Log",
        lines=20,
        interactive=False
    )
    
    # Wire up interactions
    browse_btn.click(
        fn=processor.browse_folder,
        inputs=None,
        outputs=input_folder
    )
    
    process_btn.click(
        fn=processor.process_batch,
        inputs=[input_folder, crop_percent_per_side, save_as_jpg],
        outputs=output_log
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
