"""
Image Auto-Expander - Batch Processing Gradio Application

This script provides a GUI for batch processing images to a fixed 720x1600 pixels
with configurable initial crop width.

Key Features:
- Select directory containing images using file dialog
- Configurable initial crop width (512-720px): crops oversized images horizontally before scaling
- Output format selection: choose between high-quality JPEG (Q=100) or PNG format
- Automatically scales cropped images to exactly 720x1600 pixels (width x height)
- Intelligent preprocessing: center crops images wider than selected crop width, then scales to 720px wide
- Intelligent expansion: only expands dimensions that are smaller than target
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
    
    def process_single_image(self, input_path, output_path, width_trim_percent):
        """Process a single image file"""
        try:
            # Load image
            original_image = Image.open(input_path)
            
            # Apply width trim percentage (center crop)
            if width_trim_percent > 0:
                trim_amount = original_image.width * (width_trim_percent / 100.0)
                new_width = int(original_image.width - trim_amount)
                left = (original_image.width - new_width) // 2
                right = left + new_width
                original_image = original_image.crop((left, 0, right, original_image.height))
            
            # Scale to 720px wide while maintaining aspect ratio
            if original_image.width != 720:
                aspect_ratio = original_image.height / original_image.width
                new_height = int(720 * aspect_ratio)
                original_image = original_image.resize((720, new_height), Image.Resampling.LANCZOS)
            
            # Convert image to numpy array
            img_array = np.array(original_image)
            orig_height, orig_width = img_array.shape[:2]
            
            max_blur = self.blur_amount
            max_luminance_drop = self.luminance_drop
            
            # Calculate expansion needed
            width_expansion = max(0, 720 - orig_width)
            height_expansion = max(0, self.target_height - orig_height)
            
            # Calculate padding for each side
            # Horizontal: centered (1:1 ratio)
            left_pad = width_expansion // 2
            right_pad = width_expansion - left_pad
            
            # Vertical: 1:2 ratio (top:bottom)
            # Top gets 1/3, bottom gets 2/3 of the expansion
            top_pad = height_expansion // 3
            bottom_pad = height_expansion - top_pad
            
            # Create expanded image array
            if len(img_array.shape) == 3:  # Color image
                expanded_array = np.zeros((self.target_height, 720, img_array.shape[2]), dtype=img_array.dtype)
            else:  # Grayscale
                expanded_array = np.zeros((self.target_height, 720), dtype=img_array.dtype)
            
            # Copy original image to position (top-weighted due to 1:2 ratio)
            y_start = top_pad
            y_end = y_start + orig_height
            x_start = left_pad
            x_end = x_start + orig_width
            expanded_array[y_start:y_end, x_start:x_end] = img_array
            
            # Process expansions
            if top_pad > 0:
                self._process_top_expansion_vectorized(expanded_array, img_array, 
                                                     top_pad, x_start, x_end, 
                                                     max_blur, max_luminance_drop)
            
            if bottom_pad > 0:
                self._process_bottom_expansion_vectorized(expanded_array, img_array, 
                                                        bottom_pad, y_end, x_start, x_end,
                                                        max_blur, max_luminance_drop)
            
            if left_pad > 0:
                self._process_left_expansion_vectorized(expanded_array, left_pad, x_start,
                                                      max_blur, max_luminance_drop)
            
            if right_pad > 0:
                self._process_right_expansion_vectorized(expanded_array, right_pad, x_end,
                                                       max_blur, max_luminance_drop)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(expanded_array.astype('uint8'))
            
            # Save the processed image
            if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
                processed_image.save(output_path, 'JPEG', quality=100, optimize=True)
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
    
    def process_batch(self, input_folder, width_trim_percent, save_as_jpg, progress=gr.Progress()):
        """Process all images in the selected directory"""
        if not input_folder or not os.path.exists(input_folder):
            return "Please select a valid input folder."
        
        # Convert string to float
        width_trim_percent = float(width_trim_percent)
        
        # Get image files
        image_files = self.get_image_files(input_folder)
        total_files = len(image_files)
        
        if total_files == 0:
            return "No image files found in the selected folder."
        
        # Create output directory
        if width_trim_percent > 0:
            dir_name = f"processed_720x1600_trim{int(width_trim_percent)}pct"
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
        log_lines.append(f"Width trim: {width_trim_percent}% (center crop before scaling)")
        log_lines.append(f"Format: {'JPEG (Q=100)' if save_as_jpg else 'PNG'}")
        log_lines.append(f"Found {total_files} images to process\n")
        
        for i, filename in enumerate(image_files):
            try:
                progress((i + 1) / total_files, desc=f"Processing {i+1}/{total_files}: {filename}")
                
                input_path = os.path.join(input_folder, filename)
                
                # Determine output format
                ext = ".jpg" if save_as_jpg else ".png"
                output_filename, output_path = self.generate_timestamp_filename(output_dir, ext)
                
                if self.process_single_image(input_path, output_path, width_trim_percent):
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

with gr.Blocks(title="Image Auto-Expander", css=css) as demo:
    gr.Markdown("# Image Auto-Expander - Batch Processor (720x1600)")
    gr.Markdown("Batch process images to 720x1600 px with intelligent scaling")
    
    browse_btn = gr.Button("📁 Browse Folder", size="lg", scale=1)
    
    input_folder = gr.Textbox(
        label="Input Folder",
        placeholder="Click 'Browse Folder' to select...",
        interactive=True
    )
    
    width_trim_percent = gr.Radio(
        choices=["0", "5", "10", "15", "20", "25"],
        value="0",
        label="Width Trim Percentage",
        info="Center crop applied before scaling to 720px"
    )

    
    save_as_jpg = gr.Checkbox(
        label="Save as JPEG (Q=100)",
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
        inputs=[input_folder, width_trim_percent, save_as_jpg],
        outputs=output_log
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
