"""
PNG to JPG Quality Converter
Converts PNG images to JPG at two quality levels: 90% and 15%
Creates subdirectories: jpgs_90Q and jpgs_15Q

Author: Copilot
Date: 2026-01-09
"""

import gradio as gr
from PIL import Image, ImageFilter
import os
from pathlib import Path
from tkinter import Tk, filedialog
import numpy as np


def browse_folder():
    """Open a folder selection dialog."""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    folder_path = filedialog.askdirectory(title="Select Folder Containing PNG Files")
    root.destroy()
    return folder_path if folder_path else ""


def apply_distressed_effects(img):
    """Apply noise, CRT scanlines, and color bleeding for distressed look."""
    # Convert to numpy array for processing
    img_array = np.array(img, dtype=np.float32)
    
    # 1. Add noise (grain effect)
    noise = np.random.normal(0, 8, img_array.shape)  # Mean=0, StdDev=8
    img_array = np.clip(img_array + noise, 0, 255)
    
    # 2. Color bleeding (slight blur on color channels)
    img_temp = Image.fromarray(img_array.astype(np.uint8))
    img_temp = img_temp.filter(ImageFilter.GaussianBlur(radius=0.5))
    img_array = np.array(img_temp, dtype=np.float32)
    
    # 3. CRT scanlines effect (darken every 2nd line)
    height, width = img_array.shape[:2]
    for y in range(1, height, 2):  # Every 2nd scanline
        img_array[y, :, :] *= 0.7  # Darken by 30%
    
    # Clip values and convert back to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def convert_pngs_to_jpgs(input_folder):
    """Convert all PNG files to JPG at 90% and 15% quality."""
    if not input_folder:
        return "❌ Please select an input folder."
    
    input_path = Path(input_folder)
    
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid folder path."
    
    # Find all PNG files
    png_files = list(input_path.glob("*.png")) + list(input_path.glob("*.PNG"))
    
    if not png_files:
        return "❌ No PNG files found in the selected folder."
    
    # Create output directories
    output_90q = input_path / "jpgs_90Q"
    output_15q = input_path / "jpgs_15Q"
    output_90q.mkdir(exist_ok=True)
    output_15q.mkdir(exist_ok=True)
    
    processed_count = 0
    errors = []
    
    for png_file in png_files:
        try:
            # Open PNG image
            img = Image.open(png_file)
            
            # Convert RGBA to RGB if necessary (JPG doesn't support alpha)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Generate output filename (same name, .jpg extension)
            jpg_filename = png_file.stem + ".jpg"
            
            # Save at 90% quality
            output_90q_path = output_90q / jpg_filename
            img.save(output_90q_path, "JPEG", quality=90, optimize=True)
            
            # Apply distressed effects for 15% quality version
            img_distressed = apply_distressed_effects(img)
            
            # Save at 15% quality with distressed effects
            output_15q_path = output_15q / jpg_filename
            img_distressed.save(output_15q_path, "JPEG", quality=15, optimize=True)
            
            processed_count += 1
            
        except Exception as e:
            errors.append(f"Error processing {png_file.name}: {str(e)}")
    
    # Build result message
    result = f"✓ Successfully processed {processed_count} PNG files\n"
    result += f"  → 90% quality JPGs saved to: {output_90q}\n"
    result += f"  → 15% quality JPGs saved to: {output_15q}\n"
    
    if errors:
        result += f"\n⚠ Errors encountered:\n"
        for error in errors[:10]:  # Limit to first 10 errors
            result += f"  • {error}\n"
        if len(errors) > 10:
            result += f"  ... and {len(errors) - 10} more errors\n"
    
    return result


# Build Gradio interface
with gr.Blocks(title="PNG to JPG Quality Converter") as demo:
    gr.Markdown("# PNG to JPG Quality Converter")
    gr.Markdown("Convert PNG images to JPG at 90% and 15% quality levels")
    
    input_folder = gr.Textbox(
        label="Input Folder (containing PNG files)",
        placeholder="Click 'Browse Folder' to select...",
        interactive=True
    )
    
    browse_btn = gr.Button("📁 Browse Folder", size="sm")
    convert_btn = gr.Button("Convert All PNGs", variant="primary", size="lg")
    
    output_status = gr.Textbox(
        label="Conversion Status",
        lines=10,
        interactive=False
    )
    
    gr.Markdown("""
    ### Instructions:
    1. Enter the path to the folder containing your PNG files
    2. Click "Convert All PNGs"
    3. Two subdirectories will be created:
       - `jpgs_90Q` - High quality (90%) JPG files
       - `jpgs_15Q` - Low quality (15%) JPG files with distressed effects (noise, CRT scanlines, color bleeding)
    
    **Note:** PNG files with transparency will be converted to white background.
    """)
    
    # Wire up the browse button
    browse_btn.click(
        fn=browse_folder,
        inputs=None,
        outputs=input_folder
    )
    
    # Wire up the convert button
    convert_btn.click(
        fn=convert_pngs_to_jpgs,
        inputs=[input_folder],
        outputs=[output_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
