#!/usr/bin/env python3
"""
Gradio app for applying logo watermarks to a directory of images.
Supports alpha channels and maintains aspect ratio.
"""

import gradio as gr
from PIL import Image
import os
from pathlib import Path
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog


def browse_directory():
    """Open a file dialog to select a directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.wm_attributes('-topmost', 1)  # Bring dialog to front
    directory = filedialog.askdirectory(title="Select Directory with Images")
    root.destroy()
    return directory if directory else None


def resize_logo(logo_image, max_size=48):
    """
    Resize logo to fit within max_size x max_size while maintaining aspect ratio.
    
    Args:
        logo_image: PIL Image object
        max_size: Maximum width or height in pixels
    
    Returns:
        Resized PIL Image object
    """
    width, height = logo_image.size
    
    # Calculate scaling factor
    scale = min(max_size / width, max_size / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return logo_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return logo_image


def apply_watermark(image_path, logo_image, position='bottom-right', padding=10, opacity=1.0):
    """
    Apply watermark logo to an image.
    
    Args:
        image_path: Path to the image file
        logo_image: PIL Image object of the logo (already resized)
        position: Position of the watermark (currently only bottom-right)
        padding: Padding from edges in pixels
        opacity: Logo opacity from 0.0 to 1.0
    
    Returns:
        PIL Image object with watermark applied
    """
    # Open the base image
    base_image = Image.open(image_path).convert('RGBA')
    
    # Ensure logo is in RGBA mode to handle transparency
    if logo_image.mode != 'RGBA':
        logo_image = logo_image.convert('RGBA')
    
    # Apply opacity to the logo
    if opacity < 1.0:
        # Create a copy to avoid modifying the original
        logo_with_opacity = logo_image.copy()
        # Get the alpha channel
        alpha = logo_with_opacity.split()[3]
        # Multiply alpha by opacity
        alpha = alpha.point(lambda p: int(p * opacity))
        # Put the modified alpha back
        logo_with_opacity.putalpha(alpha)
        logo_image = logo_with_opacity
    
    # Calculate position (bottom-right)
    base_width, base_height = base_image.size
    logo_width, logo_height = logo_image.size
    
    x = base_width - logo_width - padding
    y = base_height - logo_height - padding
    
    # Create a transparent layer for compositing
    transparent_layer = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    transparent_layer.paste(logo_image, (x, y), logo_image)
    
    # Composite the watermark onto the base image
    watermarked = Image.alpha_composite(base_image, transparent_layer)
    
    return watermarked


def process_directory(logo_file, input_directory, max_logo_size, padding, opacity):
    """
    Process all images in a directory and apply watermark.
    
    Args:
        logo_file: Path to logo file
        input_directory: Directory containing images to watermark
        max_logo_size: Maximum width/height for logo
        padding: Padding from edges
        opacity: Logo opacity (0.5 to 1.0)
    
    Returns:
        Tuple of (status message, list of output file paths)
    """
    if not logo_file:
        return "Error: Please upload a logo image.", []
    
    if not input_directory or not os.path.isdir(input_directory):
        return "Error: Please provide a valid input directory.", []
    
    # Create output directory as a subfolder
    output_directory = os.path.join(input_directory, "watermarked")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Load and resize logo
    try:
        logo = Image.open(logo_file)
        logo_resized = resize_logo(logo, max_size=max_logo_size)
    except Exception as e:
        return f"Error loading logo: {str(e)}", []
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if Path(file).suffix.lower() in supported_extensions:
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        return f"No images found in {input_directory}", []
    
    # Process images
    processed_count = 0
    output_files = []
    errors = []
    
    for image_path in image_files:
        try:
            # Apply watermark with opacity
            watermarked = apply_watermark(image_path, logo_resized, padding=padding, opacity=opacity)
            
            # Generate output filename
            rel_path = os.path.relpath(image_path, input_directory)
            output_path = os.path.join(output_directory, rel_path)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert back to RGB if saving as JPEG
            if Path(output_path).suffix.lower() in ['.jpg', '.jpeg']:
                watermarked = watermarked.convert('RGB')
            
            # Save the watermarked image
            watermarked.save(output_path, quality=95)
            
            processed_count += 1
            output_files.append(output_path)
            
        except Exception as e:
            errors.append(f"Error processing {image_path}: {str(e)}")
    
    # Prepare status message
    status = f"✅ Successfully processed {processed_count} of {len(image_files)} images.\n"
    status += f"📁 Output saved to: {output_directory}\n"
    
    if errors:
        status += f"\n⚠️ Errors encountered:\n" + "\n".join(errors[:5])
        if len(errors) > 5:
            status += f"\n... and {len(errors) - 5} more errors."
    
    return status


def preview_watermark(logo_file, input_directory, max_logo_size, padding, opacity):
    """
    Generate a preview of the watermark on the first image found in the directory.
    
    Args:
        logo_file: Path to logo file
        input_directory: Directory containing images
        max_logo_size: Maximum width/height for logo
        padding: Padding from edges
        opacity: Logo opacity (0.5 to 1.0)
    
    Returns:
        PIL Image with watermark applied for preview
    """
    if not logo_file:
        return None
    
    if not input_directory or not os.path.isdir(input_directory):
        return None
    
    try:
        # Load and resize logo
        logo = Image.open(logo_file)
        logo_resized = resize_logo(logo, max_size=max_logo_size)
        
        # Find first image in directory
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    first_image = os.path.join(root, file)
                    
                    # Apply watermark for preview
                    watermarked = apply_watermark(first_image, logo_resized, padding=padding, opacity=opacity)
                    
                    # Convert to RGB for display
                    if watermarked.mode == 'RGBA':
                        # Create white background
                        background = Image.new('RGB', watermarked.size, (255, 255, 255))
                        background.paste(watermarked, mask=watermarked.split()[3])
                        return background
                    
                    return watermarked.convert('RGB')
        
        # No images found
        return None
        
    except Exception as e:
        print(f"Preview error: {e}")
        return None


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Image Watermark Tool") as demo:
        gr.Markdown("# Image Watermark Tool")
        gr.Markdown("Apply a logo watermark to a directory of images. Preview shows 1:1 scale with logo at bottom-right corner.")
        
        with gr.Row():
            with gr.Column(scale=2):
                logo_input = gr.File(
                    label="Upload Logo",
                    file_types=["image"],
                    type="filepath"
                )
                
                with gr.Row():
                    input_dir = gr.Textbox(
                        label="Input Directory",
                        placeholder="/path/to/input/images",
                        info="Output will be saved in a 'watermarked' subfolder",
                        scale=4
                    )
                    browse_btn = gr.Button("Browse", scale=1, size="sm")
                
                with gr.Row():
                    max_size = gr.Slider(
                        minimum=16,
                        maximum=256,
                        value=48,
                        step=8,
                        label="Maximum Logo Size (pixels)",
                        info="Logo will be resized to fit within this dimension"
                    )
                    
                    padding = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=5,
                        label="Padding (pixels)",
                        info="Distance from bottom-right corner"
                    )
                
                opacity = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="Logo Opacity",
                    info="1.0 = 100% (fully opaque), 0.5 = 50% (semi-transparent)"
                )
                
                with gr.Row():
                    preview_btn = gr.Button("🔍 Preview First Image", variant="secondary", size="lg")
                    process_btn = gr.Button("✅ Process All Images", variant="primary", size="lg")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    max_lines=20
                )
            
            with gr.Column(scale=3):
                preview_image = gr.Image(
                    label="Preview (1:1 scale - Logo position shown at bottom-right)",
                    type="pil",
                    show_label=True,
                    interactive=False,
                    container=True,
                    elem_classes=["preview-container"]
                )
        
        # Set up browse button
        browse_btn.click(
            fn=browse_directory,
            inputs=[],
            outputs=[input_dir]
        )
        
        # Set up preview button to update on any parameter change
        preview_btn.click(
            fn=preview_watermark,
            inputs=[logo_input, input_dir, max_size, padding, opacity],
            outputs=[preview_image]
        )
        
        # Auto-update preview when parameters change
        for component in [logo_input, input_dir, max_size, padding, opacity]:
            component.change(
                fn=preview_watermark,
                inputs=[logo_input, input_dir, max_size, padding, opacity],
                outputs=[preview_image]
            )
        
        # Set up the processing action
        process_btn.click(
            fn=process_directory,
            inputs=[logo_input, input_dir, max_size, padding, opacity],
            outputs=[status_output]
        )
    
    return demo


if __name__ == "__main__":
    # Custom CSS for large preview display (moved to launch for Gradio 6.0)
    custom_css = """
    .preview-container {
        height: calc(100vh - 200px) !important;
        overflow: auto !important;
    }
    .preview-container img {
        max-width: none !important;
        width: auto !important;
        height: auto !important;
    }
    """
    
    demo = create_interface()
    demo.launch(share=False, inbrowser=True, css=custom_css)
