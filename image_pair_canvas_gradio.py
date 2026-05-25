#!/usr/bin/env python3
"""
Image Pair Canvas Creator
Creates side-by-side image pairs on a 2560x1440 canvas with a central 240px gap.
"""

import os
from pathlib import Path
from PIL import Image
import gradio as gr


# Hardcoded directories
INPUT_DIR = "images_general/1040x1464"
OUTPUT_DIR = "images_general/1040x1464/paired"


def get_image_files(folder_path):
    """Get all valid image files from the folder."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in valid_extensions:
            image_files.append(file)
    
    return image_files


def resize_if_needed(image):
    """Resize image to exactly 1440px height, maintaining aspect ratio."""
    width, height = image.size
    
    if height != 1440:
        # Calculate new width to maintain aspect ratio
        new_height = 1440
        new_width = int(width * (new_height / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def create_paired_canvas(img1, img2, output_path, use_blend=True):
    """Create a 2560x1440 canvas with two images side-by-side."""
    # Canvas dimensions
    canvas_width = 2560
    canvas_height = 1440
    central_gap = 240
    
    # Create mid-grey canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (128, 128, 128))
    
    # Resize images if needed
    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)
    
    # Calculate gap boundaries
    gap_start = (canvas_width - central_gap) // 2  # 1160px - where central gap starts
    gap_end = gap_start + central_gap  # 1400px - where central gap ends
    
    img1_width, img1_height = img1.size
    img2_width, img2_height = img2.size
    
    # Position left image tight against the central gap (right edge at gap_start)
    left_x = gap_start - img1_width
    left_y = (canvas_height - img1_height) // 2
    
    # Paste left image
    canvas.paste(img1, (left_x, left_y))
    
    # Position right image to start immediately after the central gap (left edge at gap_end)
    right_x = gap_end
    right_y = (canvas_height - img2_height) // 2
    
    # Paste right image
    canvas.paste(img2, (right_x, right_y))
    
    # Create linear blend across the central gap (if enabled)
    if use_blend:
        canvas_pixels = canvas.load()
        img1_pixels = img1.load()
        img2_pixels = img2.load()
        
        for y in range(canvas_height):
            # Determine if this row has image content
            img1_row_y = y - left_y
            img2_row_y = y - right_y
            
            # Get edge colors (use mid-grey if outside image bounds)
            if 0 <= img1_row_y < img1_height:
                left_color = img1_pixels[img1_width - 1, img1_row_y]
            else:
                left_color = (128, 128, 128)
            
            if 0 <= img2_row_y < img2_height:
                right_color = img2_pixels[0, img2_row_y]
            else:
                right_color = (128, 128, 128)
            
            # Blend across the gap
            for x in range(gap_start, gap_end):
                # Calculate blend factor (0.0 at gap_start, 1.0 at gap_end)
                t = (x - gap_start) / central_gap
                
                # Linear interpolation of RGB values
                r = int(left_color[0] * (1 - t) + right_color[0] * t)
                g = int(left_color[1] * (1 - t) + right_color[1] * t)
                b = int(left_color[2] * (1 - t) + right_color[2] * t)
                
                canvas_pixels[x, y] = (r, g, b)
    
    # Save as PNG
    canvas.save(output_path, 'PNG')
    
    return canvas


def process_image_pairs(use_blend, progress=gr.Progress()):
    """Process all images in the folder and create paired canvases."""
    input_folder = INPUT_DIR
    output_folder = OUTPUT_DIR
    
    if not os.path.exists(input_folder):
        return f"Error: Input folder '{input_folder}' does not exist", None
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(input_folder)
    
    if len(image_files) < 2:
        return f"Error: Found only {len(image_files)} image(s). Need at least 2 images to create pairs.", None
    
    # Create pairs
    pairs_created = 0
    total_pairs = len(image_files) // 2
    preview_image = None
    
    # Convert blend option to boolean
    blend_enabled = (use_blend == "Blend On")
    
    for i in progress.tqdm(range(0, len(image_files) - 1, 2), desc="Creating image pairs"):
        try:
            # Load two images
            img1_path = image_files[i]
            img2_path = image_files[i + 1]
            
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Create output filename
            output_filename = f"pair_{pairs_created + 1:04d}.png"
            output_file = output_path / output_filename
            
            # Create paired canvas with blend option
            canvas = create_paired_canvas(img1, img2, output_file, use_blend=blend_enabled)
            
            pairs_created += 1
            
            # Save first canvas as preview
            if preview_image is None:
                preview_image = canvas
                
        except Exception as e:
            print(f"Error processing {img1_path.name} and {img2_path.name}: {e}")
            continue
    
    # Summary message
    remaining = len(image_files) % 2
    blend_status = "enabled" if blend_enabled else "disabled"
    message = (f"Successfully created {pairs_created} image pairs\n"
               f"Output folder: {output_folder}\n"
               f"Canvas size: 2560x1440 pixels\n"
               f"Central gap: 240 pixels\n"
               f"Blend: {blend_status}")
    
    if remaining > 0:
        message += f"\n{remaining} image(s) remaining (unpaired)"
    
    return message, preview_image


def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="Image Pair Canvas Creator") as demo:
        gr.Markdown("# Image Pair Canvas Creator")
        gr.Markdown("Create side-by-side image pairs on a 2560x1440 canvas with a 240px central gap.")
        gr.Markdown(f"**Input:** `{INPUT_DIR}`  \n**Output:** `{OUTPUT_DIR}`")
        
        with gr.Row():
            with gr.Column():
                blend_option = gr.Radio(
                    choices=["Blend On", "Blend Off"],
                    value="Blend On",
                    label="Central Gap Blending",
                    info="Blend colors across the central gap or leave it mid-grey"
                )
                
                process_btn = gr.Button("Create Image Pairs", variant="primary")
                
                status_text = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )
        
        with gr.Row():
            preview_image = gr.Image(
                label="Preview (First Pair)",
                type="pil"
            )
        
        # Button click handler
        process_btn.click(
            fn=process_image_pairs,
            inputs=[blend_option],
            outputs=[status_text, preview_image]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(inbrowser=True)
