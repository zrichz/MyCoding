#!/usr/bin/env python3
"""
Pixel Spreader - Gradio App
Spreads pixels from image 1 evenly throughout image 2
"""

import gradio as gr
import numpy as np
from PIL import Image


def spread_pixels(image1, image2):
    """
    Spread pixels from image1 evenly throughout image2.
    Uses integer spacing based on total pixel count.
    
    Args:
        image1: Source image (smaller, typically)
        image2: Target image (larger, typically)
    
    Returns:
        Modified image2 with pixels from image1 spread evenly
    """
    if image1 is None or image2 is None:
        return None
    
    # Convert to numpy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2).copy()
    
    h1, w1 = img1_array.shape[:2]
    h2, w2 = img2_array.shape[:2]
    
    # Calculate total pixels
    total_pixels_img1 = h1 * w1
    total_pixels_img2 = h2 * w2
    
    # Calculate integer spacing based on total pixels
    spacing = total_pixels_img2 // total_pixels_img1
    
    # Spread pixels from image1 into image2
    pixel_count = 0
    for i in range(h1):
        for j in range(w1):
            # Calculate target position in linear space
            target_linear = pixel_count * spacing
            
            # Convert linear position to 2D coordinates in image2
            target_y = target_linear // w2
            target_x = target_linear % w2
            
            # Ensure we don't go out of bounds
            if target_y < h2 and target_x < w2:
                img2_array[target_y, target_x] = img1_array[i, j]
            
            pixel_count += 1
    
    result = Image.fromarray(img2_array)
    
    # Create info message
    message = (f"Spread {w1}x{h1} ({total_pixels_img1} pixels) from Image 1 into {w2}x{h2} ({total_pixels_img2} pixels) Image 2\n"
               f"Integer spacing: Every {spacing} pixels")
    
    return result, message


# Create Gradio interface
with gr.Blocks(title="Pixel Spreader") as demo:
    gr.Markdown("# Pixel Spreader")
    gr.Markdown("Spread pixels from Image 1 evenly throughout Image 2")
    
    with gr.Row():
        with gr.Column():
            image1_input = gr.Image(label="Image 1 (Source)", type="pil")
            image2_input = gr.Image(label="Image 2 (Target)", type="pil")
            process_btn = gr.Button("Spread Pixels", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Result", type="pil")
            info_text = gr.Textbox(label="Info", lines=3)
    
    process_btn.click(
        fn=spread_pixels,
        inputs=[image1_input, image2_input],
        outputs=[output_image, info_text]
    )
    
    gr.Markdown("### Usage")
    gr.Markdown("1. Upload Image 1 (source pixels)")
    gr.Markdown("2. Upload Image 2 (target canvas)")
    gr.Markdown("3. Click 'Spread Pixels' to distribute Image 1 pixels evenly across Image 2")


if __name__ == "__main__":
    demo.launch(inbrowser=True)
