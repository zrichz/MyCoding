#!/usr/bin/env python3
"""
Portrait to Landscape Expander with Blurred Edges
Expands portrait images to 2560x1440 landscape format by scaling the image to center
and filling side spaces with stretched rectangles from the image edges using bilinear scaling.
"""

import numpy as np
from PIL import Image
import gradio as gr
from scipy.ndimage import gaussian_filter


def extract_and_resize_edge(img, side='left', rect_width=50, rect_height=300, target_width=640, target_height=1440):
    """
    Extract a vertical rectangle from the edge of an image and resize it to fill a target area.
    
    Args:
        img: PIL Image
        side: 'left' or 'right'
        rect_width: width of the rectangle to extract
        rect_height: height of the rectangle to extract
        target_width: target width to resize to
        target_height: target height to resize to
    
    Returns:
        Resized PIL Image
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Calculate vertical center position
    start_y = max(0, (h - rect_height) // 2)
    end_y = min(h, start_y + rect_height)
    
    if side == 'left':
        # Extract from left edge
        start_x = 0
        end_x = min(w, rect_width)
    else:  # right
        # Extract from right edge
        start_x = max(0, w - rect_width)
        end_x = w
    
    # Extract rectangle
    rect_array = img_array[start_y:end_y, start_x:end_x]
    rect_img = Image.fromarray(rect_array)
    
    # Resize with bilinear interpolation
    resized = rect_img.resize((target_width, target_height), Image.BILINEAR)
    
    return resized


def expand_portrait_to_landscape(input_image, rect_width=50, rect_height=300, blur_kernel=3):
    """
    Main processing function to expand portrait image to landscape with blurred edges.
    
    Args:
        input_image: PIL Image or numpy array
        rect_width: width of rectangle to extract from image edges
        rect_height: height of rectangle to extract from image edges
        blur_kernel: Gaussian blur kernel size (odd number)
    
    Returns:
        Expanded image as PIL Image
    """
    if input_image is None:
        return None, "No image provided"
    
    # Convert to PIL if needed
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    else:
        img = input_image.convert('RGB')
    
    # Target canvas dimensions
    canvas_width = 2560
    canvas_height = 1440
    
    # Calculate scaled dimensions (fit to 1440 height)
    original_width, original_height = img.size
    scale_factor = canvas_height / original_height
    scaled_width = int(original_width * scale_factor)
    scaled_height = canvas_height
    
    # Resize with Lanczos
    try:
        scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    except AttributeError:
        # Fallback for older Pillow versions
        scaled_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
    
    # Create black canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    
    # Calculate position to center the scaled image
    paste_x = (canvas_width - scaled_width) // 2
    canvas.paste(scaled_img, (paste_x, 0))
    
    # Calculate side panel dimensions
    side_width = paste_x
    
    if side_width > 0:  # Only process if there are side panels
        # Process LEFT side
        left_panel = extract_and_resize_edge(
            img, side='left', 
            rect_width=rect_width, 
            rect_height=rect_height,
            target_width=side_width,
            target_height=canvas_height
        )
        
        # Apply Gaussian blur if kernel size > 1
        if blur_kernel > 1:
            left_array = np.array(left_panel)
            sigma = (blur_kernel - 1) / 6.0
            left_blurred = np.zeros_like(left_array)
            for i in range(3):  # Apply to each color channel
                left_blurred[:, :, i] = gaussian_filter(left_array[:, :, i], sigma=sigma)
            left_panel = Image.fromarray(left_blurred.astype(np.uint8))
        
        canvas.paste(left_panel, (0, 0))
        
        # Process RIGHT side
        right_panel = extract_and_resize_edge(
            img, side='right',
            rect_width=rect_width,
            rect_height=rect_height,
            target_width=side_width,
            target_height=canvas_height
        )
        
        # Apply Gaussian blur if kernel size > 1
        if blur_kernel > 1:
            right_array = np.array(right_panel)
            sigma = (blur_kernel - 1) / 6.0
            right_blurred = np.zeros_like(right_array)
            for i in range(3):  # Apply to each color channel
                right_blurred[:, :, i] = gaussian_filter(right_array[:, :, i], sigma=sigma)
            right_panel = Image.fromarray(right_blurred.astype(np.uint8))
        
        canvas.paste(right_panel, (canvas_width - side_width, 0))
    
    # Generate status message
    message = (f"Successfully expanded image to 2560x1440\n"
              f"Original size: {original_width}x{original_height}\n"
              f"Scaled size: {scaled_width}x{scaled_height}\n"
              f"Side panel width: {side_width} pixels each\n"
              f"Source rectangle: {rect_width}x{rect_height} pixels\n"
              f"Blur kernel size: {blur_kernel}")
    
    return canvas, message


# Create Gradio interface
with gr.Blocks(title="Portrait to Landscape Expander") as demo:
    gr.Markdown("""
    # Portrait to Landscape Expander with Blurred Edges
    
    Upload a portrait-oriented image to expand it to 2560x1440 landscape format.
    The image is scaled to fit 1440px height and centered, with the side spaces 
    filled using stretched rectangles from the image edges for a blurred effect.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=800)
            
            with gr.Row():
                rect_width_slider = gr.Slider(
                    minimum=5, maximum=1440, value=300, step=5,
                    label="Source Rectangle Width",
                    info="Width of rectangle extracted from image edges"
                )
                rect_height_slider = gr.Slider(
                    minimum=5, maximum=1440, value=1000, step=5,
                    label="Source Rectangle Height",
                    info="Height of rectangle extracted from image edges"
                )
            
            blur_kernel_slider = gr.Slider(
                minimum=3, maximum=501, value=351, step=2,
                label="Gaussian Blur Kernel Size",
                info="Blur applied to side panels (odd numbers only)"
            )
            
            process_btn = gr.Button("Expand Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Expanded Image (2560x1440)", height=800, format="png")
            output_message = gr.Textbox(label="Processing Info", lines=6)
    
    # Set up the processing
    process_btn.click(
        fn=expand_portrait_to_landscape,
        inputs=[input_image, rect_width_slider, rect_height_slider, blur_kernel_slider],
        outputs=[output_image, output_message]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
