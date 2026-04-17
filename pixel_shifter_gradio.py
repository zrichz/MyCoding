import cv2
import numpy as np
import gradio as gr
import random

# Global state to store current image with original dtype
current_image = None
original_image = None

def shift_pixels(image, iterations, shift_amount):
    """
    Randomly shift columns vertically and rows horizontally with wraparound.
    Preserves exact pixel values without any lossy conversion.
    """
    if image is None:
        return None
    
    # Ensure we preserve the exact dtype
    result = image.copy()
    h, w = result.shape[:2]
    
    for i in range(iterations):
        # Pick random column and shift direction
        col_idx = random.randint(0, w - 1)
        col_shift = random.choice([-shift_amount, shift_amount])
        
        # Shift column vertically (wraparound) - affects all color channels
        result[:, col_idx, :] = np.roll(result[:, col_idx, :], col_shift, axis=0)
        
        # Pick random row and shift direction
        row_idx = random.randint(0, h - 1)
        row_shift = random.choice([-shift_amount, shift_amount])
        
        # Shift row horizontally (wraparound) - affects all color channels
        result[row_idx, :, :] = np.roll(result[row_idx, :, :], row_shift, axis=0)
    
    return result

def process_image(img, iterations, shift_amount):
    """Process image with the specified number of iterations. Lossless processing."""
    global current_image, original_image
    
    if img is None:
        return None
    
    # Convert input to uint8 if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # Check if this is a new image (different from original)
    if original_image is None or not np.array_equal(img, original_image):
        # New image loaded - reset state
        original_image = img.copy()
        current_image = img.copy()
    
    # Process the current image
    current_image = shift_pixels(current_image, iterations, shift_amount)
    return current_image if current_image is not None else None

def reset_image(img):
    """Reset to the original input image."""
    global current_image, original_image
    if img is not None:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        original_image = img.copy()
        current_image = img.copy()
    else:
        current_image = None
        original_image = None
    return current_image

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Pixel Shifter\nRandomly shift columns and rows with toroidal wraparound.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Image")
            iterations_slider = gr.Slider(
                minimum=1, 
                maximum=10000, 
                step=1, 
                value=100, 
                label="Iterations"
            )
            shift_amount_slider = gr.Slider(
                minimum=1,
                maximum=24,
                step=1,
                value=1,
                label="Number of Pixels to Shift"
            )
            with gr.Row():
                process_btn = gr.Button("Process", variant="primary")
                reset_btn = gr.Button("Reset to Original", variant="secondary")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Output Image")
            gr.Markdown("Right-click → Save Image As... to save as PNG\n\n**Note:** Each process is cumulative - it builds on the previous result.")
    
    process_btn.click(
        process_image, 
        inputs=[input_image, iterations_slider, shift_amount_slider], 
        outputs=output_image
    )
    
    reset_btn.click(
        reset_image,
        inputs=input_image,
        outputs=output_image
    )

demo.launch(inbrowser=True)
