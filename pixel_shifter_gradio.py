import cv2
import numpy as np
import gradio as gr
import random

def shift_pixels(image, iterations):
    """
    Randomly shift columns vertically and rows horizontally with wraparound.
    """
    if image is None:
        return None
    
    # Work with a copy
    result = image.copy()
    h, w = result.shape[:2]
    
    for i in range(iterations):
        # Pick random column and shift direction
        col_idx = random.randint(0, w - 1)
        col_shift = random.choice([-1, 1])
        
        # Shift column vertically (wraparound)
        result[:, col_idx] = np.roll(result[:, col_idx], col_shift, axis=0)
        
        # Pick random row and shift direction
        row_idx = random.randint(0, h - 1)
        row_shift = random.choice([-1, 1])
        
        # Shift row horizontally (wraparound)
        result[row_idx, :] = np.roll(result[row_idx, :], row_shift, axis=1)
    
    return result

def process_image(img, iterations):
    """Process image with the specified number of iterations."""
    if img is None:
        return None
    return shift_pixels(img, iterations)

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌀 Pixel Shifter\nRandomly shift columns and rows with toroidal wraparound.")
    
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
            process_btn = gr.Button("Process", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Output Image")
            gr.Markdown("Right-click → Save Image As... to save as PNG")
    
    process_btn.click(
        process_image, 
        inputs=[input_image, iterations_slider], 
        outputs=output_image
    )

demo.launch(inbrowser=True)
