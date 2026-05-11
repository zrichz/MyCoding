#!/usr/bin/env python3
"""
Spiral Pixel Remapper Gradio App
Extracts pixels from center spiraling outward, then places them from top-left spiraling inward.
Uses Ulam-style spiral pattern with cardinal directions.
"""

import numpy as np
import gradio as gr
from PIL import Image


def generate_outward_spiral(width, height):
    """
    Generate coordinates for a spiral starting from center and moving outward.
    Returns list of (x, y) coordinates in spiral order.
    """
    center_x = width // 2
    center_y = height // 2
    
    coords = [(center_x, center_y)]
    
    x, y = center_x, center_y
    step = 1
    
    # Directions: right, down, left, up
    dx_list = [1, 0, -1, 0]
    dy_list = [0, 1, 0, -1]
    
    direction = 0  # Start moving right
    steps_in_direction = 0
    steps_before_turn = 1
    turns_at_current_length = 0
    
    max_coords = width * height
    
    while len(coords) < max_coords:
        # Move in current direction
        dx = dx_list[direction]
        dy = dy_list[direction]
        
        x += dx
        y += dy
        
        if 0 <= x < width and 0 <= y < height:
            coords.append((x, y))
        
        steps_in_direction += 1
        
        # Check if we need to turn
        if steps_in_direction >= steps_before_turn:
            steps_in_direction = 0
            direction = (direction + 1) % 4
            turns_at_current_length += 1
            
            # After 2 turns, increase the step length
            if turns_at_current_length >= 2:
                turns_at_current_length = 0
                steps_before_turn += 1
        
        # Safety check to prevent infinite loop
        if abs(x) > width * 2 or abs(y) > height * 2:
            break
    
    return coords[:max_coords]


def generate_inward_spiral(width, height):
    """
    Generate coordinates for a spiral starting from top-left and moving inward.
    Returns list of (x, y) coordinates in spiral order ending at center.
    """
    coords = []
    
    left, right = 0, width - 1
    top, bottom = 0, height - 1
    
    while left <= right and top <= bottom:
        # Move right along top edge
        for x in range(left, right + 1):
            if len(coords) < width * height:
                coords.append((x, top))
        top += 1
        
        # Move down along right edge
        for y in range(top, bottom + 1):
            if len(coords) < width * height:
                coords.append((right, y))
        right -= 1
        
        # Move left along bottom edge
        if top <= bottom:
            for x in range(right, left - 1, -1):
                if len(coords) < width * height:
                    coords.append((x, bottom))
            bottom -= 1
        
        # Move up along left edge
        if left <= right:
            for y in range(bottom, top - 1, -1):
                if len(coords) < width * height:
                    coords.append((left, y))
            left += 1
    
    return coords


def spiral_remap(image):
    """
    Remap pixels by reading from center-outward spiral and writing to corner-inward spiral.
    """
    if image is None:
        return None
    
    # Convert to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    height, width = image.shape[:2]
    
    # Generate spiral patterns
    read_coords = generate_outward_spiral(width, height)
    write_coords = generate_inward_spiral(width, height)
    
    # Ensure we have the same number of coordinates
    num_pixels = min(len(read_coords), len(write_coords), width * height)
    read_coords = read_coords[:num_pixels]
    write_coords = write_coords[:num_pixels]
    
    # Create output image
    output = np.zeros_like(image)
    
    # Remap pixels
    for i in range(num_pixels):
        read_x, read_y = read_coords[i]
        write_x, write_y = write_coords[i]
        
        if (0 <= read_x < width and 0 <= read_y < height and
            0 <= write_x < width and 0 <= write_y < height):
            output[write_y, write_x] = image[read_y, read_x]
    
    return output


def process_image(image):
    """
    Wrapper function for Gradio interface.
    """
    if image is None:
        return None
    
    result = spiral_remap(image)
    return result


# Create Gradio interface
with gr.Blocks(title="Spiral Pixel Remapper") as demo:
    gr.Markdown("""
    # Spiral Pixel Remapper
    
    This app remaps image pixels using spiral patterns:
    - Reads pixels starting from center and spiraling outward (Ulam-style)
    - Writes pixels starting from top-left corner and spiraling inward
    - Creates unique spatial transformations
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Image")
            input_image = gr.Image(label="Upload Image", type="numpy")
            process_btn = gr.Button("Apply Spiral Remap", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Output Image")
            output_image = gr.Image(label="Remapped Result", type="numpy")
    
    gr.Markdown("""
    ### How it works:
    1. Extracts pixels from the input image starting at the center pixel
    2. Spirals outward in cardinal directions (right, down, left, up)
    3. Places extracted pixels in output starting at top-left corner
    4. Spirals inward toward the center of the output image
    """)
    
    # Connect the process button
    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=output_image
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
