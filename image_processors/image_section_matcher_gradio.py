#!/usr/bin/env python3
"""
Image Section Matcher - Gradio version
Creates new images by matching sections between two source images.
Loads two 512x512 images, divides into sections, and finds best-fit matches using similarity matrices.
"""

import gradio as gr
from PIL import Image
import numpy as np
import random
from datetime import datetime
import os
import threading

# Global flag for stopping optimization
stop_optimization = threading.Event()


def pad_to_square(img, target_size=1024):
    """Pad image with white to 1:1 ratio, then resize to target size."""
    width, height = img.size
    
    # Determine the larger dimension
    max_dim = max(width, height)
    
    # Create a new white square image
    square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    
    # Calculate position to paste original image (centered)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    
    # Convert original to RGB if needed before pasting
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Paste original image onto white square
    square_img.paste(img, (left, top))
    
    # Resize to target size
    if square_img.size != (target_size, target_size):
        square_img = square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return square_img


def extract_sections(image, section_size):
    """Extract all sections from an image."""
    sections = []
    sections_per_side = image.size[0] // section_size
    img_array = np.array(image)
    
    for row in range(sections_per_side):
        for col in range(sections_per_side):
            y_start = row * section_size
            y_end = y_start + section_size
            x_start = col * section_size
            x_end = x_start + section_size
            
            section = img_array[y_start:y_end, x_start:x_end]
            sections.append(section)
            
    return sections, sections_per_side


def calculate_mse(section1, section2):
    """Calculate Mean Squared Error between two sections (lower is better)."""
    return np.mean((section1.astype(np.float32) - section2.astype(np.float32)) ** 2)


def calculate_section_similarity_score(result_section, source_section, similarity_func):
    """Calculate how well a result section matches its corresponding source section."""
    return similarity_func(result_section, source_section)


def infinite_swap_optimization(result_array, source_sections, sections_per_side, section_size):
    """Continuously optimize with random swaps until stopped."""
    total_sections = len(source_sections)
    improvements_made = 0
    swaps_checked = 0
    
    while not stop_optimization.is_set():
        idx1 = random.randint(0, total_sections - 1)
        idx2 = random.randint(0, total_sections - 1)
        
        if idx1 == idx2:
            continue
            
        swaps_checked += 1
        
        # Calculate positions in the grid
        row1, col1 = idx1 // sections_per_side, idx1 % sections_per_side
        row2, col2 = idx2 // sections_per_side, idx2 % sections_per_side
        
        y1_start, y1_end = row1 * section_size, (row1 + 1) * section_size
        x1_start, x1_end = col1 * section_size, (col1 + 1) * section_size
        
        y2_start, y2_end = row2 * section_size, (row2 + 1) * section_size
        x2_start, x2_end = col2 * section_size, (col2 + 1) * section_size
        
        current_section1 = result_array[y1_start:y1_end, x1_start:x1_end]
        current_section2 = result_array[y2_start:y2_end, x2_start:x2_end]
        
        # Calculate current similarity scores using MSE
        current_score1 = calculate_section_similarity_score(
            current_section1, source_sections[idx1], calculate_mse)
        current_score2 = calculate_section_similarity_score(
            current_section2, source_sections[idx2], calculate_mse)
        current_total_score = current_score1 + current_score2
        
        # Calculate scores if we swap the sections
        swapped_score1 = calculate_section_similarity_score(
            current_section2, source_sections[idx1], calculate_mse)
        swapped_score2 = calculate_section_similarity_score(
            current_section1, source_sections[idx2], calculate_mse)
        swapped_total_score = swapped_score1 + swapped_score2
        
        # If swapping improves the total score, do the swap
        if swapped_total_score < current_total_score:
            temp_section = current_section1.copy()
            result_array[y1_start:y1_end, x1_start:x1_end] = current_section2
            result_array[y2_start:y2_end, x2_start:x2_end] = temp_section
            improvements_made += 1
            
            # Yield progress every 200 improvements
            if improvements_made % 200 == 0:
                yield improvements_made, swaps_checked, result_array.copy()
    
    # Final yield when stopped
    yield improvements_made, swaps_checked, result_array.copy()


def generate_matched_image(source_image, target_image, section_size):
    """Generate the matched image using random assignment and continuous swapping."""
    global stop_optimization
    stop_optimization.clear()
    
    if source_image is None or target_image is None:
        yield None, "Please upload both source and target images."
        return
    
    try:
        # Prepare images (pad to square with white)
        source_img = pad_to_square(source_image, 1024)
        target_img = pad_to_square(target_image, 1024)
        
        # Extract sections
        source_sections, sections_per_side = extract_sections(source_img, section_size)
        target_sections, _ = extract_sections(target_img, section_size)
        
        # Create result image with RANDOM assignment
        result_array = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        total_sections = len(source_sections)
        
        # Create random assignment (shuffle target sections)
        target_indices = list(range(len(target_sections)))
        random.shuffle(target_indices)
        
        # Assign randomly shuffled target sections to source positions
        for source_idx in range(total_sections):
            target_idx = target_indices[source_idx]
            
            row = source_idx // sections_per_side
            col = source_idx % sections_per_side
            
            y_start = row * section_size
            y_end = y_start + section_size
            x_start = col * section_size
            x_end = x_start + section_size
            
            result_array[y_start:y_end, x_start:x_end] = target_sections[target_idx]
        
        # Show initial random assignment
        initial_image = Image.fromarray(result_array)
        yield initial_image, "Starting optimization with random assignment..."
        
        # Perform continuous swap optimization
        for improvements_made, swaps_checked, updated_array in infinite_swap_optimization(
            result_array, source_sections, sections_per_side, section_size):
            
            result_image = Image.fromarray(updated_array)
            status = f"Swaps checked: {swaps_checked:,}\nImprovements made: {improvements_made:,}"
            yield result_image, status
        
    except Exception as e:
        yield None, f"Stopped\nError: {str(e)}"


def stop_optimization_handler():
    """Stop the optimization process."""
    global stop_optimization
    stop_optimization.set()
    return "Stopping optimization..."


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Image Section Matcher") as demo:
        gr.Markdown("# Image Section Matcher")
        gr.Markdown("Creates new images by matching sections between two source images.")
        
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(
                    label="Source Image (sections to match)",
                    type="pil",
                    height=400
                )
                
                target_image = gr.Image(
                    label="Target Image (sections to choose from)",
                    type="pil",
                    height=400
                )
                
                with gr.Row():
                    section_size = gr.Radio(
                        choices=[8, 16, 32],
                        value=16,
                        label="Section Size (pixels)",
                        info="Size of each matching section"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button("Start Optimization", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")
            
            with gr.Column():
                result_image = gr.Image(
                    label="Matched Result",
                    type="pil",
                    height=600
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=4,
                    max_lines=10
                )
                
                stop_status = gr.Textbox(
                    label="Control",
                    lines=1,
                    visible=True
                )
        
        gr.Markdown("""
        ### How it works:
        - Divides both images into sections of the selected size
        - Randomly assigns target sections to source positions
        - Continuously performs random swaps to improve match quality
        - Each swap is kept only if it improves the overall similarity (using MSE)
        - Progress updates shown every 200 improvements
        - Click "Stop" in Gradio to end the optimization
        """)
        
        # Set up the generation action
        generate_btn.click(
            fn=generate_matched_image,
            inputs=[source_image, target_image, section_size],
            outputs=[result_image, status_output]
        )
        
        # Set up the stop button
        stop_btn.click(
            fn=stop_optimization_handler,
            inputs=[],
            outputs=[stop_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)
