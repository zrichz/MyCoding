#!/home/rich/MyCoding/venvmycoding313/bin/python
"""
Quadtree Image Shuffler
Recursively divides an image into 4x4 grids and shuffles the pieces at each level.
"""

import gradio as gr
import numpy as np
from PIL import Image
import random
import os
from datetime import datetime


def divide_into_4x4(image_array):
    """
    Divide an image array into a grid of sub-arrays.
    Uses 4x4 grid for larger images, 2x2 grid for smaller ones.
    Returns a list of image arrays and their positions.
    """
    h, w = image_array.shape[:2]
    
    # For images smaller than 4x4, use a 2x2 grid
    if h < 4 or w < 4:
        grid_size = 2
    else:
        grid_size = 4
    
    # Calculate the size of each cell
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    pieces = []
    positions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_h
            x_start = j * cell_w
            
            # For the last row/column, extend to the edge to handle non-divisible sizes
            y_end = (i + 1) * cell_h if i < grid_size - 1 else h
            x_end = (j + 1) * cell_w if j < grid_size - 1 else w
            
            piece = image_array[y_start:y_end, x_start:x_end].copy()
            pieces.append(piece)
            positions.append((i, j, y_start, y_end, x_start, x_end))
    
    return pieces, positions


def shuffle_pieces(pieces, positions, seed):
    """
    Shuffle the pieces using the given seed.
    Returns shuffled pieces with their ORIGINAL positions (so pieces move to new locations).
    """
    rng = random.Random(seed)
    
    # Create indices and shuffle them
    indices = list(range(len(pieces)))
    rng.shuffle(indices)
    
    # Reorder pieces according to shuffled indices, but keep original positions
    # This means shuffled_pieces[0] goes to positions[0], etc.
    shuffled_pieces = [pieces[i] for i in indices]
    
    return shuffled_pieces, positions


def reassemble_image(pieces, positions, original_shape):
    """
    Reassemble pieces back into a full image using their position information.
    Resizes pieces if needed to fit their destination positions.
    """
    h, w = original_shape[:2]
    channels = original_shape[2] if len(original_shape) == 3 else 1
    
    if channels == 1:
        result = np.zeros((h, w), dtype=pieces[0].dtype)
    else:
        result = np.zeros((h, w, channels), dtype=pieces[0].dtype)
    
    for piece, (i, j, y_start, y_end, x_start, x_end) in zip(pieces, positions):
        target_h = y_end - y_start
        target_w = x_end - x_start
        piece_h, piece_w = piece.shape[:2]
        
        # Resize piece if it doesn't match the target size
        if piece_h != target_h or piece_w != target_w:
            piece_pil = Image.fromarray(piece)
            piece_pil = piece_pil.resize((target_w, target_h), Image.NEAREST)
            piece = np.array(piece_pil)
        
        result[y_start:y_end, x_start:x_end] = piece
    
    return result


def can_subdivide(image_array):
    """
    Check if an image can be subdivided.
    Requires at least 2x2 pixels to subdivide into 4 pieces.
    """
    h, w = image_array.shape[:2]
    return h >= 2 and w >= 2


def recursive_shuffle(image_array, seed, level=0, save_steps=False, steps_list=None):
    """
    Recursively shuffle an image using 4x4 grid subdivision.
    
    Args:
        image_array: numpy array of the image
        seed: random seed for shuffling
        level: current recursion level (for tracking)
        save_steps: whether to save intermediate steps
        steps_list: list to accumulate intermediate images
    
    Returns:
        Shuffled image array
    """
    if steps_list is None:
        steps_list = []
    
    # Check if we can subdivide
    if not can_subdivide(image_array):
        return image_array
    
    # Divide into 4x4 grid
    pieces, positions = divide_into_4x4(image_array)
    
    # Shuffle the pieces
    shuffled_pieces, original_positions = shuffle_pieces(pieces, positions, seed)
    
    # Recursively process each piece
    processed_pieces = []
    for piece in shuffled_pieces:
        if can_subdivide(piece):
            processed_piece = recursive_shuffle(piece, seed, level + 1, save_steps, steps_list)
        else:
            processed_piece = piece
        processed_pieces.append(processed_piece)
    
    # Reassemble the image
    result = reassemble_image(processed_pieces, original_positions, image_array.shape)
    
    # Save this step if requested
    if save_steps and level == 0:
        # Only save the final result at the top level
        # We'll capture intermediate steps differently
        pass
    
    return result


def recursive_shuffle_with_steps(image_array, seed):
    """
    Perform recursive shuffling while capturing the full image at each level.
    Uses a breadth-first approach to process all pieces at each level before going deeper.
    
    Returns:
        List of image_arrays showing the full image at each recursion level
    """
    steps = []
    
    # Step 0: Original image
    steps.append(image_array.copy())
    
    # Current state of the full image
    current_image = image_array.copy()
    
    # Track regions that need processing at each level
    # Each entry is (y_start, y_end, x_start, x_end) - bounds in the full image
    current_level_regions = [(0, current_image.shape[0], 0, current_image.shape[1])]
    
    level = 0
    while current_level_regions:
        level += 1
        next_level_regions = []
        
        # Create a new image for this level
        level_image = current_image.copy()
        
        # Process all regions at current level
        for y_start, y_end, x_start, x_end in current_level_regions:
            # Extract this region from current image
            region = current_image[y_start:y_end, x_start:x_end].copy()
            
            if not can_subdivide(region):
                continue
            
            # Divide this region into 4x4
            sub_pieces, sub_positions = divide_into_4x4(region)
            
            # Shuffle the sub-pieces (positions stay in original order)
            shuffled_pieces, original_positions = shuffle_pieces(sub_pieces, sub_positions, seed)
            
            # Reassemble the shuffled pieces into original grid positions
            shuffled_region = reassemble_image(shuffled_pieces, original_positions, region.shape)
            
            # Update this region in the level image
            level_image[y_start:y_end, x_start:x_end] = shuffled_region
            
            # Add each sub-piece region to next level for further processing
            for sub_piece, (i, j, sub_y_start, sub_y_end, sub_x_start, sub_x_end) in zip(shuffled_pieces, original_positions):
                # Calculate absolute position in full image
                abs_y_start = y_start + sub_y_start
                abs_y_end = y_start + sub_y_end
                abs_x_start = x_start + sub_x_start
                abs_x_end = x_start + sub_x_end
                
                if can_subdivide(sub_piece):
                    next_level_regions.append((abs_y_start, abs_y_end, abs_x_start, abs_x_end))
        
        # Update current image to the new shuffled state
        current_image = level_image.copy()
        
        # Save this level if there were any subdivisions
        if next_level_regions or level == 1:
            steps.append(current_image.copy())
        
        current_level_regions = next_level_regions
    
    return steps


def process_image(image, seed, save_steps, image_name="uploaded_image"):
    """
    Main processing function for Gradio interface.
    
    Args:
        image: PIL Image or numpy array
        seed: integer seed for random shuffling
        save_steps: boolean, whether to save intermediate steps
        image_name: original filename (without extension)
    
    Returns:
        Final shuffled image, gallery of steps, and message about processing
    """
    if image is None:
        return None, None, "Please upload an image."
    
    # Convert to PIL Image if needed and ensure RGB mode
    if isinstance(image, Image.Image):
        # Convert RGBA, L, or other modes to RGB
        if image.mode == 'RGBA':
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
    else:
        img_array = image
    
    # Check minimum size
    h, w = img_array.shape[:2]
    if h < 4 or w < 4:
        return None, None, "Image is too small. Minimum size is 4x4 pixels."
    
    # Always process with step tracking to show progression
    steps = recursive_shuffle_with_steps(img_array, seed)
    
    # Convert steps to PIL images for gallery
    step_images = [Image.fromarray(step_img) for step_img in steps]
    
    if save_steps:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"quadtree_shuffle_steps_{timestamp}_seed{seed}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all steps using original image name
        saved_files = []
        for idx, step_img in enumerate(step_images):
            filename = f"{image_name}{idx:04d}.png"
            filepath = os.path.join(output_dir, filename)
            step_img.save(filepath)
            saved_files.append(filename)
        
        message = (f"Processing complete.\n"
                  f"Saved {len(saved_files)} images to: {output_dir}\n"
                  f"Seed: {seed}\n"
                  f"Original size: {w}x{h}\n"
                  f"Total recursion levels: {len(steps) - 1}")
    else:
        message = (f"Processing complete.\n"
                  f"Seed: {seed}\n"
                  f"Image size: {w}x{h}\n"
                  f"Total recursion levels: {len(steps) - 1}")
    
    # Get final image
    final_image = step_images[-1]
    
    return final_image, step_images, message


def create_demo():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Quadtree Image Shuffler") as demo:
        gr.Markdown("# Quadtree Image Shuffler")
        gr.Markdown(
            "This tool recursively divides your image into 4x4 grids and shuffles "
            "the pieces at each level using a fixed seed. The process continues "
            "until the pieces are too small to subdivide further (less than 4x4 pixels)."
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath"
                )
                
                seed_input = gr.Number(
                    label="Random Seed",
                    value=42,
                    precision=0,
                    info="Same seed produces same shuffle pattern"
                )
                
                save_steps_checkbox = gr.Checkbox(
                    label="Save all intermediate steps as PNG files",
                    value=False,
                    info="Creates a folder with images from each recursion level"
                )
                
                process_btn = gr.Button("Shuffle Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(
                    label="Final Shuffled Image",
                    type="pil"
                )
                
                output_message = gr.Textbox(
                    label="Processing Info",
                    lines=6
                )
        
        with gr.Row():
            step_gallery = gr.Gallery(
                label="Shuffle Progression (All Levels)",
                columns=4,
                height="auto"
            )
        
        # Connect the button to the processing function
        def process_wrapper(image_path, seed, save_steps):
            if image_path is None:
                return None, None, "Please upload an image."
            
            # Extract filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Load the image
            image = Image.open(image_path)
            
            return process_image(image, seed, save_steps, base_name)
        
        process_btn.click(
            fn=process_wrapper,
            inputs=[input_image, seed_input, save_steps_checkbox],
            outputs=[output_image, step_gallery, output_message]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(inbrowser=True)
