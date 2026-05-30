#!/usr/bin/env python3
"""
Image Montage Artist - Advanced Polygon-based Image Matching
Creates photomontages by extracting irregular polygon pieces from a source image
and placing them on a target image, rotated and positioned to best match the target
using perceptually uniform LAB color space comparison.
Simulates how a montage artist would work with physical cutouts.
"""

import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import random
from datetime import datetime
import os
import threading
from scipy.spatial import ConvexHull
import cv2

# Global flag for stopping the montage process
stop_montage = threading.Event()


def pad_to_square(img, target_size=1024):
    """Pad image with white to 1:1 ratio, then resize to target size."""
    width, height = img.size
    max_dim = max(width, height)
    square_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    square_img.paste(img, (left, top))
    
    if square_img.size != (target_size, target_size):
        square_img = square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return square_img


def generate_random_convex_polygon(min_vertices=4, max_vertices=8, min_size=20, max_size=80):
    """Generate a random convex polygon."""
    num_vertices = random.randint(min_vertices, max_vertices)
    
    # Generate random points in a circle to ensure convexity
    angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_vertices)])
    radius = random.uniform(min_size, max_size)
    
    points = []
    for angle in angles:
        # Add some randomness to the radius for each point
        r = radius * random.uniform(0.5, 1.0)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    
    # Use ConvexHull to ensure convexity
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return hull_points
    except:
        # Fallback to a simple triangle if hull fails
        return np.array([[0, -radius/2], [-radius/2, radius/2], [radius/2, radius/2]])


def rotate_polygon(polygon, angle_degrees):
    """Rotate a polygon by given angle in degrees."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    return polygon @ rotation_matrix.T


def create_polygon_mask(polygon, image_size, center):
    """Create a binary mask for the polygon at the given center position."""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Translate polygon to center position
    translated_polygon = polygon + np.array(center)
    
    # Convert to integer coordinates
    int_polygon = translated_polygon.astype(np.int32)
    
    # Draw filled polygon on mask
    cv2.fillConvexPoly(mask, int_polygon, 255)
    
    return mask


def extract_polygon_region(image_array, polygon, center, rotation_angle=0):
    """
    Extract the region defined by a polygon from an image.
    Returns the extracted region, mask, and bounding box.
    """
    # Rotate the polygon
    rotated_polygon = rotate_polygon(polygon, rotation_angle)
    
    # Create mask
    mask = create_polygon_mask(rotated_polygon, image_array.shape[0], center)
    
    # Get bounding box
    translated_polygon = rotated_polygon + np.array(center)
    min_x = max(0, int(np.min(translated_polygon[:, 0])))
    max_x = min(image_array.shape[1], int(np.max(translated_polygon[:, 0])))
    min_y = max(0, int(np.min(translated_polygon[:, 1])))
    max_y = min(image_array.shape[0], int(np.max(translated_polygon[:, 1])))
    
    # Extract region
    bbox = (min_y, max_y, min_x, max_x)
    
    return mask, bbox


def calculate_match_score(source_region, target_region, source_mask, target_mask):
    """
    Calculate how well two regions match, considering only the masked areas.
    Uses Mean Squared Error in LAB color space for perceptually uniform comparison.
    """
    # Combine masks to only compare where both have content
    combined_mask = (source_mask > 0) & (target_mask > 0)
    
    if np.sum(combined_mask) == 0:
        return float('inf')  # No overlap, worst score
    
    # Convert regions to LAB color space for perceptually uniform comparison
    source_lab = cv2.cvtColor(source_region.astype(np.uint8), cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target_region.astype(np.uint8), cv2.COLOR_RGB2LAB)
    
    # Calculate MSE only on masked region
    source_masked = source_lab[combined_mask]
    target_masked = target_lab[combined_mask]
    
    mse = np.mean((source_masked.astype(np.float32) - target_masked.astype(np.float32)) ** 2)
    
    return mse


def find_best_position_and_rotation(source_array, target_array, polygon, source_mask_global,
                                   dest_mask_global, sample_step=20, rotation_samples=8):
    """
    Find the best position and rotation for a polygon piece by sampling across the target image.
    Returns best position, rotation, and score.
    """
    best_score = float('inf')
    best_position = None
    best_rotation = None
    
    image_size = target_array.shape[0]
    
    # Sample rotations
    rotations = [i * (360 / rotation_samples) for i in range(rotation_samples)]
    
    # Sample positions across the image
    for y in range(50, image_size - 50, sample_step):
        for x in range(50, image_size - 50, sample_step):
            # Check if destination area is already occupied
            test_mask, bbox = extract_polygon_region(target_array, polygon, (x, y), 0)
            min_y, max_y, min_x, max_x = bbox
            
            if np.any(dest_mask_global[min_y:max_y, min_x:max_x] & (test_mask[min_y:max_y, min_x:max_x] > 0)):
                continue  # This position overlaps with existing pieces
            
            # Try different rotations
            for rotation in rotations:
                source_mask, source_bbox = extract_polygon_region(source_array, polygon, (x, y), rotation)
                s_min_y, s_max_y, s_min_x, s_max_x = source_bbox
                
                # Check if source area is already used
                if np.any(source_mask_global[s_min_y:s_max_y, s_min_x:s_max_x] & 
                         (source_mask[s_min_y:s_max_y, s_min_x:s_max_x] > 0)):
                    continue  # This source region already used
                
                # Calculate match score
                target_region = target_array[min_y:max_y, min_x:max_x]
                source_region = source_array[s_min_y:s_max_y, s_min_x:s_max_x]
                
                target_mask_crop = test_mask[min_y:max_y, min_x:max_x]
                source_mask_crop = source_mask[s_min_y:s_max_y, s_min_x:s_max_x]
                
                # Ensure regions are the same size for comparison
                if target_region.shape[:2] != source_region.shape[:2]:
                    continue
                
                score = calculate_match_score(source_region, target_region, 
                                             source_mask_crop, target_mask_crop)
                
                if score < best_score:
                    best_score = score
                    best_position = (x, y)
                    best_rotation = rotation
    
    return best_position, best_rotation, best_score


def place_polygon_piece(result_array, dest_mask, source_array, source_mask_global,
                       polygon, source_pos, dest_pos, rotation):
    """
    Place a polygon piece from source onto the result image.
    Updates both the result array and the mask tracking used regions.
    """
    # Get source region
    source_mask, source_bbox = extract_polygon_region(source_array, polygon, source_pos, rotation)
    s_min_y, s_max_y, s_min_x, s_max_x = source_bbox
    
    # Get destination region
    dest_mask_poly, dest_bbox = extract_polygon_region(result_array, polygon, dest_pos, 0)
    d_min_y, d_max_y, d_min_x, d_max_x = dest_bbox
    
    # Extract the source piece
    source_piece = source_array[s_min_y:s_max_y, s_min_x:s_max_x].copy()
    source_mask_crop = source_mask[s_min_y:s_max_y, s_min_x:s_max_x]
    
    # Place on destination
    dest_piece = result_array[d_min_y:d_max_y, d_min_x:d_max_x]
    dest_mask_crop = dest_mask_poly[d_min_y:d_max_y, d_min_x:d_max_x]
    
    # Ensure dimensions match
    if source_piece.shape[:2] == dest_piece.shape[:2]:
        # Apply the piece completely opaque - it obliterates whatever is underneath
        mask_3d = np.stack([dest_mask_crop] * 3, axis=-1) > 0
        dest_piece[mask_3d] = source_piece[mask_3d]
        
        # Update the destination mask
        dest_mask[d_min_y:d_max_y, d_min_x:d_max_x] |= dest_mask_crop
        
        # Update source mask to mark this region as used
        source_mask_global[s_min_y:s_max_y, s_min_x:s_max_x] |= source_mask_crop
        
        return True
    
    return False


def create_montage(source_image, target_image, min_pieces=100, max_pieces=300, 
                   min_poly_size=15, max_poly_size=60, sample_step=25, rotation_samples=8):
    """
    Create a montage by placing irregular polygon pieces from source onto target.
    """
    global stop_montage
    stop_montage.clear()
    
    if source_image is None or target_image is None:
        yield None, "Please upload both source and target images."
        return
    
    try:
        # Prepare images
        source_img = pad_to_square(source_image, 1024)
        target_img = pad_to_square(target_image, 1024)
        
        # Scale source image to 150% for better coverage
        scaled_size = int(1024 * 1.5)
        source_img_scaled = source_img.resize((scaled_size, scaled_size), Image.Resampling.LANCZOS)
        
        source_array = np.array(source_img_scaled)
        target_array = np.array(target_img)
        
        # Initialize result with white background
        result_array = np.full_like(target_array, 255)
        
        # Masks to track used regions (scaled for source)
        source_mask_global = np.zeros((scaled_size, scaled_size), dtype=np.uint8)
        dest_mask_global = np.zeros((1024, 1024), dtype=np.uint8)
        
        num_pieces = random.randint(min_pieces, max_pieces)
        pieces_placed = 0
        attempts = 0
        max_attempts = num_pieces * 5  # Allow more attempts than pieces
        
        yield Image.fromarray(result_array), f"Starting montage creation...\nTarget: {num_pieces} pieces"
        
        while pieces_placed < num_pieces and attempts < max_attempts and not stop_montage.is_set():
            attempts += 1
            
            # Generate a random convex polygon
            polygon = generate_random_convex_polygon(
                min_vertices=4,
                max_vertices=8,
                min_size=min_poly_size,
                max_size=max_poly_size
            )
            
            # Find a random source position that hasn't been used
            source_pos = None
            for _ in range(50):  # Try up to 50 random positions
                sx = random.randint(100, scaled_size - 100)
                sy = random.randint(100, scaled_size - 100)
                
                # Check if this area is mostly unused
                test_mask, test_bbox = extract_polygon_region(source_array, polygon, (sx, sy), 0)
                t_min_y, t_max_y, t_min_x, t_max_x = test_bbox
                
                overlap = np.sum(source_mask_global[t_min_y:t_max_y, t_min_x:t_max_x] & 
                               (test_mask[t_min_y:t_max_y, t_min_x:t_max_x] > 0))
                total = np.sum(test_mask[t_min_y:t_max_y, t_min_x:t_max_x] > 0)
                
                if total > 0 and overlap / total < 0.1:  # Less than 10% overlap is acceptable
                    source_pos = (sx, sy)
                    break
            
            if source_pos is None:
                continue  # Couldn't find good source position
            
            # Find best destination position and rotation
            dest_pos, rotation, score = find_best_position_and_rotation(
                source_array, target_array, polygon, source_mask_global,
                dest_mask_global, sample_step, rotation_samples
            )
            
            if dest_pos is None:
                continue  # Couldn't find good destination
            
            # Place the piece
            success = place_polygon_piece(
                result_array, dest_mask_global, source_array, source_mask_global,
                polygon, source_pos, dest_pos, rotation
            )
            
            if success:
                pieces_placed += 1
                
                # Yield progress every 5 pieces
                if pieces_placed % 5 == 0 or pieces_placed == 1:
                    result_image = Image.fromarray(result_array)
                    status = (f"Pieces placed: {pieces_placed}/{num_pieces}\n"
                            f"Attempts: {attempts}\n"
                            f"Coverage: {np.sum(dest_mask_global > 0) / (1024*1024) * 100:.1f}%")
                    yield result_image, status
        
        # Final result
        result_image = Image.fromarray(result_array)
        final_status = (f"Montage complete\n"
                       f"Pieces placed: {pieces_placed}/{num_pieces}\n"
                       f"Total attempts: {attempts}\n"
                       f"Coverage: {np.sum(dest_mask_global > 0) / (1024*1024) * 100:.1f}%")
        yield result_image, final_status
        
    except Exception as e:
        yield None, f"Error: {str(e)}"


def stop_montage_handler():
    """Stop the montage creation process."""
    global stop_montage
    stop_montage.set()
    return "Stopping montage creation..."


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Image Montage Artist") as demo:
        gr.Markdown("# Image Montage Artist")
        gr.Markdown("Creates photomontages using irregular polygon pieces, like a collage artist")
        
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(
                    label="Source Image (pieces cut from here)",
                    type="pil",
                    height=300
                )
                
                target_image = gr.Image(
                    label="Target Image (to replicate with montage)",
                    type="pil",
                    height=300
                )
                
                with gr.Row():
                    min_pieces = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Minimum Pieces"
                    )
                    
                    max_pieces = gr.Slider(
                        minimum=100,
                        maximum=500,
                        value=300,
                        step=25,
                        label="Maximum Pieces"
                    )
                
                with gr.Row():
                    min_poly_size = gr.Slider(
                        minimum=10,
                        maximum=40,
                        value=15,
                        step=5,
                        label="Min Polygon Size"
                    )
                    
                    max_poly_size = gr.Slider(
                        minimum=30,
                        maximum=120,
                        value=60,
                        step=10,
                        label="Max Polygon Size"
                    )
                
                with gr.Row():
                    sample_step = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=5,
                        label="Position Sample Step (lower = slower but better)",
                        info="Step size for position sampling"
                    )
                    
                    rotation_samples = gr.Slider(
                        minimum=4,
                        maximum=16,
                        value=8,
                        step=2,
                        label="Rotation Samples",
                        info="Number of rotation angles to try"
                    )
                
                with gr.Row():
                    generate_btn = gr.Button("Create Montage", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")
            
            with gr.Column():
                result_image = gr.Image(
                    label="Montage Result",
                    type="pil",
                    height=600
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=6,
                    max_lines=10
                )
                
                stop_status = gr.Textbox(
                    label="Control",
                    lines=1
                )
        
        gr.Markdown("""
        ### How it works:
        - Source image is scaled to 150% for better coverage and more material
        - Generates random convex polygons of varying sizes and vertex counts
        - Extracts polygon pieces from the scaled source (without reusing regions)
        - Tests different positions and rotations (0-360 degrees) on the target
        - Places each piece where it best matches the target image using LAB color space
        - LAB color space ensures perceptually uniform color matching
        - Ensures no overlapping pieces on the destination
        - Simulates the workflow of a montage artist cutting and placing pieces
        
        ### Tips:
        - Smaller polygon sizes create more detailed montages but take longer
        - Lower sample step values give better quality but slower processing
        - More rotation samples improve matching but increase processing time
        - More pieces create fuller coverage without overlapping
        - 150% source scaling provides extra material for better coverage
        """)
        
        # Set up the generation action
        generate_btn.click(
            fn=create_montage,
            inputs=[source_image, target_image, min_pieces, max_pieces, 
                   min_poly_size, max_poly_size, sample_step, rotation_samples],
            outputs=[result_image, status_output]
        )
        
        # Set up the stop button
        stop_btn.click(
            fn=stop_montage_handler,
            inputs=[],
            outputs=[stop_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)
