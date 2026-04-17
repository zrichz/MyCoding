#!/usr/bin/env python3
"""
Pixel Insertion Gradio App - LSB Steganography Edition
Embeds pixels from Image A into Image B using LSB encoding and random distribution.
"""

import gradio as gr
import numpy as np
from PIL import Image
import json
import os
import hashlib
from datetime import datetime


def generate_pixel_positions(img_b_shape, img_a_shape, seed):
    """
    Generate randomized pixel positions across image B based on a seed.
    Each pixel from A needs 6 pixels from B (3 pairs for R, G, B channels).
    
    Args:
        img_b_shape: (height, width) of image B
        img_a_shape: (height, width) of image A
        seed: String seed for randomization
    
    Returns:
        List of (row, col) positions for pixels in B, or error message
    """
    height_b, width_b = img_b_shape
    height_a, width_a = img_a_shape
    
    # Create all possible pixel positions in image B
    all_positions = []
    for i in range(height_b):
        for j in range(width_b):
            all_positions.append((i, j))
    
    # Shuffle using seed
    rng = np.random.RandomState(int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32))
    rng.shuffle(all_positions)
    
    # Take only what we need (6 pixels per pixel in A)
    needed = height_a * width_a * 6
    if len(all_positions) < needed:
        return None, f"Image B too small. Can fit {len(all_positions)} pixels but need {needed} (6 per pixel in A)"
    
    return all_positions[:needed], None


def encode_channel_pair(channel_value, pixel_b1, pixel_b2):
    """
    Encode one 8-bit channel value into a pair of pixels from B.
    Distribution: 2 bits red + 2 bits blue + 1 bit green (pixel B1) = 5 bits
                  1 bit red + 1 bit green + 1 bit blue (pixel B2) = 3 bits
    
    Args:
        channel_value: 8-bit value (0-255) to encode
        pixel_b1: First pixel from image B (RGB array)
        pixel_b2: Second pixel from image B (RGB array)
    
    Returns:
        Modified pixel_b1 and pixel_b2
    """
    # Convert channel value to 8 bits (MSB to LSB)
    bits = [(channel_value >> (7 - i)) & 1 for i in range(8)]
    
    # Copy pixels to modify
    modified_b1 = pixel_b1.copy()
    modified_b2 = pixel_b2.copy()
    
    # Encode into pixel B1: 2 bits red, 2 bits blue, 1 bit green
    # Red channel: bits 0-1
    modified_b1[0] = (modified_b1[0] & 0xFC) | (bits[0] << 1) | bits[1]
    # Blue channel: bits 2-3
    modified_b1[2] = (modified_b1[2] & 0xFC) | (bits[2] << 1) | bits[3]
    # Green channel: bit 4
    modified_b1[1] = (modified_b1[1] & 0xFE) | bits[4]
    
    # Encode into pixel B2: 1 bit red, 1 bit green, 1 bit blue
    # Red channel: bit 5
    modified_b2[0] = (modified_b2[0] & 0xFE) | bits[5]
    # Green channel: bit 6
    modified_b2[1] = (modified_b2[1] & 0xFE) | bits[6]
    # Blue channel: bit 7
    modified_b2[2] = (modified_b2[2] & 0xFE) | bits[7]
    
    return modified_b1, modified_b2


def decode_channel_pair(pixel_b1, pixel_b2):
    """
    Decode one 8-bit channel value from a pair of pixels from B.
    
    Args:
        pixel_b1: First pixel from image B (RGB array)
        pixel_b2: Second pixel from image B (RGB array)
    
    Returns:
        Decoded 8-bit channel value (0-255)
    """
    bits = []
    
    # Extract from pixel B1: 2 bits red, 2 bits blue, 1 bit green
    # Red channel: bits 0-1
    bits.append((pixel_b1[0] >> 1) & 1)
    bits.append(pixel_b1[0] & 1)
    # Blue channel: bits 2-3
    bits.append((pixel_b1[2] >> 1) & 1)
    bits.append(pixel_b1[2] & 1)
    # Green channel: bit 4
    bits.append(pixel_b1[1] & 1)
    
    # Extract from pixel B2: 1 bit red, 1 bit green, 1 bit blue
    # Red channel: bit 5
    bits.append(pixel_b2[0] & 1)
    # Green channel: bit 6
    bits.append(pixel_b2[1] & 1)
    # Blue channel: bit 7
    bits.append(pixel_b2[2] & 1)
    
    # Convert 8 bits to value
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    
    return value


def embed_image(img_a, img_b, seed="default_seed"):
    """
    Embed pixels from image A into image B using LSB steganography.
    Each pixel from A uses 6 pixels from B (3 channel pairs).
    
    Args:
        img_a: Source image (PIL Image) - max 1333x1500
        img_b: Target image (PIL Image) - min 4000x3000
        seed: Seed for random pixel distribution (password/key)
    
    Returns:
        tuple: (modified_b, position_map_viz, pixel_positions, message)
    """
    if img_a is None or img_b is None:
        return None, None, None, "Please upload both images A and B"
    
    if not seed or seed.strip() == "":
        seed = "default_seed"
    
    # Convert to RGB if needed
    if img_a.mode != 'RGB':
        img_a = img_a.convert('RGB')
    if img_b.mode != 'RGB':
        img_b = img_b.convert('RGB')
    
    # Get dimensions
    width_a, height_a = img_a.size
    width_b, height_b = img_b.size
    
    # Validate size constraints
    if width_a > 1333 or height_a > 1500:
        return None, None, None, f"Image A too large. Maximum size is 1333x1500, got {width_a}x{height_a}"
    
    if width_b < 4000 or height_b < 3000:
        return None, None, None, f"Image B too small. Minimum size is 4000x3000, got {width_b}x{height_b}"
    
    # Convert to numpy arrays
    arr_a = np.array(img_a)
    arr_b = np.array(img_b).copy()
    
    # Generate random pixel positions
    pixel_positions, error = generate_pixel_positions(
        (height_b, width_b), 
        (height_a, width_a), 
        seed
    )
    
    if error or pixel_positions is None:
        return None, None, None, error if error else "Failed to generate pixel positions"
    
    # Embed each pixel from A
    pixel_idx = 0
    for i in range(height_a):
        for j in range(width_a):
            pixel_a = arr_a[i, j]
            
            # Each pixel from A needs 6 pixels from B (3 pairs for R, G, B)
            for channel in range(3):
                # Get two pixel positions for this channel
                pos1_row, pos1_col = pixel_positions[pixel_idx]
                pos2_row, pos2_col = pixel_positions[pixel_idx + 1]
                pixel_idx += 2
                
                # Get the two pixels from B
                pixel_b1 = arr_b[pos1_row, pos1_col]
                pixel_b2 = arr_b[pos2_row, pos2_col]
                
                # Encode this channel
                modified_b1, modified_b2 = encode_channel_pair(
                    pixel_a[channel], pixel_b1, pixel_b2
                )
                
                # Store modified pixels back
                arr_b[pos1_row, pos1_col] = modified_b1
                arr_b[pos2_row, pos2_col] = modified_b2
    
    # Convert back to PIL Image
    result_img = Image.fromarray(arr_b)
    
    # Create visualization of pixel positions
    viz_img = np.zeros((height_b, width_b), dtype=np.uint8)
    for pos_idx in range(0, len(pixel_positions), 6):
        # Mark every 6th pixel (one per source pixel)
        if pos_idx < len(pixel_positions):
            row, col = pixel_positions[pos_idx]
            viz_img[row, col] = 255
    
    position_viz_img = Image.fromarray(viz_img, mode='L')
    
    # Calculate max change in pixel values
    max_change = np.max(np.abs(arr_b.astype(np.int16) - np.array(img_b).astype(np.int16)))
    
    message = (f"Successfully embedded {width_a}x{height_a} image A into image B\n"
              f"Used {len(pixel_positions)} randomly distributed pixels (6 per source pixel)\n"
              f"Seed: '{seed}'\n"
              f"Max pixel change: {max_change} (LSB encoding)\n"
              f"Image B size preserved: {width_b}x{height_b}")
    
    return result_img, position_viz_img, pixel_positions, message


def reconstruct_image(img_b, pixel_data):
    """
    Reconstruct image A from the modified image B using LSB decoding.
    
    Args:
        img_b: Modified image B (PIL Image)
        pixel_data: Pixel position data (list or dict)
    
    Returns:
        tuple: (reconstructed_a, message)
    """
    try:
        if img_b is None:
            return None, "Please upload the modified image B"
        
        if pixel_data is None:
            return None, "Please provide pixel position data"
        
        # Handle pixel_positions whether it's a list or loaded from JSON
        if isinstance(pixel_data, dict):
            pixel_positions = pixel_data['pixel_positions']
            img_a_width = pixel_data['image_a_size'][0]
            img_a_height = pixel_data['image_a_size'][1]
        else:
            pixel_positions = pixel_data
            # Calculate dimensions from pixel count (6 pixels per source pixel)
            total_source_pixels = len(pixel_positions) // 6
            img_a_height = int(np.sqrt(total_source_pixels))
            img_a_width = total_source_pixels // img_a_height
        
        # Convert to RGB if needed
        if img_b.mode != 'RGB':
            img_b = img_b.convert('RGB')
        
        arr_b = np.array(img_b)
        
        # Create array for reconstructed image A
        arr_a_reconstructed = np.zeros((img_a_height, img_a_width, 3), dtype=np.uint8)
        
        # Reconstruct each pixel
        pixel_idx = 0
        for i in range(img_a_height):
            for j in range(img_a_width):
                reconstructed_pixel = []
                
                # Decode each channel (R, G, B)
                for channel in range(3):
                    # Get two pixel positions for this channel
                    pos1_row, pos1_col = pixel_positions[pixel_idx]
                    pos2_row, pos2_col = pixel_positions[pixel_idx + 1]
                    pixel_idx += 2
                    
                    # Get the two pixels from B
                    pixel_b1 = arr_b[pos1_row, pos1_col]
                    pixel_b2 = arr_b[pos2_row, pos2_col]
                    
                    # Decode this channel
                    channel_value = decode_channel_pair(pixel_b1, pixel_b2)
                    reconstructed_pixel.append(channel_value)
                
                arr_a_reconstructed[i, j] = reconstructed_pixel
        
        # Convert to PIL Image
        reconstructed_img = Image.fromarray(arr_a_reconstructed)
        
        # Get seed from data if available for display
        seed_info = ""
        if isinstance(pixel_data, dict) and 'seed' in pixel_data:
            seed_info = f" (original seed: '{pixel_data['seed']}')"
        
        message = f"Successfully reconstructed {img_a_width}x{img_a_height} image A{seed_info}"
        
        return reconstructed_img, message
    
    except Exception as e:
        return None, f"Error during reconstruction: {str(e)}"


def save_pixel_positions(pixel_positions, img_a_size, seed):
    """Save pixel positions to a JSON file."""
    if pixel_positions is None:
        return None, "No pixel position data to save"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pixel_positions_{timestamp}.json"
    
    data = {
        'timestamp': timestamp,
        'seed': seed,
        'image_a_size': img_a_size,
        'pixel_positions': pixel_positions
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    return filename, f"Pixel positions saved to {filename}"


def save_modified_image_png(img):
    """Save the modified image B as PNG format."""
    if img is None:
        return None, "No image to save"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"modified_image_b_{timestamp}.png"
    
    img.save(filename, format='PNG', compress_level=1)  # Low compression for speed
    
    return filename, f"Modified image B saved to {filename} (PNG format)"


def load_pixel_positions(file_path):
    """Load pixel positions from a JSON file."""
    if file_path is None:
        return None, None, "No file selected"
    
    try:
        # Handle both string paths and file objects from Gradio
        if hasattr(file_path, 'name'):
            file_path = file_path.name
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Support both old and new format
        if 'pixel_positions' in data:
            pixel_positions = data['pixel_positions']
        elif 'block_positions' in data:
            pixel_positions = data['block_positions']  # backward compatibility
        else:
            return None, None, "Invalid data file format"
        
        seed = data.get('seed', 'default_seed')
        message = f"Loaded data for {data['image_a_size'][0]}x{data['image_a_size'][1]} image (seed: '{seed}')"
        
        return data, seed, message
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}"


# Global variables to store current data
current_pixel_positions = None
current_modified_image = None
current_seed = None
current_img_a_size = None


def embed_and_store(img_a, img_b, seed):
    """Wrapper to embed and store pixel positions globally."""
    global current_pixel_positions, current_modified_image, current_seed, current_img_a_size
    
    result_img, position_viz, pixel_positions, message = embed_image(img_a, img_b, seed)
    current_pixel_positions = pixel_positions
    current_modified_image = result_img
    current_seed = seed
    if img_a is not None:
        current_img_a_size = img_a.size
    
    return result_img, position_viz, message


def save_current_data(img_a):
    """Save the current pixel positions."""
    global current_pixel_positions, current_seed
    
    if current_pixel_positions is None:
        return None, "No pixel position data available. Please embed images first."
    
    if img_a is None:
        return None, "Image A is required to save pixel positions"
    
    filename, message = save_pixel_positions(current_pixel_positions, img_a.size, current_seed)
    return filename, message


def save_current_image():
    """Save the current modified image B as PNG."""
    global current_modified_image
    
    if current_modified_image is None:
        return None, "No modified image available. Please embed images first."
    
    filename, message = save_modified_image_png(current_modified_image)
    return filename, message


def reconstruct_with_loaded_data(img_b, data_file):
    """Reconstruct using a loaded pixel positions file."""
    try:
        if img_b is None:
            return None, "Please upload the modified image B"
        
        if data_file is None:
            return None, "Please upload the pixel positions JSON file"
        
        pixel_data, loaded_seed, load_msg = load_pixel_positions(data_file)
        
        if pixel_data is None:
            return None, load_msg
        
        reconstructed, recon_msg = reconstruct_image(img_b, pixel_data)
        
        if reconstructed is None:
            return None, recon_msg
        
        return reconstructed, f"{load_msg}\n{recon_msg}"
    except Exception as e:
        return None, f"Error during reconstruction: {str(e)}"


def reconstruct_with_current_data(img_b):
    """Reconstruct using the current session's data."""
    global current_pixel_positions, current_seed, current_img_a_size
    
    try:
        if img_b is None:
            return None, "Please upload the modified image B"
        
        if current_pixel_positions is None:
            return None, "No pixel position data available. Please embed images first or load a data file."
        
        if current_img_a_size is None:
            return None, "No image A size information available. Please embed images first."
        
        # Create data structure compatible with reconstruct_image
        pixel_data = {
            'pixel_positions': current_pixel_positions,
            'image_a_size': current_img_a_size,
            'seed': current_seed
        }
        
        reconstructed, message = reconstruct_image(img_b, pixel_data)
        
        if reconstructed is None:
            return None, message
        
        return reconstructed, message
    except Exception as e:
        return None, f"Error during reconstruction: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="LSB Steganography - Pixel Insertion") as demo:
    gr.Markdown("""
    # LSB Steganography - Pixel Insertion
    
    This app embeds pixels from Image A into Image B using LSB (Least Significant Bit) encoding 
    and random distribution for true steganography. Changes are nearly invisible.
    
    ## Key Features:
    - Random pixel distribution using password/seed during embedding
    - LSB encoding - max pixel change of 3 (virtually invisible)
    - Preserves image B's original size and aspect ratio
    - Secured by position file - need the JSON file to extract
    - Size requirements: Image A max 1333x1500, Image B min 4000x3000
    
    ## How it works:
    1. Embed: Upload images, enter a password/seed to randomize positions, embed A into B
    2. Save: Save the pixel position file (JSON) and modified image B (PNG)
    3. Reconstruct: Load both files to recover image A perfectly
    """)
    
    with gr.Tab("Embed"):
        with gr.Row():
            with gr.Column():
                input_a = gr.Image(label="Image A (Source to hide, max 1333x1500)", type="pil")
                input_b = gr.Image(label="Image B (Carrier image, min 4000x3000)", type="pil")
                seed_input = gr.Textbox(
                    label="Password/Seed", 
                    placeholder="Enter a password (required for reconstruction)",
                    value="my_secret_password",
                    type="password"
                )
                embed_btn = gr.Button("Embed Image A into Image B", variant="primary", size="lg")
            
            with gr.Column():
                output_b = gr.Image(label="Modified Image B (steganography applied)")
                position_viz = gr.Image(label="Pixel Distribution Visualization")
                embed_msg = gr.Textbox(label="Status", lines=5)
        
        gr.Markdown("### Save Files (Both Required for Reconstruction)")
        with gr.Row():
            with gr.Column():
                save_data_btn = gr.Button("Save Pixel Positions JSON", variant="secondary")
                data_file_output = gr.File(label="Pixel Positions File (keep this safe)")
            with gr.Column():
                save_img_btn = gr.Button("Save Modified Image B as PNG", variant="secondary")
                img_file_output = gr.File(label="Modified Image B (PNG)")
        
        with gr.Row():
            save_msg = gr.Textbox(label="Save Status", lines=2)
        
        embed_btn.click(
            fn=embed_and_store,
            inputs=[input_a, input_b, seed_input],
            outputs=[output_b, position_viz, embed_msg]
        )
        
        save_data_btn.click(
            fn=save_current_data,
            inputs=[input_a],
            outputs=[data_file_output, save_msg]
        )
        
        save_img_btn.click(
            fn=save_current_image,
            inputs=[],
            outputs=[img_file_output, save_msg]
        )
    
    with gr.Tab("Reconstruct"):
        gr.Markdown("""
        ### Reconstruct Image A from Modified Image B
        
        Instructions:
        1. Upload the modified image B (PNG file)
        2. Upload the pixel positions JSON file
        3. Click Reconstruct to recover the original image A
        
        Note: The pixel positions file contains all necessary information for reconstruction
        """)
        
        with gr.Row():
            with gr.Column():
                recon_input_b = gr.Image(label="Modified Image B (PNG)", type="pil")
                recon_data_file = gr.File(label="Pixel Positions (.json)")
                recon_btn = gr.Button("Reconstruct Image A", variant="primary", size="lg")
            
            with gr.Column():
                recon_output_a = gr.Image(label="Reconstructed Image A")
                recon_msg = gr.Textbox(label="Status", lines=4)
        
        gr.Markdown("---")
        gr.Markdown("### Quick Reconstruct (Current Session)")
        gr.Markdown("Use this if you just embedded images in this session:")
        with gr.Row():
            with gr.Column():
                quick_input_b = gr.Image(label="Modified Image B", type="pil")
                recon_current_btn = gr.Button("Quick Reconstruct", variant="secondary")
            with gr.Column():
                quick_output_a = gr.Image(label="Reconstructed Image A")
                quick_msg = gr.Textbox(label="Status", lines=3)
        
        recon_btn.click(
            fn=reconstruct_with_loaded_data,
            inputs=[recon_input_b, recon_data_file],
            outputs=[recon_output_a, recon_msg]
        )
        
        recon_current_btn.click(
            fn=reconstruct_with_current_data,
            inputs=[quick_input_b],
            outputs=[quick_output_a, quick_msg]
        )
    
    with gr.Tab("Help"):
        gr.Markdown("""
        ## How to Use This App
        
        ### Step 1: Embed (Hide Image A in Image B)
        1. Go to the Embed tab
        2. Upload Image A (the image you want to hide, max 1333x1500)
        3. Upload Image B (the carrier image, min 4000x3000)
        4. Enter a password/seed (remember this)
        5. Click Embed Image A into Image B
        6. Save both files:
           - Click Save Pixel Positions JSON
           - Click Save Modified Image B as PNG
        
        ### Step 2: Reconstruct (Recover Image A)
        1. Go to the Reconstruct tab
        2. Upload the modified image B (PNG file)
        3. Upload the pixel positions JSON file
        4. Click Reconstruct Image A
        5. You'll get back the original image A perfectly
        
        ## Technical Details
        
        ### LSB (Least Significant Bit) Encoding
        - Uses 1-2 least significant bits per color channel
        - Maximum pixel change: 3 per channel (virtually invisible)
        - Example: (255, 128, 64) might become (254, 127, 65)
        - The human eye cannot detect these tiny changes
        
        ### Encoding Scheme
        Each pixel from A is encoded in 6 pixels from B (3 channel pairs):
        - First pixel pair encodes Red channel (8 bits)
        - Second pixel pair encodes Green channel (8 bits)
        - Third pixel pair encodes Blue channel (8 bits)
        
        Per pair distribution:
        - First pixel: 2 bits in red, 2 bits in blue, 1 bit in green (5 bits total)
        - Second pixel: 1 bit in red, 1 bit in green, 1 bit in blue (3 bits total)
        
        ### Random Distribution
        - Pixel positions are randomized using your password/seed during embedding
        - Different passwords create completely different embedding patterns
        - The randomized positions are stored in the JSON file
        - Security comes from keeping the JSON file secret
        - Pixels are spread across the entire image, not clustered
        
        ### Security Features
        - Secured by pixel position file: Without the JSON file, extraction is impossible
        - No visual artifacts: Changes are LSB-only
        - Preserves image B: Original size and aspect ratio maintained
        - PNG lossless: No compression artifacts
        
        ## Size Requirements
        
        - Image A: Maximum 1333x1500 pixels (donor image)
        - Image B: Minimum 4000x3000 pixels (receptor image)
        - Each pixel from A needs 6 pixels from B
        - Larger B allows better scattering and stronger steganography
        
        ## Tips for Best Steganography
        
        - Use complex carrier images: Photos with lots of detail hide changes better than solid colors
        - Choose strong passwords: Different seeds create different embedding patterns
        - Keep JSON file secret: This file is essential for reconstruction and contains the positions
        - Use PNG format: Always save modified image as PNG (lossless)
        - Test reconstruction: Always verify you can extract before relying on it
        - Backup the JSON file: Without it, recovery is impossible
        """)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
