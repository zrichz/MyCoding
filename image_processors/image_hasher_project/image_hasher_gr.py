#!/home/rich/MyCoding/venvmycoding313/bin/python
"""
Image Hasher (gradio) - Converted from Processing
pixel scatter using deterministic random seed.
Supports images up to 4096x4096 pixels.
use a seed like access credit card or something
"""
import gradio as gr
import numpy as np
from PIL import Image
import random
import hashlib
from datetime import datetime
import os

# Global state for tracking current operations
current_encoded_image = None
current_decoded_image = None
current_seed = None

def generate_lookup_table(width, height, seed):
    """
    Generate a shuffled position lookup table for encoding/decoding.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for deterministic shuffling (string or int)
    
    Returns:
        List of shuffled position indices
    """
    total_pixels = width * height
    
    # Create sequential list of positions
    positions = list(range(total_pixels))
    
    # Convert seed to deterministic integer hash if it's a string
    if isinstance(seed, str):
        # Use SHA-256 for deterministic hashing (works across sessions)
        hash_bytes = hashlib.sha256(seed.encode('utf-8')).digest()
        # Convert first 8 bytes to integer
        seed_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    else:
        seed_int = seed
    
    # Shuffle deterministically using seed
    rng = random.Random(seed_int)
    rng.shuffle(positions)
    
    return positions


def encode_image(input_image, seed, progress=gr.Progress()):
    """
    Encode an image by shuffling pixel positions according to seed.
    
    Args:
        input_image: PIL Image object
        seed: Random seed string for encoding
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (encoded_image, status_message)
    """
    global current_encoded_image, current_seed
    
    try:
        # Validate inputs
        if input_image is None:
            return None, "upload an image to scatter"
        
        if not seed or seed.strip() == "":
            return None, "enter a seed value"
        
        # Normalize seed (strip whitespace)
        seed = seed.strip()
        
        # Convert image to RGB mode if needed (handles RGBA, grayscale, etc.)
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Check image dimensions
        width, height = input_image.size
        total_pixels = width * height
        
        if width > 4096 or height > 4096:
            return None, f"Image too large. Max 4096 pixels. Your image is {width}x{height}"
        
        if total_pixels > 4096 * 4096:
            return None, f"Image has too many pixels. Max 16,777,216. Your image has {total_pixels}"
        
        progress(0.0, desc="Generating lookup table...")
        
        # Generate lookup table based on image dimensions
        lookup_table = generate_lookup_table(width, height, seed)
        
        progress(0.2, desc="Converting image to array...")
        
        # Convert image to numpy array
        img_array = np.array(input_image)
        
        # Create output array with same shape
        encoded_array = np.zeros_like(img_array)
        
        progress(0.3, desc="Scattering pixels...")
        
        # Encode: move pixel from position i to position lookup_table[i]
        update_interval = max(total_pixels // 100, 1000)
        
        for px in range(total_pixels):
            # Calculate original position
            ox = px % width
            oy = px // width
            
            # Calculate new position
            new_pos = lookup_table[px]
            nx = new_pos % width
            ny = new_pos // width
            
            # Copy pixel to new position
            encoded_array[ny, nx] = img_array[oy, ox]
            
            # Update progress
            if px % update_interval == 0:
                progress_val = 0.3 + (0.7 * px / total_pixels)
                progress(progress_val, desc=f"Scattering pixels... {int(100 * px / total_pixels)}%")
        
        progress(1.0, desc="Scattering complete")
        
        # Convert back to PIL Image
        encoded_image = Image.fromarray(encoded_array)
        
        # Store in global state
        current_encoded_image = encoded_image
        current_seed = seed
        
        message = (f"Successfully scattered {width}x{height} image ({total_pixels:,} pixels)\n"
                  f"Seed: '{seed}'\n"
                  f"Use the same seed to unscatter this image.")
        
        return encoded_image, message
        
    except Exception as e:
        return None, f"Error during scattering: {str(e)}"


def decode_image(input_image, seed, progress=gr.Progress()):
    """
    Unscatter an image by reversing the pixel position shuffle using the same seed.
    
    Args:
        input_image: PIL Image object (encoded image)
        seed: Random seed string (must match encoding seed)
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (decoded_image, status_message)
    """
    global current_decoded_image
    
    try:
        # Validate inputs
        if input_image is None:
            return None, "Please upload a scattered image to unscatter"
        
        if not seed or seed.strip() == "":
            return None, "Please enter the seed value used for scattering"
        
        # Normalize seed (strip whitespace)
        seed = seed.strip()
        
        # Convert image to RGB mode if needed (handles RGBA, grayscale, etc.)
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Check image dimensions
        width, height = input_image.size
        total_pixels = width * height
        
        if width > 4096 or height > 4096:
            return None, f"Image too large. Max is 4096 pixels. Your image is {width}x{height}"
        
        if total_pixels > 4096 * 4096:
            return None, f"Image has too many pixels. Max is 16,777,216. Your image has {total_pixels}"
        
        progress(0.0, desc="Generating lookup table...")
        
        # Generate same lookup table using same seed
        lookup_table = generate_lookup_table(width, height, seed)
        
        progress(0.2, desc="Converting image to array...")
        
        # Convert image to numpy array
        encoded_array = np.array(input_image)
        
        # Create output array with same shape
        decoded_array = np.zeros_like(encoded_array)
        
        progress(0.3, desc="Unscattering pixels...")
        
        # Unscatter: move pixel from position lookup_table[i] back to position i
        update_interval = max(total_pixels // 100, 1000)
        
        for px in range(total_pixels):
            # Calculate original position (where pixel should end up)
            ox = px % width
            oy = px // width
            
            # Calculate encoded position (where pixel currently is)
            encoded_pos = lookup_table[px]
            ex = encoded_pos % width
            ey = encoded_pos // width
            
            # Copy pixel from encoded position back to original position
            decoded_array[oy, ox] = encoded_array[ey, ex]
            
            # Update progress
            if px % update_interval == 0:
                progress_val = 0.3 + (0.7 * px / total_pixels)
                progress(progress_val, desc=f"Unscattering pixels... {int(100 * px / total_pixels)}%")
        
        progress(1.0, desc="Unscattering complete")
        
        # Convert back to PIL Image
        decoded_image = Image.fromarray(decoded_array)
        
        # Store in global state
        current_decoded_image = decoded_image
        
        message = (f"Successfully unscattered {width}x{height} image ({total_pixels:,} pixels)\n"
                  f"Seed: '{seed}'\n"
                  f"If the output looks correct, the seed matches the scattering seed.")
        
        return decoded_image, message
        
    except Exception as e:
        return None, f"Error during unscattering: {str(e)}"


def save_encoded_image():
    """Save the currently scattered image to disk with timestamp."""
    global current_encoded_image, current_seed
    
    if current_encoded_image is None:
        return "No scattered image to save. Please scatter an image first."
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scattered_{timestamp}.png"
        filepath = os.path.join(os.getcwd(), filename)
        
        current_encoded_image.save(filepath, "PNG")
        
        return f"Saved scattered image to: {filename}\nSeed used: '{current_seed}'"
        
    except Exception as e:
        return f"Error saving scattered image: {str(e)}"


def save_decoded_image():
    """Save the currently unscattered image to disk with timestamp."""
    global current_decoded_image
    
    if current_decoded_image is None:
        return "No decoded image to save. Please decode an image first."
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decoded_{timestamp}.png"
        filepath = os.path.join(os.getcwd(), filename)
        
        current_decoded_image.save(filepath, "PNG")
        
        return f"Saved decoded image to: {filename}"
        
    except Exception as e:
        return f"Error saving decoded image: {str(e)}"


def create_ui():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Image Hash Encoder/Decoder") as demo:
        gr.Markdown("# Image Hash Encoder/Decoder")
        gr.Markdown(
            "Encode images by shuffling pixel positions using a deterministic seed. "
            "Decode by providing the same seed. Supports images up to 4096x4096 pixels."
        )
        
        with gr.Tabs():
            # ENCODE TAB
            with gr.Tab("Encode"):
                gr.Markdown("### Encode an Image")
                gr.Markdown(
                    "Upload an image and enter a seed value. The image will be scrambled "
                    "based on the seed. Save the seed to decode later."
                )
                
                with gr.Row():
                    with gr.Column():
                        encode_input_image = gr.Image(
                            type="pil",
                            label="Input Image (Original)",
                            height=400
                        )
                        encode_seed = gr.Textbox(
                            label="Seed (Scattering Key)",
                            placeholder="Enter any text as seed (e.g., 'myseed123')",
                            value="",
                            info="Save this seed to unscatter the image later"
                        )
                        encode_button = gr.Button("Scatter Image", variant="primary")
                    
                    with gr.Column():
                        encode_output_image = gr.Image(
                            type="pil",
                            label="Scattered Image",
                            height=400
                        )
                        encode_status = gr.Textbox(
                            label="Status",
                            lines=4,
                            interactive=False
                        )
                        save_encode_button = gr.Button("Save Scattered Image")
                        save_encode_status = gr.Textbox(
                            label="Save Status",
                            lines=2,
                            interactive=False
                        )
                
                # Wire up encode functionality
                encode_button.click(
                    fn=encode_image,
                    inputs=[encode_input_image, encode_seed],
                    outputs=[encode_output_image, encode_status]
                )
                
                save_encode_button.click(
                    fn=save_encoded_image,
                    inputs=[],
                    outputs=[save_encode_status]
                )
            
            # DECODE TAB
            with gr.Tab("Decode"):
                gr.Markdown("### Decode an Image")
                gr.Markdown(
                    "Upload a scattered image and enter the same seed used for scattering. "
                    "The original image will be restored if the seed is correct."
                )
                
                with gr.Row():
                    with gr.Column():
                        decode_input_image = gr.Image(
                            type="pil",
                            label="Scattered Image",
                            height=400
                        )
                        decode_seed = gr.Textbox(
                            label="Seed (Unscattering Key)",
                            placeholder="Enter the same seed used for scattering",
                            value="",
                            info="Must match the seed used during scattering"
                        )
                        decode_button = gr.Button("Unscatter Image", variant="primary")
                    
                    with gr.Column():
                        decode_output_image = gr.Image(
                            type="pil",
                            label="Unscattered Image",
                            height=400
                        )
                        decode_status = gr.Textbox(
                            label="Status",
                            lines=4,
                            interactive=False
                        )
                        save_decode_button = gr.Button("Save Unscattered Image")
                        save_decode_status = gr.Textbox(
                            label="Save Status",
                            lines=2,
                            interactive=False
                        )
                
                # Wire up decode functionality
                decode_button.click(
                    fn=decode_image,
                    inputs=[decode_input_image, decode_seed],
                    outputs=[decode_output_image, decode_status]
                )
                
                save_decode_button.click(
                    fn=save_decoded_image,
                    inputs=[],
                    outputs=[save_decode_status]
                )
        
        gr.Markdown("---")
        gr.Markdown(
            "saves as png in the current directory with timestamped filename."
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(inbrowser=True)
