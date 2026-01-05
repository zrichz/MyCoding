"""
Background Remover - High Quality AI Background Removal

Uses rembg with U^2-Net model for accurate background removal with alpha transparency.
Supports various models and post-processing options for optimal results.
"""

import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import tempfile
import uuid
from rembg import remove, new_session

# Available models in rembg
# 
# Model Selection Guide:
# - u2net: General purpose, best overall. Good for people, objects, animals, products.
#          Balanced quality and speed. Use when unsure. (~176MB)
# 
# - u2netp: Portrait/faster. Lightweight version optimized for human portraits.
#           Much faster processing, slightly less accurate on complex edges. (~4.7MB)
#           Best for: Profile pictures, headshots, simple portraits, batch processing
# 
# - u2net_human_seg: Human segmentation only. Specifically trained for people.
#                    Better at handling different poses and clothing. Ignores non-human objects.
#                    Best for: Fashion photography, fitness photos, full-body portraits
# 
# - u2net_cloth_seg: Clothing segmentation. Isolates garments specifically.
#                    Removes person AND background, keeps only clothes.
#                    Best for: Fashion catalogs, e-commerce clothing, garment isolation
# 
# - silueta: Alternative general purpose. Different architecture than u2net.
#            Try if u2net doesn't work well. Good backup option.
# 
# - isnet-general-use: Highest quality general model. Latest architecture.
#                      More accurate edge detection, better fine details (hair, fur).
#                      Slower than u2net but best results.
#                      Best for: Professional portraits, high-quality products, marketing materials
# 
# - isnet-anime: Anime/cartoon specialist. Trained on illustrated/drawn characters.
#                Handles cel-shaded art style. Better on stylized/non-photorealistic images.
#                Best for: Anime screenshots, manga panels, cartoon characters, illustrated art
# 
# Quick reference: Speed? → u2netp | Quality? → isnet-general-use | Person? → u2net_human_seg
#                  Anime? → isnet-anime | Not sure? → u2net | Clothing only? → u2net_cloth_seg

MODELS = {
    "u2net": "U^2-Net (General purpose, best overall)",
    "u2netp": "U^2-Net Portrait (Faster, lighter)",
    "u2net_human_seg": "U^2-Net Human Segmentation (People only)",
    "u2net_cloth_seg": "U^2-Net Cloth Segmentation (Clothing)",
    "silueta": "Silueta (General purpose, alternative)",
    "isnet-general-use": "IS-Net General (High quality, slower)",
    "isnet-anime": "IS-Net Anime (Anime/cartoon characters)"
}

# Global session cache
session_cache = {}

def get_session(model_name):
    """Get or create a model session (cached for performance)"""
    if model_name not in session_cache:
        print(f"Loading model: {model_name}...")
        session_cache[model_name] = new_session(model_name)
    return session_cache[model_name]


def remove_background(image, model_name="u2net", alpha_matting=True, 
                     alpha_matting_foreground_threshold=240,
                     alpha_matting_background_threshold=10,
                     alpha_matting_erode_size=10,
                     post_process_mask=False):
    """
    Remove background from image using rembg
    
    Args:
        image: Input image (PIL Image or numpy array)
        model_name: Model to use for segmentation
        alpha_matting: Enable alpha matting for better edge quality
        alpha_matting_foreground_threshold: Foreground threshold (0-255)
        alpha_matting_background_threshold: Background threshold (0-255)
        alpha_matting_erode_size: Erosion size for matting
        post_process_mask: Apply morphological operations to clean mask
    
    Returns:
        (preview_rgba_numpy, saved_png_path, status_message)
    """
    if image is None:
        return None, None, "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Ensure RGB mode and preserve original
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Store original RGB values for exact reconstruction
        original_np = np.array(pil_image)
        
        # Get model session
        session = get_session(model_name)
        
        # Remove background to get the mask
        output = remove(
            pil_image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=post_process_mask
        )
        
        # Convert to numpy array
        output_np = np.array(output)
        
        # Extract the alpha mask from rembg output
        if output_np.shape[2] == 4:
            alpha_mask = output_np[:, :, 3]
        else:
            alpha_mask = np.ones((output_np.shape[0], output_np.shape[1]), dtype=np.uint8) * 255
        
        # Create RGBA with EXACT original RGB + rembg mask
        # This ensures foreground + background layers will reconstruct perfectly
        output_rgba = np.zeros((original_np.shape[0], original_np.shape[1], 4), dtype=np.uint8)
        output_rgba[:, :, :3] = original_np  # Exact original RGB values
        output_rgba[:, :, 3] = alpha_mask    # rembg mask
        
        # Save to temporary PNG file with original RGB preserved
        temp_dir = tempfile.gettempdir()
        filename = f"bg_removed_{uuid.uuid4().hex}.png"
        temp_path = os.path.join(temp_dir, filename)
        Image.fromarray(output_rgba).save(temp_path, 'PNG')
        
        # Get statistics
        total_pixels = alpha_mask.size
        foreground_pixels = np.count_nonzero(alpha_mask > 0)
        background_pixels = total_pixels - foreground_pixels
        fg_percent = (foreground_pixels / total_pixels) * 100
        
        status = f"✓ Background removed successfully!\n"
        status += f"Model: {MODELS.get(model_name, model_name)}\n"
        status += f"Image size: {output_rgba.shape[1]}x{output_rgba.shape[0]}\n"
        status += f"Foreground: {fg_percent:.1f}% ({foreground_pixels:,} pixels)\n"
        status += f"Note: Original RGB values preserved exactly\n"
        status += f"Saved to: {temp_path}"
        
        return output_rgba, temp_path, status
        
    except Exception as e:
        import traceback
        error_msg = f"Error removing background: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


def replace_background(image, model_name, bg_color_hex, alpha_matting,
                       fg_threshold, bg_threshold, erode_size, post_process):
    """Remove background and replace with solid color"""
    if image is None:
        return None, None, "Please upload an image"
    
    # First remove background
    result_np, _, status = remove_background(
        image, model_name, alpha_matting, fg_threshold, bg_threshold, erode_size, post_process
    )
    
    if result_np is None:
        return None, None, status
    
    try:
        # Parse hex color
        hex_color = bg_color_hex.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create background with chosen color
        h, w = result_np.shape[:2]
        background = np.ones((h, w, 3), dtype=np.uint8)
        background[:, :] = [r, g, b]
        
        # Alpha blend
        alpha = result_np[:, :, 3:4] / 255.0
        foreground = result_np[:, :, :3]
        
        blended = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        
        # Save to temp file
        temp_dir = tempfile.gettempdir()
        filename = f"bg_replaced_{uuid.uuid4().hex}.png"
        temp_path = os.path.join(temp_dir, filename)
        Image.fromarray(blended).save(temp_path, 'PNG')
        
        status_new = status.replace("Background removed", "Background replaced")
        status_new += f"\nBackground color: {bg_color_hex}"
        
        return blended, temp_path, status_new
        
    except Exception as e:
        import traceback
        error_msg = f"Error replacing background: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


def save_foreground_only(image, model_name, alpha_matting, fg_threshold, 
                         bg_threshold, erode_size, post_process):
    """
    Save foreground with alpha channel (transparent background)
    This is the same as remove_background but with clearer naming
    """
    return remove_background(image, model_name, alpha_matting, fg_threshold, 
                           bg_threshold, erode_size, post_process)


def save_background_only(image, model_name, alpha_matting, fg_threshold,
                         bg_threshold, erode_size, post_process):
    """
    Save background with alpha channel (transparent foreground)
    The RGB values are preserved exactly from the original image.
    When composited with foreground in Photoshop, will reproduce the exact original image.
    """
    if image is None:
        return None, None, "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Store original image in RGB mode for exact pixel preservation
        if pil_image.mode == 'RGBA':
            # If original has alpha, keep RGB channels exactly
            original_rgba = pil_image
            pil_image_rgb = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image_rgb = pil_image.convert('RGB')
            original_rgba = None
        else:
            pil_image_rgb = pil_image
            original_rgba = None
        
        # Get model session
        session = get_session(model_name)
        
        # Remove background to get the foreground mask
        output = remove(
            pil_image_rgb,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            alpha_matting_erode_size=erode_size,
            post_process_mask=post_process
        )
        
        # Convert to numpy
        output_np = np.array(output)
        
        # Get alpha channel (foreground mask)
        if output_np.shape[2] == 4:
            foreground_mask = output_np[:, :, 3]
        else:
            foreground_mask = np.ones((output_np.shape[0], output_np.shape[1]), dtype=np.uint8) * 255
        
        # Invert mask to get background mask (0=foreground/transparent, 255=background/opaque)
        background_mask = 255 - foreground_mask
        
        # Get EXACT original image pixels (no conversion artifacts)
        original_np = np.array(pil_image_rgb)
        
        # Create RGBA image with background only
        # RGB channels: EXACT original pixels (no modification whatsoever)
        # Alpha channel: inverted mask (0 where foreground was, 255 where background is)
        background_rgba = np.zeros((original_np.shape[0], original_np.shape[1], 4), dtype=np.uint8)
        background_rgba[:, :, :3] = original_np  # Exact original RGB values
        background_rgba[:, :, 3] = background_mask  # Inverted mask
        
        # Save to temporary PNG file
        temp_dir = tempfile.gettempdir()
        filename = f"bg_only_{uuid.uuid4().hex}.png"
        temp_path = os.path.join(temp_dir, filename)
        Image.fromarray(background_rgba).save(temp_path, 'PNG')
        
        # Get statistics
        total_pixels = background_mask.size
        background_pixels = np.count_nonzero(background_mask > 0)
        foreground_pixels = total_pixels - background_pixels
        bg_percent = (background_pixels / total_pixels) * 100
        
        status = f"✓ Background extracted successfully!\n"
        status += f"Model: {MODELS.get(model_name, model_name)}\n"
        status += f"Image size: {background_rgba.shape[1]}x{background_rgba.shape[0]}\n"
        status += f"Background: {bg_percent:.1f}% ({background_pixels:,} pixels)\n"
        status += f"Foreground: {100-bg_percent:.1f}% (transparent)\n"
        status += f"Note: Original RGB values preserved exactly\n"
        status += f"Saved to: {temp_path}"
        
        return background_rgba, temp_path, status
        
    except Exception as e:
        import traceback
        error_msg = f"Error extracting background: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


def save_both_layers(image, model_name, alpha_matting, fg_threshold,
                     bg_threshold, erode_size, post_process):
    """
    Save both foreground and background as separate files
    Returns paths to both files
    """
    if image is None:
        return None, None, None, None, "Please upload an image"
    
    # Get foreground
    fg_np, fg_path, fg_status = save_foreground_only(
        image, model_name, alpha_matting, fg_threshold, 
        bg_threshold, erode_size, post_process
    )
    
    if fg_np is None:
        return None, None, None, None, fg_status
    
    # Get background
    bg_np, bg_path, bg_status = save_background_only(
        image, model_name, alpha_matting, fg_threshold,
        bg_threshold, erode_size, post_process
    )
    
    if bg_np is None:
        return None, None, None, None, bg_status
    
    status = "✓ Both layers saved successfully!\n\n"
    status += "FOREGROUND:\n" + "\n".join(fg_status.split("\n")[1:]) + "\n\n"
    status += "BACKGROUND:\n" + "\n".join(bg_status.split("\n")[1:])
    
    return fg_np, fg_path, bg_np, bg_path, status


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="AI Background Remover") as app:
        gr.Markdown("# 🎨 AI Background Remover")
        gr.Markdown("High-quality background removal using state-of-the-art AI models (rembg/U^2-Net)")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )
                
                gr.Markdown("### Model Selection")
                
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="u2net",
                    label="AI Model",
                    info="Choose model based on your image type"
                )
                
                model_info = gr.Markdown(f"**{MODELS['u2net']}**")
                
                gr.Markdown("### Processing Options")
                
                alpha_matting_check = gr.Checkbox(
                    label="Enable Alpha Matting",
                    value=True,
                    info="Better edge quality, especially for hair/fur (slower)"
                )
                
                with gr.Accordion("Advanced Alpha Matting Settings", open=False):
                    fg_threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        value=240,
                        step=1,
                        label="Foreground Threshold",
                        info="Higher = more aggressive foreground detection"
                    )
                    
                    bg_threshold = gr.Slider(
                        minimum=0,
                        maximum=255,
                        value=10,
                        step=1,
                        label="Background Threshold",
                        info="Lower = more aggressive background removal"
                    )
                    
                    erode_size = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Erode Size",
                        info="Edge refinement size"
                    )
                
                post_process_check = gr.Checkbox(
                    label="Post-Process Mask",
                    value=False,
                    info="Clean up mask with morphological operations"
                )
                
                gr.Markdown("### Actions")
                
                with gr.Row():
                    remove_btn = gr.Button("🗑️ Remove Background", variant="primary", size="lg")
                
                with gr.Row():
                    fg_only_btn = gr.Button("📄 Save Foreground Only", variant="secondary")
                    bg_only_btn = gr.Button("🖼️ Save Background Only", variant="secondary")
                
                with gr.Row():
                    both_btn = gr.Button("📑 Save Both Layers", variant="secondary", size="lg")
                
                gr.Markdown("### Replace Background (Optional)")
                
                bg_color = gr.ColorPicker(
                    label="Background Color",
                    value="#FFFFFF",
                    info="Choose color to replace background"
                )
                
                replace_btn = gr.Button("🎨 Replace Background", variant="secondary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                
                output_preview = gr.Image(
                    label="Result (PNG with Alpha)",
                    type="numpy",
                    height=400
                )
                
                output_file = gr.File(
                    label="Download PNG"
                )
                
                gr.Markdown("### Additional Output (for Both Layers)")
                
                output_preview_2 = gr.Image(
                    label="Second Layer (when saving both)",
                    type="numpy",
                    height=300
                )
                
                output_file_2 = gr.File(
                    label="Download Second Layer"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False
                )
        
        gr.Markdown("""
        ### Instructions
        1. **Upload an image** - Any format works (JPG, PNG, etc.)
        2. **Choose a model** - Select based on your subject:
           - **u2net**: Best for general use (people, objects, animals) - Default choice
           - **u2netp**: Faster, lighter version (good for portraits, batch processing)
           - **u2net_human_seg**: Optimized for people (fashion, fitness, full-body)
           - **u2net_cloth_seg**: Isolates clothing only (e-commerce garments)
           - **silueta**: Alternative general model (try if u2net fails)
           - **isnet-general-use**: Highest quality (best for professional work, slower)
           - **isnet-anime**: Anime/cartoon specialist (manga, illustrated characters)
        3. **Enable Alpha Matting** - For better edge quality (hair, fur, fine details)
        4. **Choose an action**:
           - **Remove Background**: Standard foreground extraction with transparent background
           - **Save Foreground Only**: Same as above (PNG with alpha channel)
           - **Save Background Only**: Extract background, make foreground transparent
           - **Save Both Layers**: Get both foreground and background as separate PNG files
           - **Replace Background**: Replace with solid color (no transparency)
        5. **Download PNG** - Transparent backgrounds preserved
        
        ### Output Formats
        - **Foreground Only**: Subject with transparent background (standard use case)
        - **Background Only**: Background with transparent subject area (for compositing)
        - **Both Layers**: Two separate files - foreground + background (for maximum flexibility)
        - All outputs include alpha channels for proper transparency
        - **IMPORTANT**: Foreground + Background layers use EXACT original RGB values
        - When composited together in Photoshop (Normal blend mode), they reproduce the original image EXACTLY
        - No quality loss or color shift - perfect pixel-perfect reconstruction guaranteed
        
        ### Model Selection Quick Guide
        - **Need speed?** → u2netp
        - **Best quality?** → isnet-general-use
        - **Person in photo?** → u2net_human_seg
        - **Cartoon/anime?** → isnet-anime
        - **Not sure?** → u2net (default)
        - **Product/object?** → u2net or isnet-general-use
        - **Just clothing?** → u2net_cloth_seg
        
        ### Tips
        - **For portraits**: Use u2netp or u2net_human_seg with alpha matting
        - **For products**: Use u2net with alpha matting disabled (faster)
        - **For detailed edges**: Enable alpha matting and adjust thresholds
        - **Slow processing?**: Try u2netp or disable alpha matting
        - **Complex backgrounds**: Use isnet-general-use with alpha matting
        - **Layer compositing**: Use "Save Both Layers" to get separate foreground/background
        
        ### Models Download
        Models are downloaded automatically on first use (~4.7MB to 176MB per model).
        Subsequent uses are fast as models are cached in: `~/.u2net/`
        """)
        
        # Event handlers
        def update_model_info(model_name):
            return f"**{MODELS.get(model_name, model_name)}**"
        
        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        remove_btn.click(
            fn=remove_background,
            inputs=[
                image_input, model_dropdown, alpha_matting_check,
                fg_threshold, bg_threshold, erode_size, post_process_check
            ],
            outputs=[output_preview, output_file, status_output]
        )
        
        fg_only_btn.click(
            fn=save_foreground_only,
            inputs=[
                image_input, model_dropdown, alpha_matting_check,
                fg_threshold, bg_threshold, erode_size, post_process_check
            ],
            outputs=[output_preview, output_file, status_output]
        )
        
        bg_only_btn.click(
            fn=save_background_only,
            inputs=[
                image_input, model_dropdown, alpha_matting_check,
                fg_threshold, bg_threshold, erode_size, post_process_check
            ],
            outputs=[output_preview, output_file, status_output]
        )
        
        both_btn.click(
            fn=save_both_layers,
            inputs=[
                image_input, model_dropdown, alpha_matting_check,
                fg_threshold, bg_threshold, erode_size, post_process_check
            ],
            outputs=[output_preview, output_file, output_preview_2, output_file_2, status_output]
        )
        
        replace_btn.click(
            fn=replace_background,
            inputs=[
                image_input, model_dropdown, bg_color, alpha_matting_check,
                fg_threshold, bg_threshold, erode_size, post_process_check
            ],
            outputs=[output_preview, output_file, status_output]
        )
    
    return app


if __name__ == "__main__":
    print("=" * 60)
    print("AI Background Remover")
    print("=" * 60)
    print("Models available:", ", ".join(MODELS.keys()))
    print("\nNote: Models will be downloaded on first use (~170MB)")
    print("=" * 60)
    
    app = create_interface()
    app.queue(max_size=10)
    app.launch(
        inbrowser=True,
        show_error=True,
        share=False
    )
