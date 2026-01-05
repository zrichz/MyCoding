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
        
        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Get model session
        session = get_session(model_name)
        
        # Remove background
        output = remove(
            pil_image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=post_process_mask
        )
        
        # Convert to numpy array for preview
        output_np = np.array(output)
        
        # Ensure RGBA format
        if output_np.shape[2] == 3:
            # Add alpha channel if not present
            alpha = np.ones((output_np.shape[0], output_np.shape[1], 1), dtype=np.uint8) * 255
            output_np = np.concatenate([output_np, alpha], axis=2)
        
        # Save to temporary PNG file
        temp_dir = tempfile.gettempdir()
        filename = f"bg_removed_{uuid.uuid4().hex}.png"
        temp_path = os.path.join(temp_dir, filename)
        output.save(temp_path, 'PNG')
        
        # Get statistics
        alpha_channel = output_np[:, :, 3]
        total_pixels = alpha_channel.size
        foreground_pixels = np.count_nonzero(alpha_channel > 0)
        background_pixels = total_pixels - foreground_pixels
        fg_percent = (foreground_pixels / total_pixels) * 100
        
        status = f"✓ Background removed successfully!\n"
        status += f"Model: {MODELS.get(model_name, model_name)}\n"
        status += f"Image size: {output_np.shape[1]}x{output_np.shape[0]}\n"
        status += f"Foreground: {fg_percent:.1f}% ({foreground_pixels:,} pixels)\n"
        status += f"Saved to: {temp_path}"
        
        return output_np, temp_path, status
        
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
                
                with gr.Row():
                    remove_btn = gr.Button("🗑️ Remove Background", variant="primary", size="lg")
                
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
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=6,
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
        4. **Click Remove Background** - Process the image
        5. **Download PNG** - Transparent background preserved
        
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
