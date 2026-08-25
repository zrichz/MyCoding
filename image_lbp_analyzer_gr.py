import gradio as gr
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern


def compute_lbp(image, P, R, method):
    """
    Compute Local Binary Pattern of an image.
    
    Args:
        image: PIL Image or numpy array
        P: Number of sampling points
        R: Radius of circle
        method: LBP method ('default', 'uniform', 'var', 'nri_uniform', 'ror')
    
    Returns:
        grayscale_pil: PIL Image of grayscale version
        lbp_pil: PIL Image of LBP texture
        info_message: Analysis information
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = rgb2gray(image)
    
    # Compute LBP
    lbp_image = local_binary_pattern(image, P, R, method)
    
    # Calculate histogram
    n_bins = int(lbp_image.max() + 1)
    
    # Convert grayscale image to PIL (0-255 range)
    gray_8bit = (image * 255).astype(np.uint8)
    grayscale_pil = Image.fromarray(gray_8bit, mode='L')
    
    # Normalize LBP image to 0-255 range and convert to PIL
    lbp_normalized = ((lbp_image - lbp_image.min()) / (lbp_image.max() - lbp_image.min()) * 255).astype(np.uint8)
    lbp_pil = Image.fromarray(lbp_normalized, mode='L')
    
    # Create info message
    info_message = (
        f"LBP Analysis Complete\n"
        f"Parameters: P={P}, R={R}, Method={method}\n"
        f"Number of unique patterns: {n_bins}\n"
        f"Image size: {image.shape[1]}x{image.shape[0]} pixels"
    )
    
    return grayscale_pil, lbp_pil, info_message


def process_image(image, P, R, method):
    """Wrapper function for Gradio interface."""
    if image is None:
        return None, None, "Please upload an image first"
    
    try:
        gray_img, lbp_img, info = compute_lbp(image, P, R, method)
        return gray_img, lbp_img, info
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="LBP Image Analyzer") as demo:
    gr.Markdown("# Local Binary Pattern")
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            input_image = gr.Image(
                label="get img",
                type="pil",
                height=200
            )
            
            gr.Markdown("params")
            
            P = gr.Slider(
                minimum=4,
                maximum=32,
                value=8,
                step=4,
                label="P (Number of sampling points)",
                
            )
            
            R = gr.Slider(
                minimum=1,
                maximum=15,
                value=1,
                step=1,
                label="R (radius (pixels))",
                
            )
            
            method = gr.Dropdown(
                choices=['uniform', 'default', 'ror', 'nri_uniform', 'var'],
                value='uniform',
                label="LBP Method",
                
            )
            
            process_btn = gr.Button("analyze", variant="primary")
        
        with gr.Column(scale=3):
            # Output displays
            with gr.Row():
                output_grayscale = gr.Image(label="Grayscale", type="pil")
                output_lbp = gr.Image(label="LBP Texture", type="pil")
            output_info = gr.Textbox(
                label="Info",
                lines=4,
                interactive=False
            )
    
    # Event handlers
    process_btn.click(
        fn=process_image,
        inputs=[input_image, P, R, method],
        outputs=[output_grayscale, output_lbp, output_info]
    )
    
    # Auto-process on parameter change
    for component in [P, R, method]:
        component.change(
            fn=process_image,
            inputs=[input_image, P, R, method],
            outputs=[output_grayscale, output_lbp, output_info]
        )
    
    


if __name__ == "__main__":
    demo.launch(inbrowser=True)
