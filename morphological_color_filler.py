import cv2
import numpy as np
import gradio as gr
from PIL import Image

def morphological_color_fill(image, kernel_size=5, edge_expansion=0, smoothing=0):
    """
    Fill black areas in an image using morphological dilation of surrounding colors.
    
    Args:
        image: Input image (PIL Image or numpy array)
        kernel_size: Size of the structuring element for dilation
        edge_expansion: Number of pixels to expand the black mask (helps remove dark edges)
        smoothing: Gaussian blur radius applied during filling (0 = no smoothing)
    
    Returns:
        Filled image as PIL Image
    """
    # Convert PIL Image to numpy array (RGB to BGR for OpenCV)
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # Create mask of black pixels
    mask = np.all(img == [0, 0, 0], axis=2).astype(np.uint8)
    
    # Expand the mask to include nearly-black edge pixels
    if edge_expansion > 0:
        expansion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                      (edge_expansion * 2 + 1, edge_expansion * 2 + 1))
        mask = cv2.dilate(mask, expansion_kernel)
    
    # Check if there are any black pixels
    if not np.any(mask):
        # No black pixels, return original
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Copy image to fill
    filled = img.copy()
    
    # Store original non-black pixels to preserve them
    original_mask = (mask == 0).astype(np.uint8)
    
    # Repeat dilation until no black remains (with safety limit)
    max_iterations = 1000
    iteration = 0
    
    while np.any(mask) and iteration < max_iterations:
        # Dilate the color regions
        dilated = cv2.dilate(filled, kernel)
        
        # Only update black pixels
        filled[mask == 1] = dilated[mask == 1]
        
        # Recompute mask
        mask = np.all(filled == [0, 0, 0], axis=2).astype(np.uint8)
        
        iteration += 1
    
    # Apply smoothing once at the end if enabled (helps reduce banding)
    if smoothing > 0:
        # Blur the entire filled image
        blurred = cv2.GaussianBlur(filled, (smoothing * 2 + 1, smoothing * 2 + 1), 0)
        # Keep original pixels untouched, only smooth the filled areas
        filled = np.where(original_mask[:, :, np.newaxis] == 1, img, blurred)
    
    # Convert back to RGB for display
    filled_rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(filled_rgb)


# Create Gradio interface
with gr.Blocks(title="Morphological Color Filler") as demo:
    gr.Markdown("""
    # Morphological Color Filler
    
    Upload an image with black areas, and this tool will fill them by dilating surrounding colors.
    Perfect for filling holes or gaps in images.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=500)
            kernel_size = gr.Slider(
                minimum=3,
                maximum=15,
                step=2,
                value=11,
                label="Kernel Size (larger = faster filling, less detail)"
            )
            edge_expansion = gr.Slider(
                minimum=0,
                maximum=5,
                step=1,
                value=3,
                label="Edge Expansion (removes dark edge pixels)"
            )
            smoothing = gr.Slider(
                minimum=0,
                maximum=30,
                step=1,
                value=7,
                label="Smoothing (reduces banding, higher = smoother)"
            )
            process_btn = gr.Button("Fill Black Areas", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Filled Image", height=500)
            
            gr.Markdown("""
            ### How it works:
            - Black pixels (RGB: 0,0,0) are identified
            - **Edge Expansion** dilates the mask to include nearly-black edge pixels (prevents dark fringing)
            - **Smoothing** applies Gaussian blur at the end to reduce banding artifacts
            - Colors from surrounding pixels are gradually expanded into black areas
            - The process repeats until all black areas are filled
            - Adjust kernel size for speed vs. detail trade-off
            """)
    
    # Connect the button to the function
    process_btn.click(
        fn=morphological_color_fill,
        inputs=[input_image, kernel_size, edge_expansion, smoothing],
        outputs=output_image
    )
    
    # Also allow processing on image upload
    input_image.change(
        fn=morphological_color_fill,
        inputs=[input_image, kernel_size, edge_expansion, smoothing],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
