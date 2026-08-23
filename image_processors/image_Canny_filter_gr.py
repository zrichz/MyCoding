"""
Canny Edge Detector (Gradio)
Process an image and save results
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image

def calculate_auto_thresholds(input_image):
    """Calculate automatic thresholds based on image median intensity"""
    if input_image is None:
        return 50, 150
    
    try:
        # Convert PIL image to OpenCV format
        if isinstance(input_image, str):
            img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
        else:
            img_array = np.array(input_image)
            if len(img_array.shape) == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img = img_array
        
        if img is None:
            return 50, 150
        
        # Calculate median-based thresholds
        median_intensity = np.median(img)
        low_threshold = int(max(0, 0.66 * median_intensity))
        high_threshold = int(min(255, 1.33 * median_intensity))
        
        return low_threshold, high_threshold
        
    except Exception:
        return 50, 150

def process_single_image(input_image, low_threshold, high_threshold, overlay=False):
    """Process a single image with Canny edge detection"""
    if input_image is None:
        return None, "load an image first"
    
    try:
        # Convert PIL image to OpenCV format
        if isinstance(input_image, str):
            # If it's a file path
            img = cv2.imread(input_image)
        else:
            # If it's a PIL Image
            img_array = np.array(input_image)
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img = img_array
        
        if img is None:
            return None, "Could not read the uploaded image"
        
        # Store original for overlay
        original_img = img.copy()
        
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            greyimg = img
        
        # Apply blur,canny,invert,conversion:
        gaussian_blurred_image = cv2.GaussianBlur(greyimg, (5, 5), 0)
        edges = cv2.Canny(gaussian_blurred_image, int(low_threshold), int(high_threshold), apertureSize=3)
        inverted_edges = cv2.bitwise_not(edges)
        
        # Overlay on original if requested
        if overlay:
            # Convert original to RGB if needed
            if len(original_img.shape) == 3:
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            else:
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            
            # Create edge overlay (black edges on original)
            edge_mask = edges == 255
            original_rgb[edge_mask] = [0, 0, 0]
            result_image = Image.fromarray(original_rgb)
        else:
            result_image = Image.fromarray(inverted_edges)
        
        # Create success message
        success_msg = f"Done. Right-click on result image to save"
        
        return result_image, success_msg
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Canny Edge Detection Processor",
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .gr-form label, .gr-form p, .gr-form span {
        color: #000 !important;
    }
    .gr-box label {
        color: #000 !important;
    }
    """
) as app:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 10px;">
        <h1>Canny Edge Detection Processor</h1>
        <p style="font-size: 12px; color: #000;">
            <em>right-click to save</em>
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload
            in_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            
            # Parameter controls
            with gr.Group():
                gr.Markdown("# Canny Parameters")
                
                thr_low = gr.Slider(
                    minimum=1,
                    maximum=255,
                    value=50,
                    step=1,
                    label="Low Threshold",
                    info="detect more edges"
                )
                
                thr_hi = gr.Slider(
                    minimum=1,
                    maximum=255,
                    value=150,
                    step=1,
                    label="High Threshold",
                    info="detect less edges"
                )
                
                overlay_toggle = gr.Checkbox(
                    label="Overlay edges on original image",
                    value=False,
                    info="Show black edges on original instead of white background"
                )
            
            # Buttons
            with gr.Row():
                auto_btn = gr.Button(
                    "Auto Thresholds",
                    variant="secondary",
                    size="sm"
                )
                process_btn = gr.Button(
                    "Apply Canny Edge Detection",
                    variant="primary",
                    size="lg"
                )
        
        with gr.Column(scale=1):
            # Result image
            out_image = gr.Image(
                label="Result",
                height=500
            )
    
    # Auto threshold calculation
    def auto_thresholds_and_process(img, overlay):
        low, high = calculate_auto_thresholds(img)
        result = process_single_image(img, low, high, overlay)[0]
        return low, high, result
    
    auto_btn.click(
        fn=auto_thresholds_and_process,
        inputs=[in_image, overlay_toggle],
        outputs=[thr_low, thr_hi, out_image]
    )
    
    # Wire up the processing
    process_btn.click(
        fn=lambda img, low, high, overlay: process_single_image(img, low, high, overlay)[0],
        inputs=[in_image, thr_low, thr_hi, overlay_toggle],
        outputs=[out_image]
    )
    
    # Auto-process when sliders or toggle change
    def update_image(img, low, high, overlay):
        return process_single_image(img, low, high, overlay)[0]
    
    thr_low.change(
        fn=update_image,
        inputs=[in_image, thr_low, thr_hi, overlay_toggle],
        outputs=[out_image]
    )
    
    thr_hi.change(
        fn=update_image,
        inputs=[in_image, thr_low, thr_hi, overlay_toggle],
        outputs=[out_image]
    )
    
    overlay_toggle.change(
        fn=update_image,
        inputs=[in_image, thr_low, thr_hi, overlay_toggle],
        outputs=[out_image]
    )

def main():
    """Launch the Gradio app"""
    print("Starting Canny Edge Detection Processor...")
    print("Opening web interface...")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=None,  # Gradio finds an available port
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
