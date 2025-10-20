"""
Canny Edge Detector (Gradio) - Apply Canny edge detection to individual images
Process single images with Canny edge detection and save results
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image

def process_single_image(input_image, low_threshold, high_threshold):
    """Process a single image with Canny edge detection"""
    if input_image is None:
        return None, "❌ Please upload an image first"
    
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
            return None, "❌ Could not read the uploaded image"
        
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, int(low_threshold), int(high_threshold), apertureSize=3)
        
        # Invert edges for black lines on white background
        inverted_edges = cv2.bitwise_not(edges)
        
        # Convert back to PIL Image for display
        result_image = Image.fromarray(inverted_edges)
        
        # Create success message
        success_msg = f"✅ **Canny Edge Detection Complete!**\n\n"
        success_msg += f"🔧 **Parameters Used:**\n"
        success_msg += f"   • Low Threshold: {int(low_threshold)}\n"
        success_msg += f"   • High Threshold: {int(high_threshold)}\n\n"
        success_msg += f"💡 **Tip:** Right-click on the result image to save it"
        
        return result_image, success_msg
        
    except Exception as e:
        return None, f"❌ **Error processing image:** {str(e)}"

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
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🖼️ Canny Edge Detection Processor</h1>
        <p style="font-size: 16px; color: #000;">
            <em>Results show black lines on white background - right-click to save</em>
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload
            input_image = gr.Image(
                label="📁 Upload Image",
                type="pil",
                height=400
            )
            
            # Parameter controls
            with gr.Group():
                gr.Markdown("### ⚙️ Canny Parameters")
                
                low_threshold = gr.Slider(
                    minimum=1,
                    maximum=255,
                    value=50,
                    step=1,
                    label="🔽 Low Threshold",
                    info="Lower values detect more edges"
                )
                
                high_threshold = gr.Slider(
                    minimum=1,
                    maximum=255,
                    value=150,
                    step=1,
                    label="🔼 High Threshold",
                    info="Higher values detect stronger edges"
                )
            
            # Process button
            process_btn = gr.Button(
                "🚀 Apply Canny Edge Detection",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            # Result image
            result_image = gr.Image(
                label="🎯 Canny Edge Detection Result",
                height=400,
                show_download_button=True
            )
    
    # Wire up the processing
    process_btn.click(
        fn=lambda img, low, high: process_single_image(img, low, high)[0],
        inputs=[input_image, low_threshold, high_threshold],
        outputs=[result_image]
    )
    
    # Auto-process when sliders change
    for slider in [low_threshold, high_threshold]:
        slider.change(
            fn=lambda img, low, high: process_single_image(img, low, high)[0],
            inputs=[input_image, low_threshold, high_threshold],
            outputs=[result_image]
        )

def main():
    """Launch the Gradio app"""
    print("🚀 Starting Canny Edge Detection Processor...")
    print("🌐 Opening web interface...")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=None,  # Let Gradio find an available port
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
