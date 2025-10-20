"""
Canny Batch Processor (Gradio) - Apply Canny edge detection to all images in a directory
Processes all images in a selected directory and saves Canny edge detection results
"""

import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path

def process_canny_batch(input_folder, low_threshold, high_threshold, progress=gr.Progress()):
    """Process all images in folder with Canny edge detection"""
    if not input_folder:
        return "❌ Please select an input folder", None, None
    
    try:
        # Handle folder path (Gradio returns file object or path)
        if hasattr(input_folder, 'name'):
            folder_path = input_folder.name
        else:
            folder_path = str(input_folder)
        
        if not os.path.isdir(folder_path):
            return "❌ Selected path is not a valid directory", None, None
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.tif'}
        image_files = []
        
        for file_path in Path(folder_path).iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            return "❌ No image files found in the selected folder", None, None
        
        # Create output directory
        output_dir = Path(folder_path) / "canny"
        output_dir.mkdir(exist_ok=True)
        
        processed = 0
        errors = 0
        error_details = []
        
        # Process each image with progress tracking
        for i, image_path in enumerate(image_files):
            progress((i+1)/len(image_files), f"Processing {image_path.name} ({i+1}/{len(image_files)})")
            
            try:
                # Read image
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"Could not read image: {image_path.name}")
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply Canny edge detection
                edges = cv2.Canny(blurred, int(low_threshold), int(high_threshold), apertureSize=3)
                
                # Invert edges (black lines on white background)
                inverted_edges = cv2.bitwise_not(edges)
                
                # Create output filename
                output_filename = f"{image_path.stem}_canny.png"
                output_path = output_dir / output_filename
                
                # Save result
                success = cv2.imwrite(str(output_path), inverted_edges)
                if not success:
                    raise ValueError(f"Failed to save processed image")
                
                processed += 1
                
            except Exception as e:
                errors += 1
                error_details.append(f"❌ {image_path.name}: {str(e)}")
                continue
        
        # Generate results message
        result_msg = f"✅ Processing Complete!\n\n"
        result_msg += f"📊 Summary:\n"
        result_msg += f"• Total images found: {len(image_files)}\n"
        result_msg += f"• Successfully processed: {processed}\n"
        
        if errors > 0:
            result_msg += f"• Errors: {errors}\n\n"
            result_msg += f"📝 Error Details:\n" + "\n".join(error_details[:5])  # Show first 5 errors
            if len(error_details) > 5:
                result_msg += f"\n... and {len(error_details) - 5} more errors"
        
        result_msg += f"\n\n📁 Output Location:\n{output_dir}"
        result_msg += f"\n\n💡 Results are inverted (black lines on white background)"
        
        # Return the output directory path for download
        return result_msg, str(output_dir), f"Processed {processed}/{len(image_files)} images"
        
    except Exception as e:
        return f"❌ Fatal Error: {str(e)}", None, None

def check_dependencies():
    """Check if required packages are available"""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")
    
    if missing:
        return f"❌ Missing required packages: {', '.join(missing)}\n\nInstall with: pip install {' '.join(missing)}"
    
    return "✅ All dependencies available"

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Canny Batch Processor",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 900px !important;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🎨 Canny Edge Detection Batch Processor
        
        Process all images in a folder with Canny edge detection. Results are saved as inverted images (black lines on white background) in a 'canny' subdirectory.
        """)
        
        # Dependency check
        with gr.Row():
            dependency_status = gr.Textbox(
                value=check_dependencies(),
                label="🔧 System Status",
                lines=2,
                interactive=False
            )
        
        with gr.Row():
            with gr.Column():
                # Folder selection
                input_folder = gr.File(
                    file_types=["directory"],
                    label="📁 Select Input Folder",
                    file_count="single"
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
                    "🚀 Process All Images",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                # Results display
                output_msg = gr.Textbox(
                    label="📋 Processing Results",
                    lines=12,
                    placeholder="Select a folder and click 'Process All Images' to begin...",
                    interactive=False
                )
                
                # Quick status
                status_msg = gr.Textbox(
                    label="📊 Quick Status",
                    lines=1,
                    interactive=False
                )
        
        # Output folder info
        with gr.Row():
            output_info = gr.Textbox(
                label="📂 Output Directory",
                placeholder="Output directory path will appear here after processing",
                interactive=False
            )
        
        # Wire up the processing
        process_btn.click(
            fn=process_canny_batch,
            inputs=[input_folder, low_threshold, high_threshold],
            outputs=[output_msg, output_info, status_msg],
            show_progress=True
        )
        
        # Add some helpful information
        with gr.Accordion("ℹ️ Usage Instructions", open=False):
            gr.Markdown("""
            ### How to Use:
            1. **Select Folder**: Click "Select Input Folder" and choose a directory containing images
            2. **Adjust Parameters**: 
               - **Low Threshold**: Lower values (20-50) detect more edges, including weak ones
               - **High Threshold**: Higher values (100-200) focus on strong edges only
            3. **Process**: Click "Process All Images" to start batch processing
            4. **Results**: Processed images will be saved in a 'canny' subdirectory
            
            ### Supported Formats:
            JPG, JPEG, PNG, BMP, TIFF, GIF, WebP
            
            ### Output:
            - **Inverted edges**: Black lines on white background (ready for further processing)
            - **PNG format**: Lossless quality preservation
            - **Same resolution**: No resizing applied
            """)
    
    return app

# Launch the application
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
