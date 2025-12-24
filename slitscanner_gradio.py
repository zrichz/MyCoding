"""
Slitscanner - Video to Image Processing Application (Gradio Version)

This application creates artistic images by extracting vertical pixel columns
from video frames and combining them horizontally into a single image.
"""

import cv2
import numpy as np
from PIL import Image
import gradio as gr
import os
import tempfile
import socket


def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:  # Port is available
                return port
        except Exception:
            continue
    
    # If no port found in range, let Gradio find one automatically
    return None


def process_video(video_file, sampling_method, n_frames):
    """
    Process the uploaded video and create slitscanned image
    
    Args:
        video_file: Uploaded video file
        sampling_method: Frame sampling method (every_frame, every_2, every_n)
        n_frames: Number for every_n sampling (3-1000)
    
    Returns:
        tuple: (processed_image, status_message)
    """
    if video_file is None:
        return None, "❌ Please upload a video file"
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            return None, "❌ Could not open video file. Please check the format."
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate center column
        center_x = frame_width // 2
        
        # Determine frame step based on sampling method
        if sampling_method == "Every Frame":
            frame_step = 1
        elif sampling_method == "Every 2 Frames":
            frame_step = 2
        elif sampling_method == "Every n Frames":
            try:
                frame_step = int(n_frames)
                if not (3 <= frame_step <= 1000):
                    return None, "❌ n frames value must be between 3 and 1000"
            except (ValueError, TypeError):
                return None, "❌ Please enter a valid number between 3 and 1000"
        else:
            frame_step = 1
        
        # Calculate how many frames we'll actually process
        frames_to_process = total_frames // frame_step
        max_width = 3000  # Limit output width to prevent memory issues
        
        if frames_to_process > max_width:
            frames_to_process = max_width
        
        if frames_to_process == 0:
            return None, "❌ No frames to process. Video might be too short or frame step too large."
        
        # Create output image array
        slitscanned_image = np.zeros((frame_height, frames_to_process, 3), dtype=np.uint8)
        
        frame_count = 0
        processed_count = 0
        
        # Process frames
        while cap.isOpened() and processed_count < frames_to_process:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check if we should process this frame
            if frame_count % frame_step == 0:
                # Extract center column
                center_column = frame[:, center_x, :]
                
                # Add to slitscanned image
                slitscanned_image[:, processed_count, :] = center_column
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        if processed_count == 0:
            return None, "❌ No frames were processed. Please check your video file."
        
        # Convert BGR to RGB
        slitscanned_image = cv2.cvtColor(slitscanned_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        output_image = Image.fromarray(slitscanned_image)
        
        # Create status message
        duration = total_frames / fps if fps > 0 else 0
        status_msg = f"✅ Success. Processed {processed_count} frames from {total_frames} total frames\n"
        status_msg += f"📹 Video: {frame_width}x{frame_height}px, {duration:.1f}s, {fps:.1f} FPS\n"
        status_msg += f"🖼️ Output: {processed_count}x{frame_height}px image\n"
        status_msg += f"⚙️ Sampling: Every {frame_step} frame(s)"
        
        return output_image, status_msg
        
    except Exception as e:
        return None, f"❌ Processing failed: {str(e)}"


def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:  # Port is available
                return port
        except Exception:
            continue
    
    # If no port found in range, let Gradio find one automatically
    return None


def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Slitscanner Video Processor"
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Slitscanner Video Processor</h1>
            <p>Transform videos into images by extracting and combining vertical pixel columns</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.HTML("<h3>📁 Input & Settings</h3>")
                
                video_input = gr.File(
                    label="Upload Video File",
                    file_types=["video"],
                    type="filepath"
                )
                
                gr.HTML("<br><h4>⚙️ Processing Options</h4>")
                
                sampling_method = gr.Radio(
                    choices=["Every Frame", "Every 2 Frames", "Every n Frames"],
                    value="Every Frame",
                    label="Frame Sampling Method",
                    info="Choose how often to sample frames from the video"
                )
                
                n_frames_input = gr.Number(
                    value=3,
                    minimum=3,
                    maximum=1000,
                    step=1,
                    label="N Frames (for 'Every n Frames' option)",
                    info="Extract every nth frame (3-1000)",
                    visible=False
                )
                
                # Show/hide N frames input based on selection
                def toggle_n_frames(method):
                    return gr.update(visible=(method == "Every n Frames"))
                
                sampling_method.change(
                    fn=toggle_n_frames,
                    inputs=[sampling_method],
                    outputs=[n_frames_input]
                )
                
                # Process button
                process_btn = gr.Button(
                    "🎯 Process Video",
                    variant="primary",
                    size="lg"
                )
                
                # Status output
                status_output = gr.Textbox(
                    label="Processing Status",
                    value="Ready to process video...",
                    interactive=False,
                    lines=4
                )
            
            with gr.Column(scale=2):
                # Output section
                gr.HTML("<h3>🖼️ Output Image</h3>")
                
                image_output = gr.Image(
                    label="Slitscanned Result",
                    type="pil",
                    height=400
                )
                
                # Download section
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 8px;">
                    <h4>💾 Download Options</h4>
                    <p>Click the download button above the image to save slitscan result</p>
                    <p><strong>Tip:</strong> The image will be saved in its full resolution.</p>
                </div>
                """)
        
        # Process button click handler
        process_btn.click(
            fn=process_video,
            inputs=[video_input, sampling_method, n_frames_input],
            outputs=[image_output, status_output]
        )
        
        # Examples section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 8px;">
            <h3>📖 How It Works</h3>
            <ul>
                <li><strong>Upload a video:</strong> Any common video format (MP4, AVI, MOV, etc.)</li>
                <li><strong>Choose sampling:</strong> How often to extract frames from your video</li>
                <li><strong>Processing:</strong> The app extracts the center vertical column from each sampled frame</li>
                <li><strong>Result:</strong> All columns are combined horizontally to create an image</li>
            </ul>

            
        </div>
        """)
    
    return interface


def main():
    """Main function to launch the application"""
    print("🎬 Starting Slitscanner Gradio Application...")
    
    # Find an available port
    available_port = find_available_port()
    
    interface = create_interface()
    
    # Launch the interface with improved port handling
    if available_port:
        print(f"🌐 Launching on port {available_port}")
        interface.launch(
            server_name="127.0.0.1",
            server_port=available_port,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,  # Automatically open browser
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .main-header {
                text-align: center;
                margin-bottom: 30px;
            }
            .status-box {
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
            """
        )
    else:
        print("🌐 Letting Gradio find an available port automatically")
        interface.launch(
            server_name="127.0.0.1",
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,  # Automatically open browser
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .main-header {
                text-align: center;
                margin-bottom: 30px;
            }
            .status-box {
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
            """
        )


if __name__ == "__main__":
    main()
