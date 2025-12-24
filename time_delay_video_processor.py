"""
Time-Delay Video Processor - Gradio Application

This application creates artistic videos by applying time delays to each pixel
based on the luminance of a control image. Brighter areas in the control image
cause longer delays, creating complex temporal distortions in the output video.
"""

import cv2
import numpy as np
from PIL import Image
import gradio as gr
import os
import tempfile
import socket
from collections import deque


def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:  # Port is available
                return port
        except Exception:
            continue
    
    return None


def process_video_with_delay(video_file, control_image, max_delay_ms, output_fps):
    """
    Process video with time delays based on control image luminance
    
    Args:
        video_file: Uploaded video file
        control_image: Control image (PIL Image or file path)
        max_delay_ms: Maximum delay in milliseconds for white pixels
        output_fps: Output video frame rate
    
    Returns:
        tuple: (output_video_path, status_message)
    """
    if video_file is None:
        return None, "❌ Please upload a video file"
    
    if control_image is None:
        return None, "❌ Please upload a control image"
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, "❌ Error: Could not open video file"
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0:
            cap.release()
            return None, "❌ Error: Video has no frames"
        
        status_msg = f"📹 Video: {width}x{height}, {total_frames} frames @ {input_fps:.2f} fps\n"
        
        # Load and process control image
        if isinstance(control_image, str):
            ctrl_img = Image.open(control_image)
        else:
            ctrl_img = control_image
        
        # Convert to grayscale and resize to match video dimensions
        ctrl_img = ctrl_img.convert('L')
        ctrl_img = ctrl_img.resize((width, height), Image.Resampling.LANCZOS)
        ctrl_array = np.array(ctrl_img, dtype=np.float32) / 255.0  # Normalize to 0-1
        
        status_msg += f"🖼️ Control image scaled to {width}x{height}\n"
        status_msg += f"⏱️ Maximum delay: {max_delay_ms}ms\n"
        
        # Calculate delay in frames for each pixel
        # max_delay_ms corresponds to white (1.0), 0ms to black (0.0)
        max_delay_frames = int((max_delay_ms / 1000.0) * input_fps)
        delay_map = (ctrl_array * max_delay_frames).astype(np.int32)
        
        status_msg += f"📊 Maximum delay in frames: {max_delay_frames}\n"
        status_msg += f"🎬 Loading all frames into memory...\n"
        
        # Read all frames into memory
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None, "❌ Error: No frames could be read from video"
        
        status_msg += f"✅ Loaded {len(frames)} frames\n"
        status_msg += f"🔄 Processing delayed frames...\n"
        
        # Create output video
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Process each output frame
        num_output_frames = len(frames)
        
        for output_frame_idx in range(num_output_frames):
            # Create output frame
            output_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # For each pixel, determine which input frame to sample from
            for y in range(height):
                for x in range(width):
                    delay = delay_map[y, x]
                    source_frame_idx = output_frame_idx - delay
                    
                    # Clamp to valid frame range
                    if source_frame_idx < 0:
                        source_frame_idx = 0
                    elif source_frame_idx >= len(frames):
                        source_frame_idx = len(frames) - 1
                    
                    output_frame[y, x] = frames[source_frame_idx][y, x]
            
            out.write(output_frame)
            
            # Progress update every 10 frames
            if (output_frame_idx + 1) % 10 == 0 or output_frame_idx == num_output_frames - 1:
                progress = ((output_frame_idx + 1) / num_output_frames) * 100
                print(f"Processing: {progress:.1f}% ({output_frame_idx + 1}/{num_output_frames})")
        
        out.release()
        
        status_msg += f"✅ Generated {num_output_frames} output frames\n"
        status_msg += f"💾 Output saved at {output_fps} fps\n"
        status_msg += f"🎉 Processing complete!"
        
        return output_path, status_msg
        
    except Exception as e:
        return None, f"❌ Error during processing: {str(e)}"


def get_video_info(video_file):
    """
    Get video information without processing
    Returns: (info_html, fps, width, height)
    """
    if video_file is None:
        return "No video loaded", None, None, None
    
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return "Error: Could not open video file", None, None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        info_html = f"""
        <div style="padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
            <strong>Video Information:</strong><br>
            Dimensions: {width} x {height}<br>
            Frame Rate: {fps:.2f} fps<br>
            Total Frames: {total_frames}<br>
            Duration: {duration:.2f} seconds
        </div>
        """
        
        return info_html, fps, width, height
        
    except Exception as e:
        return f"Error reading video: {str(e)}", None, None, None


def process_video_with_delay_optimized(video_file, control_image, max_delay_ms, output_fps, output_scale, progress=gr.Progress()):
    """
    Optimized version using vectorized operations where possible
    """
    if video_file is None:
        return None, None, "Please upload a video file", ""
    
    if control_image is None:
        return None, None, "Please upload a control image", ""
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, None, "Error: Could not open video file", ""
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0:
            cap.release()
            return None, None, "Error: Video has no frames", ""
        
        # Calculate output dimensions based on scale
        if output_scale == "100%":
            output_width = input_width
            output_height = input_height
        elif output_scale == "50%":
            output_width = (input_width // 2) // 2 * 2  # Ensure divisible by 2
            output_height = (input_height // 2) // 2 * 2
        else:  # 25%
            output_width = (input_width // 4) // 2 * 2  # Ensure divisible by 2
            output_height = (input_height // 4) // 2 * 2
        
        status_msg = f"Input: {input_width}x{input_height} @ {input_fps:.2f} fps, {total_frames} frames\n"
        status_msg += f"Output: {output_width}x{output_height} @ {output_fps} fps\n"
        
        # Load and process control image
        if isinstance(control_image, str):
            ctrl_img = Image.open(control_image)
        else:
            ctrl_img = control_image
        
        # Convert to grayscale and resize to match INPUT video dimensions
        ctrl_img = ctrl_img.convert('L')
        ctrl_img = ctrl_img.resize((input_width, input_height), Image.Resampling.LANCZOS)
        ctrl_array = np.array(ctrl_img, dtype=np.float32) / 255.0
        
        status_msg += f"Control image scaled to {input_width}x{input_height}\n"
        status_msg += f"Maximum delay: {max_delay_ms}ms ({max_delay_ms / 1000.0} seconds)\n"
        
        # Calculate delay in frames for each pixel
        max_delay_frames = int((max_delay_ms / 1000.0) * input_fps)
        delay_map = (ctrl_array * max_delay_frames).astype(np.int32)
        
        status_msg += f"Maximum delay: {max_delay_frames} frames\n\n"
        
        # Read all frames into memory
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
            if frame_count % 50 == 0:
                percent = int((frame_count / total_frames) * 10)
                status_msg_loading = status_msg + f"Loading frames: {frame_count}/{total_frames} ({percent}%)"
                yield None, None, status_msg_loading, ""
        
        cap.release()
        
        if len(frames) == 0:
            return None, None, "Error: No frames could be read from video", ""
        
        status_msg += f"Loaded {len(frames)} frames\n"
        
        # Calculate total output frames including extra frames for delays
        num_input_frames = len(frames)
        num_output_frames = num_input_frames + max_delay_frames
        status_msg += f"Rendering {num_output_frames} frames (input: {num_input_frames} + delay buffer: {max_delay_frames})\n\n"
        
        # Create output video
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
        
        # Convert frames list to numpy array for faster indexing
        frames_array = np.array(frames)
        num_input_frames = len(frames)
        
        # Process each output frame (including extra frames for delayed pixels)
        preview_image = None
        for output_idx in range(num_output_frames):
            # Calculate source frame indices for all pixels at once (using INPUT dimensions)
            source_indices = output_idx - delay_map
            
            # Clamp to valid range
            source_indices = np.clip(source_indices, 0, num_input_frames - 1)
            
            # Create output frame by sampling from appropriate source frames
            output_frame_full = np.zeros((input_height, input_width, 3), dtype=np.uint8)
            
            for y in range(input_height):
                for x in range(input_width):
                    src_idx = source_indices[y, x]
                    output_frame_full[y, x] = frames_array[src_idx, y, x]
            
            # Scale down if needed
            if output_scale != "100%":
                output_frame = cv2.resize(output_frame_full, (output_width, output_height), interpolation=cv2.INTER_AREA)
            else:
                output_frame = output_frame_full
            
            out.write(output_frame)
            
            # Progress update with preview every 10 frames
            if (output_idx + 1) % 10 == 0 or output_idx == num_output_frames - 1:
                # Convert BGR to RGB for display
                preview_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                preview_image = Image.fromarray(preview_rgb)
                
                # Yield intermediate results with simple percentage
                percent = int((output_idx + 1) * 100 / num_output_frames)
                status_update = status_msg + f"Processing: {output_idx + 1}/{num_output_frames} frames ({percent}%)"
                yield None, preview_image, status_update, f"{percent}%"
        
        out.release()
        
        status_msg += f"Generated {num_output_frames} output frames\n"
        status_msg += f"Output saved at {output_fps} fps\n"
        status_msg += "Processing complete!"
        
        # Final yield to ensure video is returned
        yield output_path, preview_image, status_msg, "100%"
        
    except Exception as e:
        yield None, None, f"Error during processing: {str(e)}", "Error"


def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Time-Delay Video Processor"
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Time-Delay Video Processor</h1>
            <p>Apply temporal distortions to videos using luminance-based delays</p>
        </div>
        """)
        
        # Main content - 3 column layout for 2560x1440 screens
        with gr.Row():
            # LEFT COLUMN - Video Input
            with gr.Column(scale=1):
                gr.HTML("<h3>Input Video</h3>")
                
                video_input = gr.Video(
                    label="Upload Video (use trim controls to select segment)",
                    sources=["upload"],
                    height=500,
                    include_audio=True
                )
                
                # Video info display
                video_info_display = gr.HTML(
                    value="<div style='padding: 10px;'>No video loaded</div>"
                )
            
            # MIDDLE COLUMN - Control Image & Settings
            with gr.Column(scale=1):
                gr.HTML("<h3>Control Image</h3>")
                
                control_image_input = gr.Image(
                    label="Luminance Map",
                    type="pil",
                    height=400
                )
                
                gr.HTML("""
                <div style="padding: 10px; background-color: #e3f2fd; border-radius: 5px; margin: 10px 0;">
                    <strong>Control Image Guide:</strong><br>
                    White areas = Maximum delay<br>
                    Black areas = No delay<br>
                    Gray areas = Proportional delay<br>
                    Image will be scaled to match video dimensions
                </div>
                """)
                
                # Settings
                gr.HTML("<h3>Settings</h3>")
                
                output_scale_input = gr.Radio(
                    choices=["100%", "50%", "25%"],
                    value="100%",
                    label="Output Scale",
                    info="Scale output video dimensions (always divisible by 2)"
                )
                
                max_delay_input = gr.Slider(
                    minimum=10,
                    maximum=5000,
                    value=1000,
                    step=10,
                    label="Maximum Delay (milliseconds)",
                    info="Delay applied to white areas of control image"
                )
                
                output_fps_input = gr.Slider(
                    minimum=10,
                    maximum=60,
                    value=30,
                    step=1,
                    label="Output FPS",
                    info="Frame rate of output video"
                )
                
                # Process button
                process_btn = gr.Button(
                    "Process Video",
                    variant="primary",
                    size="lg"
                )
            
            # RIGHT COLUMN - Output & Status
            with gr.Column(scale=1):
                gr.HTML("<h3>Output</h3>")
                
                # Preview image
                preview_output = gr.Image(
                    label="Preview (updates during processing)",
                    type="pil",
                    height=300
                )
                
                video_output = gr.Video(
                    label="Processed Video (with download)",
                    height=400,
                    autoplay=False
                )
                
                # Progress percentage
                progress_text = gr.Textbox(
                    label="Progress",
                    value="0%",
                    interactive=False,
                    lines=1
                )
                
                # Status
                status_output = gr.Textbox(
                    label="Processing Status",
                    value="Ready to process video...",
                    interactive=False,
                    lines=6
                )
        
        # Connect video info display
        video_input.change(
            fn=lambda x: get_video_info(x)[0],
            inputs=[video_input],
            outputs=[video_info_display]
        )
        
        # Connect the processing function
        process_btn.click(
            fn=process_video_with_delay_optimized,
            inputs=[
                video_input,
                control_image_input,
                max_delay_input,
                output_fps_input,
                output_scale_input
            ],
            outputs=[video_output, preview_output, status_output, progress_text]
        )
        
    return interface


def main():
    """Main function to launch the application"""
    print("Starting Time-Delay Video Processor...")
    
    # Find an available port
    available_port = find_available_port()
    
    interface = create_interface()
    
    # Custom CSS for wide layout optimized for 2560x1440
    custom_css = """
    .gradio-container {
        max-width: 2400px !important;
        width: 95% !important;
        margin: auto;
    }
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    """
    
    # Launch the interface
    if available_port:
        print(f"Launching on port {available_port}")
        interface.launch(
            server_name="127.0.0.1",
            server_port=available_port,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,
            css=custom_css
        )
    else:
        print("Letting Gradio find an available port automatically")
        interface.launch(
            server_name="127.0.0.1",
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,
            css=custom_css
        )


if __name__ == "__main__":
    main()
