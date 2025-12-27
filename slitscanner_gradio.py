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
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:  # Port is available
                return port
        except Exception:
            continue
    
    # If no port found in range, let Gradio find one automatically
    return None


def process_video_to_voxel_cube(video_file, frame_step, output_frames, output_fps, downsample_factor, progress=gr.Progress()):
    """
    Process video into a voxel cube and create rotating slice video output
    
    Args:
        video_file: Uploaded video file
        frame_step: Take every nth frame (e.g., 3 means frames 0, 3, 6, 9...)
        output_frames: Number of output video frames (rotation steps)
        output_fps: FPS for output video
        downsample_factor: Resolution reduction (1=full, 2=half, 4=quarter)
        progress: Gradio progress tracker
    
    Returns:
        tuple: (output_video_path, status_message, preview_image)
    """
    if video_file is None:
        return None, "❌ Please upload a video file", None
    
    try:
        progress(0, desc="Opening video...")
        # Open video file
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            return None, "❌ Could not open video file. Please check the format.", None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Apply downsampling to dimensions (ensure divisible by 2)
        frame_width = (orig_frame_width // downsample_factor) // 2 * 2
        frame_height = (orig_frame_height // downsample_factor) // 2 * 2
        
        # Calculate how many frames will be in the voxel cube
        n_frames_to_use = total_frames // frame_step
        
        if n_frames_to_use < 3:
            return None, f"Not enough frames. With every {frame_step} frames, only {n_frames_to_use} frames available. Need at least 3.", None
        
        # Read frames into voxel cube
        progress(0.1, desc=f"Building voxel cube ({n_frames_to_use} frames)...")
        voxel_cube = np.zeros((frame_height, frame_width, n_frames_to_use, 3), dtype=np.uint8)
        
        frame_count = 0
        cube_idx = 0
        
        while cap.isOpened() and cube_idx < n_frames_to_use:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_step == 0:
                # Resize frame if downsampling
                if downsample_factor > 1:
                    frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                voxel_cube[:, :, cube_idx, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cube_idx += 1
                # Update progress during cube building
                progress(0.1 + 0.3 * (cube_idx / n_frames_to_use), desc=f"Loading frames... {cube_idx}/{n_frames_to_use}")
            
            frame_count += 1
        
        cap.release()
        
        if cube_idx == 0:
            return None, "No frames were loaded into voxel cube.", None
        
        # Update actual cube size if fewer frames loaded
        if cube_idx < n_frames_to_use:
            voxel_cube = voxel_cube[:, :, :cube_idx, :]
            n_frames_to_use = cube_idx
        
        status_msg = f"Voxel cube built: {frame_height}x{frame_width}x{n_frames_to_use}\n"
        status_msg += f"Creating {output_frames} output frames with rotation...\n"
        
        progress(0.4, desc="Initializing video writer...")
        # Create output video
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_output.name
        temp_output.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))
        
        # Calculate center of cube
        center_y = frame_height / 2
        center_x = frame_width / 2
        center_z = n_frames_to_use / 2
        
        # Store preview frame
        preview_frame = None
        
        # Generate output frames by rotating the slice plane
        for frame_idx in range(output_frames):
            # Update progress
            progress(0.4 + 0.6 * (frame_idx / output_frames), desc=f"Generating frame {frame_idx+1}/{output_frames}...")
            
            # Calculate rotation angle (0 to 360 degrees)
            angle = (frame_idx / output_frames) * 2 * np.pi
            
            # VECTORIZED APPROACH - process all pixels at once
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:frame_height, 0:frame_width]
            
            # Translate to center-origin coordinates
            x_rel = x_coords - center_x
            y_rel = y_coords - center_y
            
            # Rotate around Z-axis (vectorized)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            x_rot = x_rel * cos_angle
            z_rot = x_rel * sin_angle
            
            # Translate back
            x_voxel = x_rot + center_x
            z_voxel = z_rot + center_z
            
            # Use modulo wrapping (tiling) for coordinates
            x_wrapped = x_voxel % frame_width
            z_wrapped = z_voxel % n_frames_to_use
            
            # Bilinear interpolation indices (vectorized)
            x0 = np.floor(x_wrapped).astype(int) % frame_width
            x1 = np.ceil(x_wrapped).astype(int) % frame_width
            z0 = np.floor(z_wrapped).astype(int) % n_frames_to_use
            z1 = np.ceil(z_wrapped).astype(int) % n_frames_to_use
            y_idx = y_coords % frame_height
            
            # Interpolation weights
            wx = x_wrapped - np.floor(x_wrapped)
            wz = z_wrapped - np.floor(z_wrapped)
            
            # Expand dimensions for broadcasting
            wx = wx[:, :, np.newaxis]
            wz = wz[:, :, np.newaxis]
            
            # Sample all 4 corners (vectorized)
            c00 = voxel_cube[y_idx, x0, z0, :]
            c01 = voxel_cube[y_idx, x0, z1, :]
            c10 = voxel_cube[y_idx, x1, z0, :]
            c11 = voxel_cube[y_idx, x1, z1, :]
            
            # Bilinear interpolation (vectorized)
            c0 = c00 * (1 - wx) + c10 * wx
            c1 = c01 * (1 - wx) + c11 * wx
            output_frame = (c0 * (1 - wz) + c1 * wz).astype(np.uint8)
            
            # Write frame to output video
            out.write(cv2.cvtColor(output_frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # Capture preview frame every 10 frames
            if frame_idx % 10 == 0 or frame_idx == output_frames - 1:
                preview_frame = Image.fromarray(output_frame.astype(np.uint8))
        
        out.release()
        
        progress(1.0, desc="Complete!")
        status_msg += f"Video created: {output_frames} frames at {output_fps} FPS\n"
        status_msg += f"Output: {frame_width}x{frame_height}px video\n"
        status_msg += f"Rotation: 360° around Z-axis"
        
        return output_path, status_msg, preview_frame
        
    except Exception as e:
        return None, f"Processing failed: {str(e)}", None


def process_video(video_file, sampling_method, n_frames, progress=gr.Progress()):
    """
    Process the uploaded video and create slitscanned image
    
    Args:
        video_file: Uploaded video file
        sampling_method: Frame sampling method (every_frame, every_2, every_n)
        n_frames: Number for every_n sampling (3-1000)
        progress: Gradio progress tracker
    
    Returns:
        tuple: (processed_image, status_message, None)
    """
    if video_file is None:
        return None, "❌ Please upload a video file", None
    
    try:
        progress(0, desc="Opening video...")
        # Open video file
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            return None, "❌ Could not open video file. Please check the format.", None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate center column
        center_x = frame_width // 2
        
        progress(0.1, desc="Calculating frame sampling...")
        # Determine frame step based on sampling method
        if sampling_method == "Every Frame":
            frame_step = 1
        elif sampling_method == "Every 2 Frames":
            frame_step = 2
        elif sampling_method == "Every n Frames":
            try:
                frame_step = int(n_frames)
                if not (3 <= frame_step <= 1000):
                    return None, "n frames value must be between 3 and 1000", None
            except (ValueError, TypeError):
                return None, "Please enter a valid number between 3 and 1000", None
        else:
            frame_step = 1
        
        # Calculate how many frames we'll actually process
        frames_to_process = total_frames // frame_step
        max_width = 3000  # Limit output width to prevent memory issues
        
        if frames_to_process > max_width:
            frames_to_process = max_width
        
        if frames_to_process == 0:
            return None, "No frames to process. Video might be too short or frame step too large.", None
        
        progress(0.2, desc=f"Processing {frames_to_process} frames...")
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
                
                # Update progress
                if processed_count % 10 == 0:
                    progress(0.2 + 0.7 * (processed_count / frames_to_process), desc=f"Processing frame {processed_count}/{frames_to_process}")
            
            frame_count += 1
        
        cap.release()
        
        if processed_count == 0:
            return None, "No frames were processed. Please check your video file.", None
        
        progress(0.95, desc="Converting to image...")
        # Convert BGR to RGB
        slitscanned_image = cv2.cvtColor(slitscanned_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        output_image = Image.fromarray(slitscanned_image)
        
        progress(1.0, desc="Complete!")
        # Create status message
        duration = total_frames / fps if fps > 0 else 0
        status_msg = f"Success. Processed {processed_count} frames from {total_frames} total frames\n"
        status_msg += f"Video: {frame_width}x{frame_height}px, {duration:.1f}s, {fps:.1f} FPS\n"
        status_msg += f"Output: {processed_count}x{frame_height}px image\n"
        status_msg += f"Sampling: Every {frame_step} frame(s)"
        
        return output_image, status_msg, None
        
    except Exception as e:
        return None, f"Processing failed: {str(e)}", None


def process_dispatcher(video_file, output_mode, sampling_method, n_frames, frame_step_cube, output_frames, output_fps, downsample_factor, progress=gr.Progress()):
    """Dispatch to appropriate processing function based on output mode"""
    if output_mode == "Still Image":
        return process_video(video_file, sampling_method, n_frames, progress)
    else:  # Video output
        return process_video_to_voxel_cube(video_file, frame_step_cube, output_frames, output_fps, downsample_factor, progress)


def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="Slitscanner Video Processor",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1800px;
            margin: auto;
            padding: 20px;
        }
        .main-header {
            text-align: center;
            margin-bottom: 40px;
        }
        .status-box {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        #status_output {
            min-height: 120px;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Slitscanner Video Processor</h1>
            <p>Transform videos into images OR create voxel cube rotating slice videos</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                # Input section
                gr.HTML("<h3>Input & Settings</h3>")
                
                video_input = gr.File(
                    label="Upload Video File",
                    file_types=["video"],
                    type="filepath"
                )
                
                # Input video preview
                input_preview = gr.Image(
                    label="Input Video Preview",
                    type="pil",
                    height=150,
                    show_label=True,
                    interactive=False
                )
                
                # Function to show first frame of video
                def show_video_preview(video_file):
                    if video_file is None:
                        return None
                    try:
                        cap = cv2.VideoCapture(video_file)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            return Image.fromarray(frame_rgb)
                    except:
                        pass
                    return None
                
                video_input.change(
                    fn=show_video_preview,
                    inputs=[video_input],
                    outputs=[input_preview]
                )
                
                gr.HTML("<br><h4>Output Mode</h4>")
                
                output_mode = gr.Radio(
                    choices=["Still Image", "Video (Voxel Cube Rotation)"],
                    value="Still Image",
                    label="Output Type"
                )
                
                gr.HTML("<br><h4>Processing Options</h4>")
                
                # Still image options
                sampling_method = gr.Radio(
                    choices=["Every Frame", "Every 2 Frames", "Every n Frames"],
                    value="Every Frame",
                    label="Frame Sampling",
                    info=""
                )
                
                n_frames_input = gr.Number(
                    value=3,
                    minimum=3,
                    maximum=1000,
                    step=1,
                    label="N Frames",
                    info="Extract every nth frame (3-1000)",
                    visible=False
                )
                
                # Video output options
                frame_step_cube = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=3,
                    step=1,
                    label="Frame Step",
                    info="Use every nth frame",
                    visible=False
                )
                
                output_frames = gr.Slider(
                    minimum=30,
                    maximum=1200,
                    value=120,
                    step=10,
                    label="Output Video Frames",
                    info="",
                    visible=False
                )
                
                output_fps = gr.Slider(
                    minimum=10,
                    maximum=60,
                    value=30,
                    step=5,
                    label="Output Video FPS",
                    info="",
                    visible=False
                )
                
                downsample_factor = gr.Radio(
                    choices=["1 (Full Resolution)", "2 (Half)", "4 (Quarter)"],
                    value="2 (Half)",
                    label="Resolution",
                    info="Lower resolution = faster",
                    visible=False
                )
                
                # Show/hide options based on selections
                def toggle_n_frames(method):
                    return gr.update(visible=(method == "Every n Frames"))
                
                def toggle_options(mode):
                    is_image = mode == "Still Image"
                    is_video = mode == "Video (Voxel Cube Rotation)"
                    return (
                        gr.update(visible=is_image),  # sampling_method
                        gr.update(visible=False),  # n_frames_input (hide by default)
                        gr.update(visible=is_video),  # frame_step_cube
                        gr.update(visible=is_video),  # output_frames
                        gr.update(visible=is_video),  # output_fps
                        gr.update(visible=is_video)   # downsample_factor
                    )
                
                output_mode.change(
                    fn=toggle_options,
                    inputs=[output_mode],
                    outputs=[sampling_method, n_frames_input, frame_step_cube, output_frames, output_fps, downsample_factor]
                )
                
                sampling_method.change(
                    fn=toggle_n_frames,
                    inputs=[sampling_method],
                    outputs=[n_frames_input]
                )
                
                # Process button
                process_btn = gr.Button(
                    "Process Video",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=3, min_width=800):
                # Output section
                gr.HTML("<h3>Output</h3>")
                
                # Status output at top of output column
                status_output = gr.Textbox(
                    label="Processing Status",
                    value="Ready to process video...",
                    interactive=False,
                    lines=5,
                    elem_id="status_output"
                )
                
                # Frame preview during generation
                frame_preview = gr.Image(
                    label="Generation Preview",
                    type="pil",
                    height=300,
                    show_label=True,
                    visible=True
                )
                
                image_output = gr.Image(
                    label="Result",
                    type="pil",
                    height=600,
                    show_download_button=True,
                    visible=True
                )
                
                video_output = gr.Video(
                    label="Video",
                    height=600,
                    visible=False
                )
        
        # Helper function to manage outputs
        def process_and_route(video_file, output_mode, sampling_method, n_frames, frame_step_cube, output_frames, output_fps, downsample_str, progress=gr.Progress()):
            # Parse downsample factor from string
            downsample_factor = int(downsample_str.split()[0])
            
            result, status, preview = process_dispatcher(video_file, output_mode, sampling_method, n_frames, frame_step_cube, output_frames, output_fps, downsample_factor, progress)
            
            if output_mode == "Still Image":
                return result, None, status, preview, gr.update(visible=True), gr.update(visible=False)
            else:
                return None, result, status, preview, gr.update(visible=False), gr.update(visible=True)
        
        # Process button click handler
        process_btn.click(
            fn=process_and_route,
            inputs=[video_input, output_mode, sampling_method, n_frames_input, frame_step_cube, output_frames, output_fps, downsample_factor],
            outputs=[image_output, video_output, status_output, frame_preview, image_output, video_output]
        )
    
    return interface


def main():
    """Main function to launch the application"""
    print("Starting Slitscanner Gradio Application...")
    
    # Find an available port
    available_port = find_available_port()
    
    interface = create_interface()
    
    # Launch the interface with improved port handling
    if available_port:
        print(f"Launching on port {available_port}")
        interface.launch(
            server_name="127.0.0.1",
            server_port=available_port,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,  # Automatically open browser
            max_threads=10,  # Increase concurrent thread limit
            max_file_size="500mb"  # Increase max file upload size
        )
    else:
        print("Letting Gradio find an available port automatically")
        interface.launch(
            server_name="127.0.0.1",
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True,  # Automatically open browser
            max_threads=10,  # Increase concurrent thread limit
            max_file_size="800mb"  # Increase max file upload size
        )


if __name__ == "__main__":
    main()
