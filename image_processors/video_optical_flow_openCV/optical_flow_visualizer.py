import cv2
import numpy as np
import gradio as gr
import os
from pathlib import Path


def process_video(input_video, flow_density, line_thickness, magnitude_threshold, progress=gr.Progress()):
    """
    Process video with optical flow visualization
    
    Args:
        input_video: Path to input video file
        flow_density: Spacing between flow lines (5-30)
        line_thickness: Thickness of flow lines (1-5)
        magnitude_threshold: Minimum motion to visualize (0.1-5.0)
        progress: Gradio progress tracker
        
    Returns:
        Path to output video file
    """
    if input_video is None:
        return None
        
    try:
        input_path = input_video
        
        # Auto-generate output path
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_optical_flow{input_path_obj.suffix}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Read first frame
        ret, frame1 = cap.read()
        if not ret:
            raise Exception("Could not read first frame")
            
        # Convert to grayscale
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Create HSV image for flow visualization
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        
        frame_count = 0
        
        while True:
            ret, frame2 = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress_val = (frame_count / total_frames)
            progress(progress_val, desc=f"Processing frame {frame_count}/{total_frames}")
            
            # Convert to grayscale
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Create flow array
            flow = np.zeros((height, width, 2), dtype=np.float32)
            
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, flow, 
                                               0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Convert flow to magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Set HSV values based on flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            # Normalize magnitude to 0-255 range
            if mag.max() > 0:
                mag_normalized = (mag / mag.max()) * 255
            else:
                mag_normalized = np.zeros_like(mag)
            hsv[..., 2] = mag_normalized.astype(np.uint8)
            hsv[..., 2] = mag_normalized.astype(np.uint8)
            
            # Convert HSV to BGR for visualization
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Create flow visualization with lines
            flow_vis = draw_flow_lines(frame2.copy(), flow, flow_density, line_thickness, magnitude_threshold)
            
            # Combine original frame with flow visualization
            alpha = 0.7  # Transparency factor
            result = cv2.addWeighted(frame2, alpha, flow_vis, 1 - alpha, 0)
            
            # Write frame to output video
            out.write(result)
            
            # Update previous frame
            prvs = next_frame.copy()
                
        # Release everything
        cap.release()
        out.release()
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")


def draw_flow_lines(img, flow, flow_density, line_thickness, magnitude_threshold):
    """Draw optical flow lines on the image"""
    h, w = img.shape[:2]
    step = int(flow_density)
    thickness = int(line_thickness)
    threshold = magnitude_threshold
    
    # Create a copy for drawing
    flow_img = np.zeros_like(img)
    
    # Sample points at regular intervals
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    # Get flow vectors at sample points
    fx, fy = flow[y, x].T
    
    # Calculate magnitude
    magnitude = np.sqrt(fx*fx + fy*fy)
    
    # Filter by magnitude threshold
    valid = magnitude > threshold
    
    if np.any(valid):
        x_valid = x[valid]
        y_valid = y[valid]
        fx_valid = fx[valid]
        fy_valid = fy[valid]
        
        # Scale flow vectors for better visualization
        scale = 3
        fx_scaled = fx_valid * scale
        fy_scaled = fy_valid * scale
        
        # Calculate end points
        x_end = (x_valid + fx_scaled).astype(int)
        y_end = (y_valid + fy_scaled).astype(int)
        
        # Ensure end points are within image bounds
        x_end = np.clip(x_end, 0, w-1)
        y_end = np.clip(y_end, 0, h-1)
        
        # Draw flow lines
        for i in range(len(x_valid)):
            # Color based on flow direction
            color = get_flow_color(fx_valid[i], fy_valid[i])
            cv2.arrowedLine(flow_img, (x_valid[i], y_valid[i]), 
                           (x_end[i], y_end[i]), color, thickness, tipLength=0.3)
            
    return flow_img


def get_flow_color(fx, fy):
    """Get color based on flow direction"""
    # Convert flow to angle
    angle = np.arctan2(fy, fx) * 180 / np.pi
    angle = (angle + 360) % 360  # Ensure positive angle
    
    # Map angle to color
    if angle < 60:  # Right (red)
        return (0, 0, 255)
    elif angle < 120:  # Up-right (yellow)
        return (0, 255, 255)
    elif angle < 180:  # Up (green)
        return (0, 255, 0)
    elif angle < 240:  # Up-left (cyan)
        return (255, 255, 0)
    elif angle < 300:  # Left (blue)
        return (255, 0, 0)
    else:  # Down-left (magenta)
        return (255, 0, 255)


# Create Gradio interface
with gr.Blocks(title="Optical Flow Video Visualizer") as demo:
    gr.Markdown("# Video Optical Flow Visualizer")
    gr.Markdown("""
    Upload a video to visualize optical flow (motion patterns).
    The output shows directional arrows colored by movement direction:
    - **Red**: Right
    - **Yellow**: Up-right
    - **Green**: Up
    - **Cyan**: Up-left
    - **Blue**: Left
    - **Magenta**: Down-left
    """)
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            
            gr.Markdown("### Optical Flow Parameters")
            flow_density = gr.Slider(
                minimum=5, maximum=30, step=1, value=15,
                label="Flow Line Density",
                info="Spacing between flow vectors (lower = more dense)"
            )
            line_thickness = gr.Slider(
                minimum=1, maximum=5, step=1, value=2,
                label="Flow Line Thickness",
                info="Thickness of flow arrows"
            )
            magnitude_threshold = gr.Slider(
                minimum=0.1, maximum=5.0, step=0.1, value=1.0,
                label="Motion Threshold",
                info="Minimum motion magnitude to display"
            )
            
            process_btn = gr.Button("Process Video", variant="primary", size="lg")
        
        with gr.Column():
            output_video = gr.Video(label="Output Video with Optical Flow")
            status_text = gr.Textbox(label="Status", interactive=False)
    
    # Connect the button
    process_btn.click(
        fn=process_video,
        inputs=[input_video, flow_density, line_thickness, magnitude_threshold],
        outputs=output_video
    )


if __name__ == "__main__":
    demo.launch()
