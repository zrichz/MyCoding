"""
Slideshow Video Creator

Creates a video slideshow from images with:
- Each image displays for 5 seconds
- 2-second fade transitions between images
- Gentle zoom effect (100% to 105% or 105% to 100%) while each image is visible
- Exports as 24fps MP4 video

Usage:
    python slideshow_video_creator.py

The script will:
1. Launch Gradio interface
2. Select image directory and options
3. Create the slideshow with fade transitions and zoom effects
4. Save as output.mp4
"""

import os
from PIL import Image
import random
import numpy as np
import gradio as gr
from pathlib import Path
import cv2
from tqdm import tqdm
import tempfile
import shutil

class SlideshowVideoCreator:
    def __init__(self, half_resolution=False):
        # Resolution option
        self.half_resolution = half_resolution
        
        # Output dimensions (will be set based on first image)
        self.output_width = None
        self.output_height = None
        
        # Timing settings
        self.fps = 24
        self.display_duration = 4.0  # seconds per image
        self.fade_duration = 2.0  # seconds for fade transition
        
        # Image management
        self.image_directory = None
        self.image_files = []
        
        # Video frames
        self.frames = []
        
    def load_image_list(self, directory):
        """Load list of supported image files from directory"""
        self.image_directory = directory
        if not self.image_directory:
            return
            
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = []
        
        for filename in os.listdir(self.image_directory):
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                self.image_files.append(filename)
        
        self.image_files.sort()
        
        return len(self.image_files)
    
    def load_and_prepare_image(self, filename):
        """Load and prepare an image at output resolution"""
        filepath = os.path.join(self.image_directory, filename)
        
        try:
            # Load image with PIL
            pil_image = Image.open(filepath)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Set output dimensions from first image
            if self.output_width is None:
                img_width, img_height = pil_image.size
                if self.half_resolution:
                    self.output_width = img_width // 2
                    self.output_height = img_height // 2
                else:
                    self.output_width = img_width
                    self.output_height = img_height
            
            # Resize to output dimensions with high-quality LANCZOS
            pil_image = pil_image.resize((self.output_width, self.output_height), Image.Resampling.LANCZOS)
            
            # Convert to numpy array (RGB format)
            image_array = np.array(pil_image, dtype=np.uint8)
            
            return image_array
            
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None
    
    def render_frame(self, image_array, alpha=1.0):
        """Render a frame with the given image, zoom, and opacity"""
        # Create a black background at output resolution
        frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.float32)
        
        # Scale the image with high-quality interpolation
        scaled_image = self.scale_surface(image_array, zoom_scale).astype(np.float32)
        scaled_height, scaled_width = scaled_image.shape[:2]
        
        # Center the scaled image
        y = (self.output_height - scaled_height) // 2
        x = (self.output_width - scaled_width) // 2
        
        # Calculate the region to place the image
        y1 = max(0, y)
        y2 = min(self.output_height, y + scaled_height)
        x1 = max(0, x)
        x2 = min(self.output_width, x + scaled_width)
        
        # Calculate source region
        sy1 = max(0, -y)
        sy2 = sy1 + (y2 - y1)
        sx1 = max(0, -x)
        sx2 = sx1 + (x2 - x1)
        
        # Apply alpha blending with proper float precision
        if alpha < 1.0:
            frame[y1:y2, x1:x2] = scaled_image[sy1:sy2, sx1:sx2] * alpha
        else:
            frame[y1:y2, x1:x2] = scaled_image[sy1:sy2, sx1:sx2]
        
        # Convert back to uint8
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    def create_slideshow_frames(self, progress=gr.Progress()):
        """Create all frames for the slideshow"""
        if not self.image_files:
            return "No images found!"
        
        progress(0, desc="Starting...")
        
        display_frames = int(self.display_duration * self.fps)
        fade_frames = int(self.fade_duration * self.fps)
        total_visible_frames = display_frames + fade_frames
        
        total_frames_needed = len(self.image_files) * display_frames + (len(self.image_files) - 1) * fade_frames
        current_frame = 0
        
        for i in range(len(self.image_files)):
            filename = self.image_files[i]
            progress(current_frame / total_frames_needed, 
                    desc=f"Processing {i+1}/{len(self.image_files)}: {filename}")
            
            # Load current image as numpy array
            current_image = self.load_and_prepare_image(filename)
            if current_image is None:
                continue
            
            # Prepare next image if available (for fade transition)
            next_image = None
            if i < len(self.image_files) - 1:
                next_filename = self.image_files[i + 1]
                next_image = self.load_and_prepare_image(next_filename)
            
            # Generate frames for the display duration
            for frame_num in range(display_frames):
                # Simply show the image without zoom
                self.frames.append(current_image.copy())
                current_frame += 1
            
            # Generate frames for fade transition
            if next_image is not None:
                for frame_num in range(fade_frames):
                    fade_progress = float(frame_num) / float(fade_frames - 1) if fade_frames > 1 else 1.0
                    
                    # Simple fade between images
                    current_alpha = 1.0 - fade_progress
                    next_alpha = fade_progress
                    
                    # Blend frames in float space for smooth fading
                    frame1 = current_image.astype(np.float32) * current_alpha
                    frame2 = next_image.astype(np.float32) * next_alpha
                    
                    composite_frame = (frame1 + frame2)
                    composite_frame = np.clip(composite_frame, 0, 255).astype(np.uint8)
                    self.frames.append(composite_frame)
                    current_frame += 1
        
        progress(1.0, desc="Frames complete!")
        return f"Created {len(self.frames)} frames"
    
    def save_video(self, output_path="output.mp4", progress=gr.Progress()):
        """Save frames as MP4 video using opencv with optimized encoding"""
        if not self.frames:
            return "No frames to save!"
        
        try:
            progress(0, desc="Initializing video writer...")
            
            # Get frame dimensions
            height, width = self.frames[0].shape[:2]
            
            # Use H.264 codec for better compression and quality
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            if not out.isOpened():
                # Fallback to mp4v if avc1 fails
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            # Write frames with progress
            total_frames = len(self.frames)
            for i, frame in enumerate(self.frames):
                if i % 10 == 0:
                    progress(i / total_frames, desc=f"Writing frame {i+1}/{total_frames}")
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            progress(1.0, desc="Video saved!")
            
            return f"Video saved successfully: {output_path}"
            
        except Exception as e:
            return f"Error saving video: {e}"

def create_slideshow(image_folder, half_resolution, progress=gr.Progress()):
    """Main function to create slideshow video"""
    try:
        if not image_folder or not os.path.isdir(image_folder):
            return "Please select a valid image folder", None
        
        # Create slideshow creator
        creator = SlideshowVideoCreator(half_resolution=half_resolution)
        
        # Load images
        num_images = creator.load_image_list(image_folder)
        if num_images == 0:
            return "No images found in the selected folder", None
        
        yield f"Found {num_images} images. Starting frame generation...", None
        
        # Create frames
        result = creator.create_slideshow_frames(progress=progress)
        yield f"{result}. Now encoding video...", None
        
        # Save video to temp directory for Gradio 6.0 compatibility
        temp_output = os.path.join(tempfile.gettempdir(), "slideshow_output.mp4")
        result = creator.save_video(temp_output, progress=progress)
        
        # Also save a copy to the image folder for user convenience
        final_output = os.path.join(image_folder, "slideshow_output.mp4")
        if os.path.exists(temp_output):
            shutil.copy2(temp_output, final_output)
            yield f"{result}\n✓ Video saved to: {final_output}", temp_output
        else:
            yield result, None
            
    except Exception as e:
        yield f"Error: {str(e)}", None


# Gradio Interface
def launch_gradio():
    with gr.Blocks(title="Slideshow Video Creator") as app:
        gr.Markdown("# Slideshow Video Creator")
        
        with gr.Row():
            with gr.Column():
                # Use file explorer for folder selection
                image_folder = gr.Textbox(
                    label="Image Folder Path",
                    placeholder="Click below to browse for a folder",
                    info="All images should be the same dimensions"
                )
                browse_btn = gr.Button("Browse for Folder", size="sm")
                
                half_res = gr.Checkbox(
                    label="Half Resolution Output",
                    value=False,
                    info="Enable to create output at 50% of original image size"
                )
                
                create_btn = gr.Button("Create Slideshow", variant="primary", size="lg")
            
            with gr.Column():
                status_text = gr.Textbox(
                    label="Status",
                    lines=3,
                    interactive=False
                )
                
                video_output = gr.Video(
                    label="Output Video",
                    interactive=False
                )
        
        gr.Markdown("### Instructions")
        gr.Markdown("""
        1. Click 'Browse for Folder' to select a folder containing images (all should be same dimensions)
        2. Optionally enable half-resolution for faster processing
        3. Click 'Create Slideshow' and wait for processing
        4. Video will be saved as `slideshow_output.mp4` in the image folder
        """)
        
        def browse_folder():
            """Open folder selection dialog"""
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            folder = filedialog.askdirectory(title="Select Image Folder")
            root.destroy()
            return folder if folder else ""
        
        browse_btn.click(
            fn=browse_folder,
            inputs=[],
            outputs=[image_folder]
        )
        
        create_btn.click(
            fn=create_slideshow,
            inputs=[image_folder, half_res],
            outputs=[status_text, video_output]
        )
    
    return app


if __name__ == "__main__":
    app = launch_gradio()
    # Increase timeouts significantly for longer processing
    app.queue(max_size=20)
    app.launch(max_threads=10, show_error=True, inbrowser=True)
