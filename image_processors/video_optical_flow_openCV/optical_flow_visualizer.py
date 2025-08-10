import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
from pathlib import Path


class OpticalFlowVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Optical Flow Visualizer")
        self.root.geometry("500x400")
        
        # Variables
        self.input_video_path = tk.StringVar()
        self.output_video_path = tk.StringVar()
        self.is_processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="Optical Flow Video Processor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input video selection
        ttk.Label(main_frame, text="Select Input Video:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(main_frame, textvariable=self.input_video_path, width=50, state="readonly").grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(main_frame, text="Browse", command=self.select_input_video).grid(row=2, column=2, padx=(5, 0), pady=5)
        
        # Output video path (auto-generated)
        ttk.Label(main_frame, text="Output Video Path:").grid(row=3, column=0, sticky="w", pady=(20, 5))
        ttk.Entry(main_frame, textvariable=self.output_video_path, width=50, state="readonly").grid(row=4, column=0, columnspan=3, pady=5, sticky="ew")
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Optical Flow Parameters", padding="10")
        params_frame.grid(row=5, column=0, columnspan=3, pady=(20, 10), sticky="ew")
        
        # Flow visualization density
        ttk.Label(params_frame, text="Flow Line Density:").grid(row=0, column=0, sticky="w")
        self.flow_density = tk.IntVar(value=15)
        density_scale = ttk.Scale(params_frame, from_=5, to=30, variable=self.flow_density, orient=tk.HORIZONTAL)
        density_scale.grid(row=0, column=1, sticky="ew", padx=(10, 5))
        ttk.Label(params_frame, textvariable=self.flow_density).grid(row=0, column=2)
        
        # Flow line thickness
        ttk.Label(params_frame, text="Flow Line Thickness:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.line_thickness = tk.IntVar(value=2)
        thickness_scale = ttk.Scale(params_frame, from_=1, to=5, variable=self.line_thickness, orient=tk.HORIZONTAL)
        thickness_scale.grid(row=1, column=1, sticky="ew", padx=(10, 5), pady=(10, 0))
        ttk.Label(params_frame, textvariable=self.line_thickness).grid(row=1, column=2, pady=(10, 0))
        
        # Flow magnitude threshold
        ttk.Label(params_frame, text="Motion Threshold:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.magnitude_threshold = tk.DoubleVar(value=1.0)
        threshold_scale = ttk.Scale(params_frame, from_=0.1, to=5.0, variable=self.magnitude_threshold, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=2, column=1, sticky="ew", padx=(10, 5), pady=(10, 0))
        threshold_label = ttk.Label(params_frame, text="")
        threshold_label.grid(row=2, column=2, pady=(10, 0))
        
        # Update threshold label
        def update_threshold_label(*args):
            threshold_label.config(text=f"{self.magnitude_threshold.get():.1f}")
        self.magnitude_threshold.trace('w', update_threshold_label)
        update_threshold_label()
        
        # Configure column weights for proper resizing
        params_frame.columnconfigure(1, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, pady=(10, 5), sticky="ew")
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to process video")
        self.status_label.grid(row=7, column=0, columnspan=3, pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Video", command=self.start_processing)
        self.process_button.grid(row=8, column=0, columnspan=3, pady=(20, 0))
        
        # Configure main frame column weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def select_input_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.input_video_path.set(file_path)
            # Auto-generate output path
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}_optical_flow{input_path.suffix}"
            self.output_video_path.set(str(output_path))
            
    def start_processing(self):
        if not self.input_video_path.get():
            messagebox.showerror("Error", "Please select an input video file.")
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress.")
            return
            
        # Start processing in a separate thread
        self.is_processing = True
        self.process_button.config(state="disabled", text="Processing...")
        self.progress_var.set(0)
        
        processing_thread = threading.Thread(target=self.process_video)
        processing_thread.daemon = True
        processing_thread.start()
        
    def process_video(self):
        try:
            input_path = self.input_video_path.get()
            output_path = self.output_video_path.get()
            
            self.update_status("Opening video file...")
            
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
            
            self.update_status("Processing frames...")
            
            while True:
                ret, frame2 = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                
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
                
                # Convert HSV to BGR for visualization
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Create flow visualization with lines
                flow_vis = self.draw_flow_lines(frame2.copy(), flow)
                
                # Combine original frame with flow visualization
                alpha = 0.7  # Transparency factor
                result = cv2.addWeighted(frame2, alpha, flow_vis, 1 - alpha, 0)
                
                # Write frame to output video
                out.write(result)
                
                # Update previous frame
                prvs = next_frame.copy()
                
                if frame_count % 30 == 0:  # Update status every 30 frames
                    self.update_status(f"Processing frame {frame_count}/{total_frames}")
                    
            # Release everything
            cap.release()
            out.release()
            
            self.update_status("Processing complete!")
            self.progress_var.set(100)
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", 
                f"Optical flow video saved to:\n{output_path}"
            ))
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            self.update_status(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_button.config(state="normal", text="Process Video"))
            
    def draw_flow_lines(self, img, flow):
        """Draw optical flow lines on the image"""
        h, w = img.shape[:2]
        step = self.flow_density.get()
        thickness = self.line_thickness.get()
        threshold = self.magnitude_threshold.get()
        
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
                color = self.get_flow_color(fx_valid[i], fy_valid[i])
                cv2.arrowedLine(flow_img, (x_valid[i], y_valid[i]), 
                               (x_end[i], y_end[i]), color, thickness, tipLength=0.3)
                
        return flow_img
        
    def get_flow_color(self, fx, fy):
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
            
    def update_status(self, message):
        """Thread-safe status update"""
        self.root.after(0, lambda: self.status_label.config(text=message))


def main():
    root = tk.Tk()
    app = OpticalFlowVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
