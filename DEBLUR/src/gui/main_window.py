"""
Simple GUI interface for the DEBLUR application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import threading
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.image_utils import load_image, save_image, show_comparison
from src.deblur.gaussian_deblur import GaussianDeblur
from src.deblur.motion_deblur import MotionDeblur


class DeblurApp:
    """
    Main GUI application for image deblurring.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DEBLUR - Image Deblurring Tool")
        self.root.geometry("800x600")
        
        self.current_image = None
        self.processed_image = None
        self.image_path = None
        
        self.gaussian_deblur = GaussianDeblur()
        self.motion_deblur = MotionDeblur()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # File operations
        file_frame = ttk.LabelFrame(main_frame, text="File Operations", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Result", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Method selection
        method_frame = ttk.LabelFrame(main_frame, text="Deblurring Method", padding="5")
        method_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.method_var = tk.StringVar(value="gaussian")
        ttk.Radiobutton(method_frame, text="Gaussian Deblur", variable=self.method_var, 
                       value="gaussian", command=self.update_parameters).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(method_frame, text="Motion Deblur", variable=self.method_var, 
                       value="motion", command=self.update_parameters).pack(side=tk.LEFT, padx=10)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        self.params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Gaussian parameters
        self.gaussian_frame = ttk.Frame(self.params_frame)
        
        ttk.Label(self.gaussian_frame, text="Kernel Size:").grid(row=0, column=0, padx=5)
        self.kernel_size_var = tk.IntVar(value=15)
        ttk.Scale(self.gaussian_frame, from_=5, to=31, variable=self.kernel_size_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, padx=5)
        ttk.Label(self.gaussian_frame, textvariable=self.kernel_size_var).grid(row=0, column=2, padx=5)
        
        ttk.Label(self.gaussian_frame, text="Iterations:").grid(row=1, column=0, padx=5)
        self.iterations_var = tk.IntVar(value=30)
        ttk.Scale(self.gaussian_frame, from_=10, to=100, variable=self.iterations_var, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, padx=5)
        ttk.Label(self.gaussian_frame, textvariable=self.iterations_var).grid(row=1, column=2, padx=5)
        
        ttk.Label(self.gaussian_frame, text="Downsample:").grid(row=2, column=0, padx=5)
        self.downsample_var = tk.StringVar(value="auto")
        downsample_combo = ttk.Combobox(self.gaussian_frame, textvariable=self.downsample_var, 
                                      values=["auto", "1x (none)", "2x", "4x"], state="readonly")
        downsample_combo.grid(row=2, column=1, padx=5, sticky="ew")
        
        ttk.Label(self.gaussian_frame, text="Show Progress:").grid(row=3, column=0, padx=5)
        self.show_progress_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.gaussian_frame, variable=self.show_progress_var).grid(row=3, column=1, padx=5, sticky="w")
        
        # Motion parameters
        self.motion_frame = ttk.Frame(self.params_frame)
        
        ttk.Label(self.motion_frame, text="Angle (degrees):").grid(row=0, column=0, padx=5)
        self.angle_var = tk.DoubleVar(value=0)
        ttk.Scale(self.motion_frame, from_=-90, to=90, variable=self.angle_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, padx=5)
        ttk.Label(self.motion_frame, textvariable=self.angle_var).grid(row=0, column=2, padx=5)
        
        ttk.Label(self.motion_frame, text="Length:").grid(row=1, column=0, padx=5)
        self.length_var = tk.IntVar(value=20)
        ttk.Scale(self.motion_frame, from_=5, to=50, variable=self.length_var, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, padx=5)
        ttk.Label(self.motion_frame, textvariable=self.length_var).grid(row=1, column=2, padx=5)
        
        self.update_parameters()
        
        # Process button
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(process_frame, text="Process Image", command=self.process_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="Compare Images", command=self.compare_images).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, pady=5)
        
        # Image display (placeholder)
        self.image_label = ttk.Label(main_frame, text="Load an image to get started")
        self.image_label.grid(row=5, column=0, columnspan=2, pady=10)
    
    def update_parameters(self):
        """Update parameter display based on selected method."""
        # Hide all parameter frames
        self.gaussian_frame.pack_forget()
        self.motion_frame.pack_forget()
        
        # Show relevant parameters
        if self.method_var.get() == "gaussian":
            self.gaussian_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.motion_frame.pack(fill=tk.X, padx=5, pady=5)
    
    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image = load_image(file_path)
                self.image_path = file_path
                self.display_image(self.current_image)
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display an image in the GUI."""
        if image is None:
            return
        
        # Convert to PIL Image and resize for display
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L')
        else:
            pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Resize to fit display
        display_size = (400, 300)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def process_image(self):
        """Process the current image with selected method."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        # Start processing in a separate thread
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        """Process image in a separate thread."""
        try:
            # Update UI
            self.progress.start()
            self.status_var.set("Processing...")
            
            if self.method_var.get() == "gaussian":
                # Parse downsample setting
                downsample_setting = self.downsample_var.get()
                if downsample_setting == "auto":
                    auto_downsample = True
                    downsample_factor = None
                elif downsample_setting == "1x (none)":
                    auto_downsample = False
                    downsample_factor = 1
                else:
                    auto_downsample = False
                    downsample_factor = int(downsample_setting.split('x')[0])
                
                result = self.gaussian_deblur.deblur_image(
                    self.current_image,
                    kernel_size=self.kernel_size_var.get(),
                    iterations=self.iterations_var.get(),
                    auto_downsample=auto_downsample,
                    downsample_factor=downsample_factor,
                    show_progress=self.show_progress_var.get()
                )
            else:
                result = self.motion_deblur.remove_motion_blur(
                    self.current_image,
                    angle=self.angle_var.get(),
                    length=self.length_var.get()
                )
            
            self.processed_image = result
            
            # Update UI in main thread
            self.root.after(0, self._processing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self._processing_error(str(e)))
    
    def _processing_complete(self):
        """Called when processing is complete."""
        self.progress.stop()
        self.display_image(self.processed_image)
        self.status_var.set("Processing complete!")
    
    def _processing_error(self, error_msg):
        """Called when processing encounters an error."""
        self.progress.stop()
        self.status_var.set("Ready")
        messagebox.showerror("Processing Error", f"Failed to process image: {error_msg}")
    
    def compare_images(self):
        """Show before/after comparison."""
        if self.current_image is None or self.processed_image is None:
            messagebox.showwarning("Warning", "Please load and process an image first")
            return
        
        show_comparison(self.current_image, self.processed_image)
    
    def save_image(self):
        """Save the processed image."""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                save_image(self.processed_image, file_path)
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = DeblurApp()
    app.run()
