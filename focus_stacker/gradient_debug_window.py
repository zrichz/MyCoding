"""
Visual debugging for gradient-based focus stacking
"""

import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import List, Optional
import threading
import time


class GradientStackingDebugWindow(ctk.CTkToplevel):
    """Debug window to visualize gradient-based stacking process."""
    
    def __init__(self, parent, images: List[np.ndarray]):
        super().__init__(parent)
        
        self.title("Gradient Stacking Debug - Process Visualization")
        self.geometry("840x920")  # Increased size for 400x400 images
        self.resizable(False, False)
        
        # Center on parent
        self.transient(parent)
        self.grab_set()
        
        # Store images (downsample for display)
        self.original_images = images
        self.display_images = []
        
        # Downsample images to reasonable size for processing
        target_size = 400  # Process at this size for speed
        for img in images:
            if img.shape[0] > target_size or img.shape[1] > target_size:
                scale = min(target_size / img.shape[1], target_size / img.shape[0])
                new_width = int(img.shape[1] * scale)
                new_height = int(img.shape[0] * scale)
                resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.display_images.append(resized)
            else:
                self.display_images.append(img.copy())
        
        self.current_step = 0
        self.gradient_maps = []
        self.weight_maps = []
        self.normalized_weights = []
        self.setup_ui()
        
        # Start processing in background
        threading.Thread(target=self.process_debug_data, daemon=True).start()
    
    def setup_ui(self):
        """Setup the debug window UI."""
        # Title
        title_label = ctk.CTkLabel(self, text="Gradient-Based Stacking Process", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        # Progress info
        self.progress_label = ctk.CTkLabel(self, text="Processing gradient maps...")
        self.progress_label.pack(pady=5)
        
        # Main grid frame
        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create 2x2 grid of 400x400 canvases
        self.canvases = []
        self.labels = []
        
        for row in range(2):
            for col in range(2):
                # Label for each quadrant
                label = ctk.CTkLabel(self.grid_frame, text=f"Step {row*2 + col + 1}", 
                                   font=ctk.CTkFont(size=12, weight="bold"))
                label.grid(row=row*2, column=col, padx=5, pady=5, sticky="ew")
                self.labels.append(label)
                
                # Canvas for image display - increased to 400x400
                canvas = tk.Canvas(self.grid_frame, width=400, height=400, 
                                 bg='gray20', highlightthickness=1, highlightbackground="gray50")
                canvas.grid(row=row*2 + 1, column=col, padx=5, pady=5)
                self.canvases.append(canvas)
        
        # Control buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.prev_button = ctk.CTkButton(button_frame, text="◀ Previous", 
                                        command=self.prev_step, width=100)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.step_label = ctk.CTkLabel(button_frame, text="Image 1/4")
        self.step_label.pack(side=tk.LEFT, expand=True)
        
        self.next_button = ctk.CTkButton(button_frame, text="Next ▶", 
                                        command=self.next_step, width=100)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        # Info panel
        self.info_text = ctk.CTkTextbox(self, height=80)
        self.info_text.pack(fill=tk.X, padx=10, pady=5)
        
        # Close button
        close_button = ctk.CTkButton(self, text="Close", command=self.destroy)
        close_button.pack(pady=10)
        
        # Initially disable navigation
        self.prev_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
    
    def process_debug_data(self):
        """Process gradient data for visualization."""
        try:
            self.after(0, lambda: self.progress_label.configure(text="Computing gradient maps..."))
            
            # Step 1: Compute gradient maps for each image
            kernel_size = 5
            threshold = 0.1
            
            for i, img in enumerate(self.display_images):
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # Calculate gradients
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
                gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # Apply threshold
                gradient_mag = np.where(gradient_mag > threshold, gradient_mag, 0)
                self.gradient_maps.append(gradient_mag)
                
                time.sleep(0.1)  # Small delay for visual effect
            
            self.after(0, lambda: self.progress_label.configure(text="Computing weight maps..."))
            
            # Step 2: Create weight maps
            gradient_stack = np.stack(self.gradient_maps, axis=-1)
            epsilon = 1e-6
            weight_sum = np.sum(gradient_stack, axis=-1, keepdims=True) + epsilon
            weights = gradient_stack / weight_sum
            
            for i in range(len(self.display_images)):
                self.weight_maps.append(weights[:, :, i])
            
            # Step 3: Apply smoothing (if enabled)
            self.after(0, lambda: self.progress_label.configure(text="Creating smooth weights..."))
            
            # For demo, we'll show both hard and smooth weights
            smooth_radius = 3
            blend_sigma = 1.0
            
            smooth_weights = []
            for i in range(len(self.display_images)):
                weight = self.weight_maps[i].copy()
                
                # Apply morphological operations
                if smooth_radius > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (smooth_radius*2+1, smooth_radius*2+1))
                    weight = cv2.morphologyEx(weight, cv2.MORPH_CLOSE, kernel)
                    weight = cv2.morphologyEx(weight, cv2.MORPH_OPEN, kernel)
                
                # Apply Gaussian smoothing
                if blend_sigma > 0:
                    weight = cv2.GaussianBlur(weight, (0, 0), blend_sigma)
                
                smooth_weights.append(weight)
            
            # Re-normalize
            smooth_weight_stack = np.stack(smooth_weights, axis=-1)
            weight_sum = np.sum(smooth_weight_stack, axis=-1, keepdims=True) + epsilon
            self.normalized_weights = smooth_weight_stack / weight_sum
            
            self.after(0, lambda: self.progress_label.configure(text="Ready! Navigate through images to see the process."))
            self.after(0, self.enable_navigation)
            self.after(0, self.update_display)
            
        except Exception as e:
            self.after(0, lambda: self.progress_label.configure(text=f"Error: {str(e)}"))
    
    def enable_navigation(self):
        """Enable navigation buttons."""
        self.next_button.configure(state="normal")
        if len(self.display_images) > 1:
            self.step_label.configure(text=f"Image 1/{len(self.display_images)}")
    
    def prev_step(self):
        """Go to previous image."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()
    
    def next_step(self):
        """Go to next image."""
        if self.current_step < len(self.display_images) - 1:
            self.current_step += 1
            self.update_display()
    
    def update_display(self):
        """Update the 2x2 grid display."""
        if not self.gradient_maps:
            return
        
        idx = self.current_step
        img = self.display_images[idx]
        
        # Update step label
        self.step_label.configure(text=f"Image {idx + 1}/{len(self.display_images)}")
        
        # Update navigation buttons
        self.prev_button.configure(state="normal" if idx > 0 else "disabled")
        self.next_button.configure(state="normal" if idx < len(self.display_images) - 1 else "disabled")
        
        # Prepare images for display
        display_data = [
            ("Original Image", img),
            ("Gradient Map", self.gradient_maps[idx]),
            ("Weight Map", self.weight_maps[idx]),
            ("Smooth Weights", self.normalized_weights[:, :, idx])
        ]
        
        # Update each quadrant
        for i, (title, data) in enumerate(display_data):
            self.labels[i].configure(text=title)
            
            # Convert data to displayable format
            if data is not None:
                display_img = self.prepare_for_display(data)
                photo = ImageTk.PhotoImage(display_img)
                
                canvas = self.canvases[i]
                canvas.delete("all")
                canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)  # Center at 200,200 for 400x400 canvas
                canvas.image = photo  # Keep reference
        
        # Update info text
        info_text = f"Image {idx + 1} Analysis:\n\n"
        info_text += f"Original size: {img.shape[1]}x{img.shape[0]}\n"
        
        if idx < len(self.gradient_maps):
            grad_map = self.gradient_maps[idx]
            info_text += f"Gradient range: {grad_map.min():.2f} - {grad_map.max():.2f}\n"
            info_text += f"Mean gradient: {grad_map.mean():.2f}\n"
        
        if idx < len(self.weight_maps):
            weight_map = self.weight_maps[idx]
            info_text += f"Weight range: {weight_map.min():.3f} - {weight_map.max():.3f}\n"
            info_text += f"Mean weight: {weight_map.mean():.3f}\n"
        
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", info_text)
    
    def prepare_for_display(self, data):
        """Convert numpy array to PIL Image for display."""
        if len(data.shape) == 3:
            # Color image
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            # Convert BGR to RGB for PIL
            data_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(data_rgb)
        else:
            # Grayscale image - normalize to 0-255
            if data.dtype != np.uint8:
                # Normalize to 0-255 range
                data_norm = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
            else:
                data_norm = data
            pil_img = Image.fromarray(data_norm, mode='L')
        
        # Resize to fit 400x400
        pil_img = pil_img.resize((400, 400), Image.Resampling.LANCZOS)
        return pil_img
