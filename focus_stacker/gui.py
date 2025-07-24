"""
Modern GUI for Focus Stacker
============================
Beautiful and intuitive interface using CustomTkinter.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
from pathlib import Path
from typing import List, Optional, Callable
import logging
import time

from focus_stacking_algorithms import FocusStackingAlgorithms
from image_alignment import ImageAligner
from gradient_debug_window import GradientStackingDebugWindow

logger = logging.getLogger(__name__)

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImagePreviewWidget(ctk.CTkFrame):
    """Custom widget for image preview with zoom and pan."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.image = None
        self.photo = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.canvas = tk.Canvas(self, bg='gray20', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.zoom_image)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_image(e, 1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.zoom_image(e, -1))  # Linux
        
        self.last_x = 0
        self.last_y = 0
    
    def set_image(self, image: np.ndarray):
        """Set the image to display, automatically scaled to fit 800x800."""
        if image is None:
            return
        
        # Convert OpenCV image to PIL
        if len(image.shape) == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = Image.fromarray(image)
        
        # Calculate scale to fit within 800x800 while maintaining aspect ratio
        max_size = 800
        img_width, img_height = image_pil.size
        
        if img_width > max_size or img_height > max_size:
            # Calculate scale factor to fit within max_size
            scale_factor = min(max_size / img_width, max_size / img_height)
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Resize image to fit within 800x800
            image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.image = image_pil
        self.zoom_factor = 1.0  # Reset zoom when setting new image
        self.pan_x = 0  # Reset pan when setting new image
        self.pan_y = 0
        self.update_display()
    
    def update_display(self):
        """Update the canvas display."""
        if self.image is None:
            return
        
        # Calculate display size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.after(100, self.update_display)
            return
        
        # Scale image
        img_width, img_height = self.image.size
        display_width = int(img_width * self.zoom_factor)
        display_height = int(img_height * self.zoom_factor)
        
        # Resize image
        if self.zoom_factor != 1.0:
            display_image = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        else:
            display_image = self.image
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        x = canvas_width // 2 + self.pan_x
        y = canvas_height // 2 + self.pan_y
        self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)
    
    def start_pan(self, event):
        """Start panning."""
        self.last_x = event.x
        self.last_y = event.y
    
    def pan_image(self, event):
        """Pan the image."""
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.pan_x += dx
        self.pan_y += dy
        self.last_x = event.x
        self.last_y = event.y
        self.update_display()
    
    def zoom_image(self, event, delta=None):
        """Zoom the image."""
        if delta is None:
            delta = 1 if event.delta > 0 else -1
        
        zoom_in = delta > 0
        factor = 1.1 if zoom_in else 0.9
        
        new_zoom = self.zoom_factor * factor
        if 0.1 <= new_zoom <= 10.0:
            self.zoom_factor = new_zoom
            self.update_display()
    
    def reset_view(self):
        """Reset zoom and pan."""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_display()


class ProgressDialog(ctk.CTkToplevel):
    """Progress dialog for long operations."""
    
    def __init__(self, parent, title="Processing", message="Please wait..."):
        super().__init__(parent)
        
        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        
        # Center on parent
        self.transient(parent)
        self.grab_set()
        
        # Create widgets
        self.message_label = ctk.CTkLabel(self, text=message, font=ctk.CTkFont(size=14))
        self.message_label.pack(pady=20)
        
        self.progress_bar = ctk.CTkProgressBar(self, width=350)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        self.cancel_button = ctk.CTkButton(self, text="Cancel", command=self.cancel)
        self.cancel_button.pack(pady=10)
        
        self.cancelled = False
    
    def update_progress(self, value: float, message: str = None):
        """Update progress bar and message."""
        self.progress_bar.set(value)
        if message:
            self.message_label.configure(text=message)
        self.update()
    
    def cancel(self):
        """Cancel the operation."""
        self.cancelled = True
        self.destroy()


class FocusStackerGUI:
    """Main GUI class for the Focus Stacker application."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Image Stacking Tool")
        self.root.geometry("1200x900")
        
        # Data
        self.loaded_images = []
        self.aligned_images = []
        self.stacked_image = None
        self.current_preview_index = 0
        
        # Progress tracking
        self.is_processing = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ctk.CTkFrame(main_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Preview
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
        
        # Initialize parameter visibility (show only for Laplacian Pyramid by default)
        self.on_stack_method_change("Laplacian Pyramid")
    
    def setup_left_panel(self, parent):
        """Setup the left control panel."""
        # Title
        title_label = ctk.CTkLabel(parent, text="Focus Stacker", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(20, 10))
        
        # Image loading section
        load_frame = ctk.CTkFrame(parent)
        load_frame.pack(fill=tk.X, padx=20, pady=10)
        
        load_title = ctk.CTkLabel(load_frame, text="Load Images", 
                                 font=ctk.CTkFont(size=16, weight="bold"))
        load_title.pack(pady=(10, 5))
        
        load_button = ctk.CTkButton(load_frame, text="Select Images", 
                                   command=self.load_images)
        load_button.pack(pady=5)
        
        self.image_count_label = ctk.CTkLabel(load_frame, text="No images loaded")
        self.image_count_label.pack(pady=5)
        
        # Preview controls
        preview_frame = ctk.CTkFrame(parent)
        preview_frame.pack(fill=tk.X, padx=20, pady=10)
        
        preview_title = ctk.CTkLabel(preview_frame, text="Preview", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        preview_title.pack(pady=(10, 5))
        
        nav_frame = ctk.CTkFrame(preview_frame)
        nav_frame.pack(pady=5)
        
        self.prev_button = ctk.CTkButton(nav_frame, text="◀", width=50,
                                        command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=2)
        
        self.image_index_label = ctk.CTkLabel(nav_frame, text="0/0")
        self.image_index_label.pack(side=tk.LEFT, padx=10)
        
        self.next_button = ctk.CTkButton(nav_frame, text="▶", width=50,
                                        command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=2)
        
        # Alignment settings
        align_frame = ctk.CTkFrame(parent)
        align_frame.pack(fill=tk.X, padx=20, pady=10)
        
        align_title = ctk.CTkLabel(align_frame, text="Alignment", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        align_title.pack(pady=(10, 5))
        
        self.align_method = ctk.CTkOptionMenu(align_frame, 
                                             values=["Auto", "ECC", "Feature-based", "Phase Correlation", "None"])
        self.align_method.pack(pady=5)
        self.align_method.set("Auto")
        
        # Alignment optimization settings
        align_opt_frame = ctk.CTkFrame(align_frame)
        align_opt_frame.pack(fill=tk.X, pady=5)
        
        self.use_proxy_var = ctk.BooleanVar(value=True)
        self.use_proxy_checkbox = ctk.CTkCheckBox(align_opt_frame, 
                                                 text="Use fast proxy alignment (4x faster)", 
                                                 variable=self.use_proxy_var)
        self.use_proxy_checkbox.pack(anchor=tk.W, pady=2)
        
        align_button = ctk.CTkButton(align_frame, text="Align Images", 
                                    command=self.align_images)
        align_button.pack(pady=5)
        
        # Stacking settings
        stack_frame = ctk.CTkFrame(parent)
        stack_frame.pack(fill=tk.X, padx=20, pady=10)
        
        stack_title = ctk.CTkLabel(stack_frame, text="Focus Stacking", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        stack_title.pack(pady=(10, 5))
        
        self.stack_method = ctk.CTkOptionMenu(stack_frame, 
                                             values=["Laplacian Pyramid", "Gradient-based", "Variance-based", "Simple Average"],
                                             command=self.on_stack_method_change)
        self.stack_method.pack(pady=5)
        self.stack_method.set("Laplacian Pyramid")
        
        # Algorithm parameters (only for Laplacian Pyramid)
        self.param_frame = ctk.CTkFrame(stack_frame)
        self.param_frame.pack(fill=tk.X, pady=5)
        
        # Pyramid levels
        self.pyramid_label = ctk.CTkLabel(self.param_frame, text="Pyramid Levels:")
        self.pyramid_label.pack(anchor=tk.W)
        self.pyramid_levels = ctk.CTkSlider(self.param_frame, from_=3, to=8, number_of_steps=5,
                                          command=self.update_pyramid_value)
        self.pyramid_levels.pack(fill=tk.X, pady=2)
        self.pyramid_levels.set(5)
        
        # Pyramid levels value display
        self.pyramid_value_label = ctk.CTkLabel(self.param_frame, text="Value: 5",
                                              font=ctk.CTkFont(size=12, weight="bold"),
                                              text_color="lightgreen")
        self.pyramid_value_label.pack(anchor=tk.W, pady=(2, 5))
        
        # Sigma
        self.sigma_label = ctk.CTkLabel(self.param_frame, text="Gaussian Sigma:")
        self.sigma_label.pack(anchor=tk.W)
        
        # Sigma description
        self.sigma_desc_label = ctk.CTkLabel(self.param_frame, 
                                           text="(approx 0.5 to 2.0, recommended 1.0)",
                                           font=ctk.CTkFont(size=11), 
                                           text_color="gray60")
        self.sigma_desc_label.pack(anchor=tk.W, pady=(0, 2))
        
        self.gaussian_sigma = ctk.CTkSlider(self.param_frame, from_=0.1, to=2.5, number_of_steps=48,
                                          command=self.update_sigma_value)
        self.gaussian_sigma.pack(fill=tk.X, pady=2)
        self.gaussian_sigma.set(1.0)
        
        # Sigma value display
        self.sigma_value_label = ctk.CTkLabel(self.param_frame, text="Value: 1.0",
                                            font=ctk.CTkFont(size=12, weight="bold"),
                                            text_color="lightblue")
        self.sigma_value_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Gradient smoothing controls (for Gradient-based method)
        self.smooth_label = ctk.CTkLabel(self.param_frame, text="Smoothing Radius:")
        self.smooth_desc_label = ctk.CTkLabel(self.param_frame, 
                                            text="(0=sharp, 3-5=smooth, reduces noise)",
                                            font=ctk.CTkFont(size=11), 
                                            text_color="gray60")
        self.smooth_radius = ctk.CTkSlider(self.param_frame, from_=0, to=10, number_of_steps=10,
                                         command=self.update_smooth_value)
        self.smooth_radius.set(3)
        
        # Smooth radius value display
        self.smooth_value_label = ctk.CTkLabel(self.param_frame, text="Value: 3",
                                             font=ctk.CTkFont(size=12, weight="bold"),
                                             text_color="lightyellow")
        
        # Debug button for gradient method
        self.debug_button = ctk.CTkButton(self.param_frame, text="Debug Process 🔍", 
                                         command=self.show_gradient_debug,
                                         height=32, font=ctk.CTkFont(size=12))
        
        stack_button = ctk.CTkButton(stack_frame, text="Stack Images", 
                                    command=self.stack_images)
        stack_button.pack(pady=10)
        
        # Export
        export_frame = ctk.CTkFrame(parent)
        export_frame.pack(fill=tk.X, padx=20, pady=10)
        
        export_title = ctk.CTkLabel(export_frame, text="Export", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        export_title.pack(pady=(10, 5))
        
        export_button = ctk.CTkButton(export_frame, text="Save Result", 
                                     command=self.save_result)
        export_button.pack(pady=5)
        
        # Progress section
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        progress_title = ctk.CTkLabel(progress_frame, text="Progress", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        progress_title.pack(pady=(10, 5))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Processing details
        self.details_text = ctk.CTkTextbox(progress_frame, height=100)
        self.details_text.pack(fill=tk.X, padx=10, pady=5)
        
        # Status
        self.status_label = ctk.CTkLabel(parent, text="Ready")
        self.status_label.pack(side=tk.BOTTOM, pady=20)
    
    def setup_right_panel(self, parent):
        """Setup the right preview panel."""
        # Preview title
        preview_title = ctk.CTkLabel(parent, text="Image Preview", 
                                    font=ctk.CTkFont(size=18, weight="bold"))
        preview_title.pack(pady=(20, 10))
        
        # Preview controls
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.preview_mode = ctk.CTkOptionMenu(control_frame, 
                                             values=["Original", "Aligned", "Stacked"])
        self.preview_mode.pack(side=tk.LEFT, padx=5)
        self.preview_mode.set("Original")
        
        reset_view_button = ctk.CTkButton(control_frame, text="Reset View", 
                                         command=self.reset_preview_view)
        reset_view_button.pack(side=tk.RIGHT, padx=5)
        
        # Image preview
        self.preview_widget = ImagePreviewWidget(parent)
        self.preview_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Bind preview mode change
        self.preview_mode.configure(command=self.update_preview)
    
    def update_status(self, message, progress=None):
        """Update status and progress."""
        self.status_label.configure(text=message)
        if progress is not None:
            self.progress_bar.set(progress / 100.0)
        
        # Add to details
        timestamp = time.strftime("%H:%M:%S")
        self.details_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.details_text.see(tk.END)
        
        self.root.update()
    
    def clear_status(self):
        """Clear status and progress."""
        self.details_text.delete(1.0, tk.END)
        self.progress_bar.set(0)
        self.status_label.configure(text="Ready")
    
    def load_images(self):
        """Load images for focus stacking."""
        file_paths = filedialog.askopenfilenames(
            title="Select images for focus stacking",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("TIFF files", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        self.status_label.configure(text="Loading images...")
        self.root.update()
        
        # Load images
        self.loaded_images = []
        for path in file_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    self.loaded_images.append(img)
                else:
                    logger.warning(f"Failed to load image: {path}")
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
        
        if self.loaded_images:
            self.image_count_label.configure(text=f"{len(self.loaded_images)} images loaded")
            self.current_preview_index = 0
            self.update_navigation()
            self.update_preview()
            self.status_label.configure(text="Images loaded successfully")
        else:
            messagebox.showerror("Error", "No valid images could be loaded.")
            self.status_label.configure(text="Failed to load images")
    
    def prev_image(self):
        """Show previous image."""
        if self.loaded_images and self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.update_navigation()
            self.update_preview()
    
    def next_image(self):
        """Show next image."""
        if self.loaded_images and self.current_preview_index < len(self.loaded_images) - 1:
            self.current_preview_index += 1
            self.update_navigation()
            self.update_preview()
    
    def update_navigation(self):
        """Update navigation controls."""
        if self.loaded_images:
            total = len(self.loaded_images)
            current = self.current_preview_index + 1
            self.image_index_label.configure(text=f"{current}/{total}")
            
            self.prev_button.configure(state="normal" if self.current_preview_index > 0 else "disabled")
            self.next_button.configure(state="normal" if self.current_preview_index < total - 1 else "disabled")
        else:
            self.image_index_label.configure(text="0/0")
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
    
    def update_preview(self, *args):
        """Update the preview image."""
        mode = self.preview_mode.get()
        
        if mode == "Original" and self.loaded_images:
            img = self.loaded_images[self.current_preview_index]
            self.preview_widget.set_image(img)
        elif mode == "Aligned" and self.aligned_images:
            img = self.aligned_images[self.current_preview_index]
            self.preview_widget.set_image(img)
        elif mode == "Stacked" and self.stacked_image is not None:
            self.preview_widget.set_image(self.stacked_image)
    
    def reset_preview_view(self):
        """Reset preview zoom and pan."""
        self.preview_widget.reset_view()
    
    def update_pyramid_value(self, value):
        """Update the pyramid levels value display when slider changes."""
        self.pyramid_value_label.configure(text=f"Value: {int(value)}")
    
    def update_sigma_value(self, value):
        """Update the sigma value display when slider changes."""
        self.sigma_value_label.configure(text=f"Value: {value:.2f}")
    
    def update_smooth_value(self, value):
        """Update the smooth radius value display when slider changes."""
        self.smooth_value_label.configure(text=f"Value: {int(value)}")
    
    def on_stack_method_change(self, method):
        """Handle stacking method change to show/hide parameters."""
        # Hide all parameters first
        self.pyramid_label.pack_forget()
        self.pyramid_levels.pack_forget()
        self.pyramid_value_label.pack_forget()
        self.sigma_label.pack_forget()
        self.sigma_desc_label.pack_forget()
        self.gaussian_sigma.pack_forget()
        self.sigma_value_label.pack_forget()
        self.smooth_label.pack_forget()
        self.smooth_desc_label.pack_forget()
        self.smooth_radius.pack_forget()
        self.smooth_value_label.pack_forget()
        self.debug_button.pack_forget()
        
        # Show relevant parameters
        if method == "Laplacian Pyramid":
            self.pyramid_label.pack(anchor=tk.W)
            self.pyramid_levels.pack(fill=tk.X, pady=2)
            self.pyramid_value_label.pack(anchor=tk.W, pady=(2, 5))
            self.sigma_label.pack(anchor=tk.W)
            self.sigma_desc_label.pack(anchor=tk.W, pady=(0, 2))
            self.gaussian_sigma.pack(fill=tk.X, pady=2)
            self.sigma_value_label.pack(anchor=tk.W, pady=(2, 0))
        elif method == "Gradient-based":
            self.smooth_label.pack(anchor=tk.W)
            self.smooth_desc_label.pack(anchor=tk.W, pady=(0, 2))
            self.smooth_radius.pack(fill=tk.X, pady=2)
            self.smooth_value_label.pack(anchor=tk.W, pady=(2, 5))
            self.debug_button.pack(fill=tk.X, pady=(5, 0))
    
    def show_gradient_debug(self):
        """Show gradient-based stacking debug window."""
        images_to_debug = self.aligned_images if self.aligned_images else self.loaded_images
        
        if not images_to_debug:
            messagebox.showwarning("Warning", "Please load images first.")
            return
        
        # Limit to first 4 images for debugging to keep it manageable
        debug_images = images_to_debug[:4]
        debug_window = GradientStackingDebugWindow(self.root, debug_images)
    
    def align_images(self):
        """Align loaded images."""
        if not self.loaded_images:
            messagebox.showwarning("Warning", "Please load images first.")
            return
        
        method = self.align_method.get().lower().replace("-", "_").replace(" ", "_")
        if method == "none":
            self.aligned_images = self.loaded_images.copy()
            self.status_label.configure(text="No alignment applied")
            return
        
        # Show progress dialog
        progress = ProgressDialog(self.root, "Aligning Images", "Aligning images...")
        
        def progress_callback(prog_value, message):
            """Progress callback for alignment."""
            if not progress.cancelled:
                progress.update_progress(prog_value, message)
        
        def align_thread():
            try:
                progress_callback(0.05, "Initializing alignment...")
                
                # Get proxy setting
                use_proxy = self.use_proxy_var.get()
                
                if method == "auto":
                    aligned, align_time = ImageAligner.auto_align(self.loaded_images, progress_callback=progress_callback)
                elif method == "ecc":
                    aligned, align_time = ImageAligner.align_images_ecc(self.loaded_images, 
                                                                      use_proxy=use_proxy, 
                                                                      proxy_scale=0.25,
                                                                      progress_callback=progress_callback)
                elif method == "feature_based":
                    aligned, align_time = ImageAligner.align_images_feature_based(self.loaded_images, progress_callback=progress_callback)
                elif method == "phase_correlation":
                    aligned, align_time = ImageAligner.align_images_phase_correlation(self.loaded_images, progress_callback=progress_callback)
                else:
                    aligned = self.loaded_images.copy()
                    align_time = 0.0
                
                if not progress.cancelled:
                    self.aligned_images = aligned
                    proxy_msg = " (using fast proxy)" if use_proxy and method == "ecc" else ""
                    self.root.after(0, lambda: self.status_label.configure(text=f"Images aligned successfully in {align_time:.2f}s{proxy_msg}"))
                    self.root.after(0, self.update_preview)
                
            except Exception as e:
                if not progress.cancelled:
                    logger.error(f"Alignment failed: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Alignment failed: {e}"))
                    self.root.after(0, lambda: self.status_label.configure(text="Alignment failed"))
            finally:
                self.root.after(0, progress.destroy)
        
        threading.Thread(target=align_thread, daemon=True).start()
    
    def stack_images(self):
        """Stack the images using the selected algorithm with progress dialog."""
        images_to_stack = self.aligned_images if self.aligned_images else self.loaded_images
        
        if not images_to_stack:
            messagebox.showwarning("Warning", "Please load images first.")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Already processing images...")
            return
        
        self.is_processing = True
        
        # Show progress dialog
        progress = ProgressDialog(self.root, "Stacking Images", "Starting focus stacking...")
        
        def progress_callback(prog_value, message):
            """Progress callback for stacking."""
            if not progress.cancelled:
                progress.update_progress(prog_value, message)
        
        def algorithm_progress_callback(message):
            """Progress callback adapter for stacking algorithms (they only expect message)."""
            # Estimate progress based on message content
            prog_value = 0.1
            if "Converting" in message:
                prog_value = 0.2
            elif "Computing" in message or "Building" in message:
                prog_value = 0.4
            elif "Processing" in message or "Selecting" in message:
                prog_value = 0.6
            elif "Adding" in message or "Reconstructing" in message:
                prog_value = 0.8
            elif "complete" in message.lower() or "finished" in message.lower():
                prog_value = 0.9
            
            progress_callback(prog_value, message)
        
        def stack_thread():
            try:
                progress_callback(0.05, "Initializing focus stacking...")
                
                # Get parameters
                method = self.stack_method.get()
                levels = int(self.pyramid_levels.get())
                sigma = self.gaussian_sigma.get()
                
                progress_callback(0.1, f"Stacking using {method} method...")
                
                if method == "Laplacian Pyramid":
                    result = FocusStackingAlgorithms.laplacian_pyramid_stack(
                        images_to_stack, levels=levels, sigma=sigma, progress_callback=algorithm_progress_callback)
                elif method == "Gradient-based":
                    # Get smoothing parameters
                    smooth_radius = int(self.smooth_radius.get())
                    result = FocusStackingAlgorithms.gradient_based_stack(
                        images_to_stack, smooth_radius=smooth_radius, progress_callback=algorithm_progress_callback)
                elif method == "Variance-based":
                    result = FocusStackingAlgorithms.variance_based_stack(images_to_stack, progress_callback=algorithm_progress_callback)
                elif method == "Simple Average":
                    result = FocusStackingAlgorithms.average_stack(images_to_stack, progress_callback=algorithm_progress_callback)
                else:
                    raise ValueError(f"Unknown stacking method: {method}")
                
                if not progress.cancelled:
                    progress_callback(1.0, "Stacking completed!")
                    
                    # Store results
                    self.stacked_image = result
                    self.root.after(0, self.update_preview)
                    self.root.after(0, lambda: self.status_label.configure(text="Stacking complete!"))
                
            except Exception as e:
                if not progress.cancelled:
                    logger.error(f"Stacking failed: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Stacking failed: {e}"))
                    self.root.after(0, lambda: self.status_label.configure(text="Stacking failed"))
            finally:
                self.is_processing = False
                self.root.after(0, progress.destroy)
        
        threading.Thread(target=stack_thread, daemon=True).start()
    
    def save_result(self):
        """Save the stacked result."""
        if self.stacked_image is None:
            messagebox.showwarning("Warning", "No stacked image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save stacked image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("TIFF files", "*.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.stacked_image)
                messagebox.showinfo("Success", f"Image saved successfully to {file_path}")
                self.status_label.configure(text=f"Image saved to {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
                self.status_label.configure(text="Failed to save image")
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = FocusStackerGUI()
    app.run()
