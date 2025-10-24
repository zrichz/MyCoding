# Filmic Effects Processor v3 - Applies film grain and vignette effects to images of any size

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageChops
import numpy as np
import os
import threading
from pathlib import Path

# Face detection imports
try:
    import cv2
    import mediapipe as mp
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("Warning: Face detection not available. Install opencv-python and mediapipe for face-centered effects.")

class FilmicEffectsProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Filmic Effects Processor v3 - Any Image Size")
        self.root.geometry("1800x1200")
        self.root.configure(bg="#EBE1D2")
        
        # Variables
        self.input_directory = None
        self.preview_image = None
        self.processed_preview = None
        self.image_files = []
        self.current_preview_index = 0
        
        # Image references for canvas display (prevent garbage collection)
        self.original_photo_ref = None
        self.processed_photo_ref = None
        
        # Dynamic image dimensions
        self.current_image_width = 0
        self.current_image_height = 0
        self.preview_scale = 1.0
        self.max_canvas_width = 800
        self.max_canvas_height = 900
        
        # Default values for effects
        self.DEFAULT_GRAIN = 0.22
        self.DEFAULT_VIGNETTE = 0.34
        self.DEFAULT_SATURATION = 0.30
        self.DEFAULT_CHROMATIC = 0.09
        
        # Effect parameters
        self.grain_intensity = tk.DoubleVar(value=self.DEFAULT_GRAIN)
        self.grain_edge_boost = tk.DoubleVar(value=1.8)
        self.vignette_strength = tk.DoubleVar(value=self.DEFAULT_VIGNETTE)
        self.saturation_reduction = tk.DoubleVar(value=self.DEFAULT_SATURATION)
        self.chromatic_aberration = tk.DoubleVar(value=self.DEFAULT_CHROMATIC)
        
        # Face detection
        if FACE_DETECTION_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Long-range model for better small face detection
                min_detection_confidence=0.3  # Lower threshold for small faces
            )
        else:
            self.mp_face_detection = None
            
        # Effect center (default to image center, updated by face detection)
        self.effect_center = (0, 0)
        
        # Border effect settings
        self.photo_border = tk.BooleanVar(value=False)
        self.unsharp_mask = tk.BooleanVar(value=False)
        self.auto_contrast = tk.BooleanVar(value=False)
        self.show_ca_center = tk.BooleanVar(value=False)
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main container
        main_container = tk.Frame(self.root, bg="#EBE1D2")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section for controls and image info
        top_section = tk.Frame(main_container, bg="#EBE1D2")
        top_section.pack(fill=tk.X, pady=(0, 10))
        
        # Left controls frame
        left_controls = tk.Frame(top_section, bg="#EBE1D2")
        left_controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Directory selection
        dir_frame = tk.LabelFrame(left_controls, text="Directory Selection", 
                                 font=("Arial", 10, "bold"), bg="#EBE1D2")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(dir_frame, text="Select Image Directory", command=self.browse_directory,
                 font=("Arial", 9), bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=5)
        
        self.dir_label = tk.Label(dir_frame, text="No directory selected", 
                                 font=("Arial", 8), bg="#EBE1D2", wraplength=200)
        self.dir_label.pack(pady=(0, 5))
        
        # Image info frame
        info_frame = tk.LabelFrame(left_controls, text="Current Image Info", 
                                  font=("Arial", 10, "bold"), bg="#EBE1D2")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.image_info_label = tk.Label(info_frame, text="No image loaded", 
                                        font=("Arial", 8), bg="#EBE1D2", wraplength=200)
        self.image_info_label.pack(pady=5)
        
        # Effect controls
        effects_frame = tk.LabelFrame(left_controls, text="Filmic Effects", 
                                     font=("Arial", 10, "bold"), bg="#EBE1D2")
        effects_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Grain intensity
        self.create_slider(effects_frame, "Grain Intensity", self.grain_intensity, 
                          0.0, 1.0, self.update_preview)
        
        # Vignette strength
        self.create_slider(effects_frame, "Vignette Strength", self.vignette_strength, 
                          0.0, 1.0, self.update_preview)
        
        # Saturation reduction
        self.create_slider(effects_frame, "Saturation Reduction", self.saturation_reduction, 
                          0.0, 1.0, self.update_preview)
        
        # Chromatic aberration
        self.create_slider(effects_frame, "Chromatic Aberration", self.chromatic_aberration, 
                          0.0, 0.5, self.update_preview)
        
        # Additional effects frame
        additional_frame = tk.LabelFrame(left_controls, text="Additional Effects", 
                                        font=("Arial", 10, "bold"), bg="#EBE1D2")
        additional_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Checkboxes for additional effects
        tk.Checkbutton(additional_frame, text="Photo Border", variable=self.photo_border,
                      command=self.update_preview, font=("Arial", 8), bg="#EBE1D2").pack(anchor='w')
        
        tk.Checkbutton(additional_frame, text="Unsharp Mask", variable=self.unsharp_mask,
                      command=self.update_preview, font=("Arial", 8), bg="#EBE1D2").pack(anchor='w')
        
        tk.Checkbutton(additional_frame, text="Auto Contrast", variable=self.auto_contrast,
                      command=self.update_preview, font=("Arial", 8), bg="#EBE1D2").pack(anchor='w')
        
        tk.Checkbutton(additional_frame, text="Show CA Center", variable=self.show_ca_center,
                      command=self.update_preview, font=("Arial", 8), bg="#EBE1D2").pack(anchor='w')
        
        # Control buttons
        buttons_frame = tk.Frame(left_controls, bg="#EBE1D2")
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(buttons_frame, text="Reset All Defaults", command=self.reset_defaults,
                 font=("Arial", 9), bg="#FF9800", fg="white", padx=10).pack(fill=tk.X, pady=2)
        
        tk.Button(buttons_frame, text="Process All Images", command=self.process_all_images,
                 font=("Arial", 9), bg="#2196F3", fg="white", padx=10).pack(fill=tk.X, pady=2)
        
        # Right side - image comparison
        right_section = tk.Frame(top_section, bg="#EBE1D2")
        right_section.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image navigation
        nav_frame = tk.Frame(right_section, bg="#EBE1D2")
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(nav_frame, text="◀ Previous", command=self.previous_image,
                 font=("Arial", 9), padx=10).pack(side=tk.LEFT, padx=(0, 5))
        
        self.image_counter_label = tk.Label(nav_frame, text="No images", 
                                           font=("Arial", 9), bg="#EBE1D2")
        self.image_counter_label.pack(side=tk.LEFT, padx=10)
        
        tk.Button(nav_frame, text="Next ▶", command=self.next_image,
                 font=("Arial", 9), padx=10).pack(side=tk.LEFT, padx=(5, 0))
        
        # Image comparison area
        comparison_frame = tk.Frame(right_section, bg="#EBE1D2")
        comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        original_frame = tk.LabelFrame(comparison_frame, text="Original", 
                                      font=("Arial", 10, "bold"), bg="#EBE1D2")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Original canvas with scrollbars
        original_canvas_frame = tk.Frame(original_frame, bg="#EBE1D2")
        original_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_canvas = tk.Canvas(original_canvas_frame, bg="white", 
                                        highlightthickness=1, highlightbackground="gray")
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        original_v_scroll = tk.Scrollbar(original_canvas_frame, orient=tk.VERTICAL, 
                                        command=self.original_canvas.yview)
        original_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.original_canvas.configure(yscrollcommand=original_v_scroll.set)
        
        original_h_scroll = tk.Scrollbar(original_frame, orient=tk.HORIZONTAL, 
                                        command=self.original_canvas.xview)
        original_h_scroll.pack(side=tk.BOTTOM, fill=tk.X, padx=5)
        self.original_canvas.configure(xscrollcommand=original_h_scroll.set)
        
        # Processed image frame
        processed_frame = tk.LabelFrame(comparison_frame, text="Processed", 
                                       font=("Arial", 10, "bold"), bg="#EBE1D2")
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Processed canvas with scrollbars
        processed_canvas_frame = tk.Frame(processed_frame, bg="#EBE1D2")
        processed_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_canvas = tk.Canvas(processed_canvas_frame, bg="white", 
                                         highlightthickness=1, highlightbackground="gray")
        self.processed_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        processed_v_scroll = tk.Scrollbar(processed_canvas_frame, orient=tk.VERTICAL, 
                                         command=self.processed_canvas.yview)
        processed_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.processed_canvas.configure(yscrollcommand=processed_v_scroll.set)
        
        processed_h_scroll = tk.Scrollbar(processed_frame, orient=tk.HORIZONTAL, 
                                         command=self.processed_canvas.xview)
        processed_h_scroll.pack(side=tk.BOTTOM, fill=tk.X, padx=5)
        self.processed_canvas.configure(xscrollcommand=processed_h_scroll.set)
        
        # Sync scrolling between canvases
        self.setup_synchronized_scrolling()
        
        # Status bar
        status_frame = tk.Frame(main_container, bg="#EBE1D2")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = tk.Label(status_frame, text="Ready - Select a directory to begin", 
                                    font=("Arial", 9), bg="#EBE1D2", anchor='w')
        self.status_label.pack(fill=tk.X)
        
    def create_slider(self, parent, label, variable, min_val, max_val, callback):
        """Create a labeled slider control"""
        frame = tk.Frame(parent, bg="#EBE1D2")
        frame.pack(fill=tk.X, pady=2)
        
        tk.Label(frame, text=f"{label}:", font=("Arial", 8), bg="#EBE1D2").pack(anchor='w')
        
        slider = tk.Scale(frame, from_=min_val, to=max_val, resolution=0.01, 
                         orient=tk.HORIZONTAL, variable=variable, command=lambda x: callback(),
                         font=("Arial", 8), bg="#EBE1D2", highlightthickness=0)
        slider.pack(fill=tk.X)
        
        return slider
        
    def setup_synchronized_scrolling(self):
        """Set up synchronized scrolling between original and processed canvases"""
        def sync_scroll_vertical(*args):
            self.original_canvas.yview(*args)
            self.processed_canvas.yview(*args)
        
        def sync_scroll_horizontal(*args):
            self.original_canvas.xview(*args)
            self.processed_canvas.xview(*args)
        
        # Bind mouse wheel events
        def on_mousewheel(event):
            canvas = event.widget
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            # Sync the other canvas
            if canvas == self.original_canvas:
                self.processed_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                self.original_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.original_canvas.bind("<MouseWheel>", on_mousewheel)
        self.processed_canvas.bind("<MouseWheel>", on_mousewheel)
        
    def calculate_preview_scale(self, image_width, image_height):
        """Calculate the scale factor to fit image in canvas while maintaining aspect ratio"""
        width_scale = self.max_canvas_width / image_width
        height_scale = self.max_canvas_height / image_height
        
        # Use the smaller scale to ensure the image fits entirely
        scale = min(width_scale, height_scale, 1.0)  # Don't scale up
        return scale
        
    def update_canvas_size(self, width, height):
        """Update canvas sizes based on image dimensions"""
        # Update canvas scroll regions
        self.original_canvas.configure(scrollregion=(0, 0, width, height))
        self.processed_canvas.configure(scrollregion=(0, 0, width, height))
        
    def browse_directory(self):
        """Browse for image directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.input_directory = Path(directory)
            self.dir_label.config(text=f"...{str(self.input_directory)[-30:]}")
            self.load_image_list()
            
    def load_image_list(self):
        """Load list of supported image files from directory"""
        if not self.input_directory:
            return
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = []
        
        for ext in image_extensions:
            self.image_files.extend(self.input_directory.glob(f"*{ext}"))
            self.image_files.extend(self.input_directory.glob(f"*{ext.upper()}"))
        
        self.image_files.sort()
        self.current_preview_index = 0
        
        if self.image_files:
            self.update_image_counter()
            self.load_preview_image()
            self.status_label.config(text=f"Loaded {len(self.image_files)} images")
        else:
            self.status_label.config(text="No supported images found in directory")
            self.image_counter_label.config(text="No images")
            
    def update_image_counter(self):
        """Update the image counter display"""
        if self.image_files:
            current = self.current_preview_index + 1
            total = len(self.image_files)
            self.image_counter_label.config(text=f"Image {current} of {total}")
        else:
            self.image_counter_label.config(text="No images")
            
    def load_preview_image(self):
        """Load current preview image"""
        if not self.image_files or self.current_preview_index >= len(self.image_files):
            return
            
        try:
            image_path = self.image_files[self.current_preview_index]
            self.preview_image = Image.open(image_path)
            
            # Store original dimensions
            self.current_image_width = self.preview_image.width
            self.current_image_height = self.preview_image.height
            
            # Update image info
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
            self.image_info_label.config(
                text=f"File: {image_path.name}\\n"
                     f"Size: {self.current_image_width}x{self.current_image_height}\\n"
                     f"File Size: {file_size:.1f} MB"
            )
            
            # Calculate preview scale
            self.preview_scale = self.calculate_preview_scale(
                self.current_image_width, self.current_image_height
            )
            
            # Update canvas sizes for actual image dimensions
            self.update_canvas_size(self.current_image_width, self.current_image_height)
            
            # Detect face center if face detection is available
            if FACE_DETECTION_AVAILABLE:
                self.detect_face_center()
            else:
                # Default to image center
                self.effect_center = (self.current_image_width // 2, self.current_image_height // 2)
            
            self.update_preview()
            self.status_label.config(text=f"Loaded: {image_path.name} ({self.current_image_width}x{self.current_image_height})")
            
        except Exception as e:
            self.status_label.config(text=f"Error loading image: {str(e)}")
            
    def detect_face_center(self):
        """Detect face center using MediaPipe Face Detection"""
        if not self.mp_face_detection or not self.preview_image:
            self.effect_center = (self.current_image_width // 2, self.current_image_height // 2)
            return
            
        try:
            # Convert PIL image to RGB numpy array
            image_rgb = np.array(self.preview_image.convert('RGB'))
            
            # Process with MediaPipe
            results = self.mp_face_detection.process(image_rgb)
            
            if results.detections:
                # Use the first detected face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Calculate center point of the bounding box
                center_x = int((bbox.xmin + bbox.width / 2) * self.current_image_width)
                center_y = int((bbox.ymin + bbox.height / 2) * self.current_image_height)
                
                # Ensure center is within image bounds
                center_x = max(0, min(center_x, self.current_image_width - 1))
                center_y = max(0, min(center_y, self.current_image_height - 1))
                
                self.effect_center = (center_x, center_y)
                print(f"Face detected at: {self.effect_center}")
            else:
                # No face detected, use image center
                self.effect_center = (self.current_image_width // 2, self.current_image_height // 2)
                print("No face detected, using image center")
                
        except Exception as e:
            print(f"Face detection error: {e}")
            self.effect_center = (self.current_image_width // 2, self.current_image_height // 2)
            
    def next_image(self):
        """Navigate to next image"""
        if self.image_files and self.current_preview_index < len(self.image_files) - 1:
            self.current_preview_index += 1
            self.update_image_counter()
            self.load_preview_image()
            
    def previous_image(self):
        """Navigate to previous image"""
        if self.image_files and self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.update_image_counter()
            self.load_preview_image()
            
    def reset_defaults(self):
        """Reset all effect parameters to defaults"""
        self.grain_intensity.set(self.DEFAULT_GRAIN)
        self.grain_edge_boost.set(1.8)
        self.vignette_strength.set(self.DEFAULT_VIGNETTE)
        self.saturation_reduction.set(self.DEFAULT_SATURATION)
        self.chromatic_aberration.set(self.DEFAULT_CHROMATIC)
        self.photo_border.set(False)
        self.unsharp_mask.set(False)
        self.auto_contrast.set(False)
        self.show_ca_center.set(False)
        self.update_preview()
        
    def apply_filmic_effects(self, image):
        """Apply all filmic effects to the image"""
        if image is None:
            return None
            
        try:
            # Work with a copy
            processed = image.copy()
            
            # Apply auto contrast if enabled
            if self.auto_contrast.get():
                processed = self.apply_auto_contrast(processed)
            
            # Apply chromatic aberration
            if self.chromatic_aberration.get() > 0:
                processed = self.apply_chromatic_aberration(processed)
            
            # Apply vignette
            if self.vignette_strength.get() > 0:
                processed = self.apply_vignette(processed)
            
            # Apply grain
            if self.grain_intensity.get() > 0:
                processed = self.apply_film_grain(processed)
            
            # Apply saturation reduction
            if self.saturation_reduction.get() > 0:
                processed = self.apply_saturation_reduction(processed)
            
            # Apply unsharp mask
            if self.unsharp_mask.get():
                processed = self.apply_unsharp_mask(processed)
            
            # Apply photo border
            if self.photo_border.get():
                processed = self.apply_photo_border(processed)
            
            return processed
            
        except Exception as e:
            print(f"Error applying effects: {e}")
            return image
            
    def apply_auto_contrast(self, image):
        """Apply automatic contrast enhancement"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate percentiles for contrast stretching
            p2, p98 = np.percentile(img_array, (2, 98))
            
            # Apply contrast stretching
            img_array = np.clip((img_array - p2) / (p98 - p2) * 255, 0, 255)
            
            return Image.fromarray(img_array.astype(np.uint8))
        except:
            return image
            
    def apply_chromatic_aberration(self, image):
        """Apply chromatic aberration effect centered on detected face or image center"""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Get effect center
            center_x, center_y = self.effect_center
            
            # Scale aberration strength based on image size
            base_strength = self.chromatic_aberration.get()
            strength = base_strength * max(width, height) / 1000  # Normalize to image size
            
            # Create coordinate grids
            y, x = np.ogrid[:height, :width]
            
            # Calculate distance from center
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Normalize distance
            max_distance = np.sqrt(center_x**2 + center_y**2)
            if max_distance > 0:
                normalized_distance = distance / max_distance
            else:
                normalized_distance = distance
            
            # Calculate displacement (quadratic falloff)
            displacement = strength * normalized_distance**2
            
            # Apply chromatic aberration
            result = img_array.copy()
            
            # Red channel - shift outward
            red_shift_x = (dx / (distance + 1e-6)) * displacement
            red_shift_y = (dy / (distance + 1e-6)) * displacement
            
            # Blue channel - shift inward
            blue_shift_x = -red_shift_x * 0.5
            blue_shift_y = -red_shift_y * 0.5
            
            # Apply shifts using numpy indexing
            for i in range(height):
                for j in range(width):
                    # Red channel
                    new_x = int(j + red_shift_x[i, j])
                    new_y = int(i + red_shift_y[i, j])
                    if 0 <= new_x < width and 0 <= new_y < height:
                        result[i, j, 0] = img_array[new_y, new_x, 0]
                    
                    # Blue channel
                    new_x = int(j + blue_shift_x[i, j])
                    new_y = int(i + blue_shift_y[i, j])
                    if 0 <= new_x < width and 0 <= new_y < height:
                        result[i, j, 2] = img_array[new_y, new_x, 2]
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"Chromatic aberration error: {e}")
            return image
            
    def apply_vignette(self, image):
        """Apply vignette effect"""
        try:
            img_array = np.array(image, dtype=np.float32)
            height, width = img_array.shape[:2]
            
            # Create vignette mask
            center_x, center_y = width // 2, height // 2
            y, x = np.ogrid[:height, :width]
            
            # Calculate distance from center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Normalize distance
            max_distance = np.sqrt(center_x**2 + center_y**2)
            normalized_distance = distance / max_distance
            
            # Create vignette mask
            vignette_strength = self.vignette_strength.get()
            vignette = 1 - (normalized_distance * vignette_strength)
            vignette = np.clip(vignette, 0, 1)
            
            # Apply vignette
            if len(img_array.shape) == 3:
                vignette = np.stack([vignette] * 3, axis=-1)
            
            result = img_array * vignette
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
            
        except:
            return image
            
    def apply_film_grain(self, image):
        """Apply film grain effect (same values as v2, not scaled by image size)"""
        try:
            img_array = np.array(image, dtype=np.float32)
            height, width = img_array.shape[:2]
            
            # Generate noise - use same intensity as v2, not scaled by image size
            grain_intensity = self.grain_intensity.get() * 255
            noise = np.random.normal(0, grain_intensity, (height, width))
            
            # Apply edge boost (using fixed value from v2)
            edge_boost = self.grain_edge_boost.get()
            if edge_boost > 1:
                # Convert to grayscale for edge detection
                gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
                
                # Simple edge detection using gradients
                grad_x = np.abs(np.gradient(gray, axis=1))
                grad_y = np.abs(np.gradient(gray, axis=0))
                edges = np.sqrt(grad_x**2 + grad_y**2)
                
                # Normalize edges
                edges = edges / np.max(edges) if np.max(edges) > 0 else edges
                
                # Apply edge boost to noise
                noise = noise * (1 + edges * (edge_boost - 1))
            
            # Apply noise to all channels
            if len(img_array.shape) == 3:
                noise = np.stack([noise] * 3, axis=-1)
            
            result = img_array + noise
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
            
        except:
            return image
            
    def apply_saturation_reduction(self, image):
        """Apply saturation reduction"""
        try:
            # Convert to HSV
            hsv = image.convert('HSV')
            h, s, v = hsv.split()
            
            # Reduce saturation
            s_array = np.array(s, dtype=np.float32)
            reduction = self.saturation_reduction.get()
            s_array = s_array * (1 - reduction)
            
            # Recombine and convert back to RGB
            s_reduced = Image.fromarray(np.clip(s_array, 0, 255).astype(np.uint8))
            hsv_reduced = Image.merge('HSV', (h, s_reduced, v))
            return hsv_reduced.convert('RGB')
            
        except:
            return image
            
    def apply_unsharp_mask(self, image):
        """Apply unsharp mask for sharpening"""
        try:
            # Create blurred version
            blurred = image.filter(ImageFilter.GaussianBlur(radius=1.0))
            
            # Enhance using original - blurred
            return ImageChops.add(image, ImageChops.subtract(image, blurred))
            
        except:
            return image
            
    def apply_photo_border(self, image):
        """Apply subtle photo border effect"""
        try:
            img_array = np.array(image, dtype=np.float32)
            height, width = img_array.shape[:2]
            
            # Create border mask
            border_size = max(width, height) // 100  # 1% of largest dimension
            mask = np.ones((height, width))
            
            # Apply border fade
            for i in range(border_size):
                fade = i / border_size
                mask[i, :] *= fade  # top
                mask[-(i+1), :] *= fade  # bottom
                mask[:, i] *= fade  # left
                mask[:, -(i+1)] *= fade  # right
            
            # Apply mask
            if len(img_array.shape) == 3:
                mask = np.stack([mask] * 3, axis=-1)
            
            result = img_array * mask
            return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
            
        except:
            return image
            
    def update_preview(self):
        """Update preview with current effects"""
        if not self.preview_image:
            return
            
        try:
            # Apply effects to full image
            processed_full = self.apply_filmic_effects(self.preview_image)
            
            # Show CA center if checkbox is enabled
            if self.show_ca_center.get():
                from PIL import ImageDraw
                processed_with_center = processed_full.copy()
                draw = ImageDraw.Draw(processed_with_center)
                
                # Draw center point
                center_x, center_y = self.effect_center
                radius = max(8, max(self.current_image_width, self.current_image_height) // 200)
                
                # Draw outer circle (border)
                draw.ellipse([center_x-radius-1, center_y-radius-1, 
                             center_x+radius+1, center_y+radius+1], 
                           fill=None, outline=(0, 0, 0), width=2)
                # Draw inner circle (green)
                draw.ellipse([center_x-radius, center_y-radius, 
                             center_x+radius, center_y+radius], 
                           fill=(0, 255, 0), outline=None)
                
                processed_full = processed_with_center
            
            # Convert images to PhotoImage for display
            self.original_photo_ref = ImageTk.PhotoImage(self.preview_image)
            self.processed_photo_ref = ImageTk.PhotoImage(processed_full)
            
            # Clear canvases
            self.original_canvas.delete("all")
            self.processed_canvas.delete("all")
            
            # Display images
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo_ref)
            self.processed_canvas.create_image(0, 0, anchor=tk.NW, image=self.processed_photo_ref)
            
        except Exception as e:
            self.status_label.config(text=f"Preview error: {str(e)}")
            
    def process_all_images(self):
        """Process all images in the directory"""
        if not self.input_directory or not self.image_files:
            messagebox.showwarning("Warning", "Please select a directory with images first.")
            return
            
        def process_thread():
            try:
                # Create output directory
                output_dir = self.input_directory / "filmic_processed"
                output_dir.mkdir(exist_ok=True)
                
                total_images = len(self.image_files)
                processed_count = 0
                
                for i, image_path in enumerate(self.image_files):
                    try:
                        # Update status
                        self.root.after(0, lambda: self.status_label.config(
                            text=f"Processing {i+1}/{total_images}: {image_path.name}"))
                        
                        # Load image
                        image = Image.open(image_path)
                        
                        # Store current dimensions for this image
                        self.current_image_width = image.width
                        self.current_image_height = image.height
                        
                        # Detect face center for this image
                        if FACE_DETECTION_AVAILABLE:
                            self.detect_face_center_for_image(image)
                        else:
                            self.effect_center = (image.width // 2, image.height // 2)
                        
                        # Apply effects
                        processed_image = self.apply_filmic_effects(image)
                        
                        # Save processed image
                        output_path = output_dir / f"filmic_{image_path.name}"
                        processed_image.save(output_path, quality=95)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {image_path.name}: {e}")
                
                # Update final status
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Processing complete! {processed_count}/{total_images} images processed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Processing error: {str(e)}"))
        
        # Start processing in background thread
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
        
    def detect_face_center_for_image(self, image):
        """Detect face center for a specific image (used during batch processing)"""
        if not self.mp_face_detection:
            self.effect_center = (image.width // 2, image.height // 2)
            return
            
        try:
            # Convert PIL image to RGB numpy array
            image_rgb = np.array(image.convert('RGB'))
            
            # Process with MediaPipe
            results = self.mp_face_detection.process(image_rgb)
            
            if results.detections:
                # Use the first detected face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Calculate center point of the bounding box
                center_x = int((bbox.xmin + bbox.width / 2) * image.width)
                center_y = int((bbox.ymin + bbox.height / 2) * image.height)
                
                # Ensure center is within image bounds
                center_x = max(0, min(center_x, image.width - 1))
                center_y = max(0, min(center_y, image.height - 1))
                
                self.effect_center = (center_x, center_y)
            else:
                # No face detected, use image center
                self.effect_center = (image.width // 2, image.height // 2)
                
        except Exception as e:
            print(f"Face detection error: {e}")
            self.effect_center = (image.width // 2, image.height // 2)


def main():
    # Check if required packages are available
    missing_packages = []
    
    try:
        import numpy as np
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_packages.append("pillow")
    
    if missing_packages:
        error_msg = f"Missing required packages: {', '.join(missing_packages)}\\n\\n"
        error_msg += "Please install with:\\n"
        error_msg += f"pip install {' '.join(missing_packages)}\\n\\n"
        error_msg += "Or activate the virtual environment first."
        
        # Show error in GUI if possible, otherwise print
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Missing Dependencies", error_msg)
            root.destroy()
        except:
            print(error_msg)
        return
    
    root = tk.Tk()
    app = FilmicEffectsProcessor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
