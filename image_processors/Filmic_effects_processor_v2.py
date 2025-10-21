# Filmic Effects Processor v2 - Applies film grain and vignette effects to 720x1600 images

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
        self.root.title("Filmic Effects Processor v2 - 720x1600")
        self.root.geometry("1600x1400")
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
        
        # Effect enable/disable toggles
        self.grain_enabled = tk.BooleanVar(value=True)
        self.vignette_enabled = tk.BooleanVar(value=True)
        self.saturation_enabled = tk.BooleanVar(value=True)
        self.chromatic_enabled = tk.BooleanVar(value=True)
        
        # Chromatic aberration center display toggle
        self.show_ca_center = tk.BooleanVar(value=False)
        
        # Other options
        self.vintage_border = tk.BooleanVar(value=True)
        self.unsharp_sharpening = tk.BooleanVar(value=True)
        self.auto_contrast_stretch = tk.BooleanVar(value=True)
        
        # Internal parameters (not exposed in GUI)
        self.unsharp_radius = 1.0        # Gaussian blur radius for unsharp mask
        self.unsharp_amount = 1.3        # Sharpening strength multiplier
        self.unsharp_threshold = 3       # Minimum contrast threshold
        self.contrast_percentile = 0.5   # Percentile for auto-contrast (% on each end)
        
        # Face detection setup
        self.face_detection = None
        if FACE_DETECTION_AVAILABLE:
            try:
                # Initialize Face Detection (long-range model for smaller faces)
                mp_face_detection = mp.solutions.face_detection
                self.face_detection = mp_face_detection.FaceDetection(
                    model_selection=1,  # Long-range model for smaller faces
                    min_detection_confidence=0.3  # Lower confidence for better detection
                )
                print("Face detection initialized with long-range model for small faces")
            except Exception as e:
                print(f"Face detection initialization failed: {e}")
                self.face_detection = None
        
        # Default center point (fallback)
        self.effect_center = (360, 360)
        self.face_detected = False  # Track if face was detected for current image
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill='both', expand=True)
        
        # Configure style for mid-grey theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#808080')
        style.configure('Dark.TLabel', background='#808080', foreground='black')
        style.configure('Dark.TButton', background='#606060', foreground='white')
        
        # Configure slider style...
        style.configure('Blue.Horizontal.TScale', 
                       background="#2C7078",  
                       troughcolor='#0080FF',  
                       sliderthickness=20,     
                       sliderrelief='raised')
        style.map('Blue.Horizontal.TScale',
                 background=[('active', '#FFFFFF'), ('pressed', '#FFFFFF')],  # White slider button
                 troughcolor=[('active', '#0080FF'), ('pressed', '#0080FF')])  # Keep blue trough
        
        # Directory selection
        dir_frame = ttk.LabelFrame(main_frame, text="Image Directory", padding=10)
        dir_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(dir_frame, text="Select Directory", 
                  command=self.select_directory).pack(side=tk.LEFT, padx=(0, 10))
        
        self.dir_label = ttk.Label(dir_frame, text="No directory selected", 
                                  foreground="gray")
        self.dir_label.pack(side=tk.LEFT)
        
        # Face detection status indicator
        face_detection_text = "✓ Face Detection" if FACE_DETECTION_AVAILABLE else "✗ Face Detection"
        face_detection_color = "green" if FACE_DETECTION_AVAILABLE else "red"
        face_status_label = ttk.Label(dir_frame, text=face_detection_text, 
                                     foreground=face_detection_color, 
                                     font=("Arial", 8, "bold"))
        face_status_label.pack(side=tk.RIGHT, padx=(20, 10))
        
        # Status message
        self.progress_var = tk.StringVar(value="Select a directory to begin")
        status_label = ttk.Label(dir_frame, textvariable=self.progress_var, 
                               foreground="blue", font=("Arial", 9, "italic"))
        status_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Effect Controls", padding=15)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Create left and right sections for controls
        controls_container = ttk.Frame(control_frame)
        controls_container.pack(fill='x')
        
        # Left side - main sliders
        left_controls = ttk.Frame(controls_container)
        left_controls.pack(side=tk.LEFT, fill='x', expand=True)
        
        # Right side - additional options
        right_controls = ttk.Frame(controls_container)
        right_controls.pack(side=tk.RIGHT, padx=(20, 0))
        
        # Film grain controls
        grain_frame = ttk.Frame(left_controls)
        grain_frame.pack(fill='x', pady=5)
        
        grain_check = tk.Checkbutton(grain_frame, text="", variable=self.grain_enabled, 
                                    command=self.update_preview, bg='#808080', selectcolor='#606060')
        grain_check.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(grain_frame, text="Film Grain Intensity:", 
                 font=("Arial", 10, "bold"), width=19).pack(side=tk.LEFT)
        grain_scale = ttk.Scale(grain_frame, from_=0.0, to=0.56, 
                               variable=self.grain_intensity, orient='horizontal', 
                               length=500, style='Blue.Horizontal.TScale')
        grain_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.grain_label = ttk.Label(grain_frame, text="0.22", width=8,
                                    font=("Arial", 11, "bold"))
        self.grain_label.pack(side=tk.LEFT)
        
        ttk.Button(grain_frame, text="Reset", width=8,
                  command=lambda: self.reset_slider(self.grain_intensity, self.grain_label, self.DEFAULT_GRAIN)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Vignette controls
        vignette_frame = ttk.Frame(left_controls)
        vignette_frame.pack(fill='x', pady=5)
        
        vignette_check = tk.Checkbutton(vignette_frame, text="", variable=self.vignette_enabled, 
                                       command=self.update_preview, bg='#808080', selectcolor='#606060')
        vignette_check.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(vignette_frame, text="Vignette Strength:", 
                 font=("Arial", 10, "bold"), width=19).pack(side=tk.LEFT)
        vignette_scale = ttk.Scale(vignette_frame, from_=0.0, to=0.75, 
                                  variable=self.vignette_strength, orient='horizontal', 
                                  length=500, style='Blue.Horizontal.TScale')
        vignette_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.vignette_label = ttk.Label(vignette_frame, text="0.34", width=8,
                                       font=("Arial", 11, "bold"))
        self.vignette_label.pack(side=tk.LEFT)
        
        ttk.Button(vignette_frame, text="Reset", width=8,
                  command=lambda: self.reset_slider(self.vignette_strength, self.vignette_label, self.DEFAULT_VIGNETTE)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Saturation controls
        saturation_frame = ttk.Frame(left_controls)
        saturation_frame.pack(fill='x', pady=5)
        
        saturation_check = tk.Checkbutton(saturation_frame, text="", variable=self.saturation_enabled, 
                                         command=self.update_preview, bg='#808080', selectcolor='#606060')
        saturation_check.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(saturation_frame, text="Saturation Reduction:", 
                 font=("Arial", 10, "bold"), width=19).pack(side=tk.LEFT)
        saturation_scale = ttk.Scale(saturation_frame, from_=0.0, to=1.0, 
                                    variable=self.saturation_reduction, orient='horizontal', 
                                    length=500, style='Blue.Horizontal.TScale')
        saturation_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.saturation_label = ttk.Label(saturation_frame, text="0.30", width=8,
                                         font=("Arial", 11, "bold"))
        self.saturation_label.pack(side=tk.LEFT)
        
        ttk.Button(saturation_frame, text="Reset", width=8,
                  command=lambda: self.reset_slider(self.saturation_reduction, self.saturation_label, self.DEFAULT_SATURATION)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Chromatic aberration controls
        aberration_frame = ttk.Frame(left_controls)
        aberration_frame.pack(fill='x', pady=5)
        
        chromatic_check = tk.Checkbutton(aberration_frame, text="", variable=self.chromatic_enabled, 
                                        command=self.update_preview, bg='#808080', selectcolor='#606060')
        chromatic_check.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(aberration_frame, text="Chromatic Aberration:", 
                 font=("Arial", 10, "bold"), width=19).pack(side=tk.LEFT)
        aberration_scale = ttk.Scale(aberration_frame, from_=0.0, to=1.0, 
                                    variable=self.chromatic_aberration, orient='horizontal', 
                                    length=500, style='Blue.Horizontal.TScale')
        aberration_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.aberration_label = ttk.Label(aberration_frame, text="0.09", width=8,
                                         font=("Arial", 11, "bold"))
        self.aberration_label.pack(side=tk.LEFT)
        
        ttk.Button(aberration_frame, text="Reset", width=8,
                  command=lambda: self.reset_slider(self.chromatic_aberration, self.aberration_label, self.DEFAULT_CHROMATIC)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Show face outline checkbox
        ca_center_check = tk.Checkbutton(aberration_frame, text="Show Face centre", 
                                        variable=self.show_ca_center, command=self.update_preview,
                                        bg='#808080', fg='black', selectcolor='#606060',
                                        activebackground='#808080', activeforeground='black',
                                        font=("Arial", 10, "bold"))
        ca_center_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # Right side controls - Additional options
        ttk.Label(right_controls, text="Additional Options", 
                 font=("Arial", 11, "bold")).pack(pady=(0, 10))
        
        # Vintage border checkbox
        vintage_border_check = tk.Checkbutton(right_controls, text="Photo Border", 
                                             variable=self.vintage_border, command=self.update_preview,
                                             bg='#808080', fg='black', selectcolor='#606060',
                                             activebackground='#808080', activeforeground='black',
                                             font=("Arial", 9))
        vintage_border_check.pack(anchor='w', pady=2)
        
        # Unsharp sharpening checkbox
        unsharp_check = tk.Checkbutton(right_controls, text="Unsharp Sharpening", 
                                      variable=self.unsharp_sharpening, command=self.update_preview,
                                      bg='#808080', fg='black', selectcolor='#606060',
                                      activebackground='#808080', activeforeground='black',
                                      font=("Arial", 9))
        unsharp_check.pack(anchor='w', pady=2)
        
        # Auto-contrast stretch checkbox
        contrast_check = tk.Checkbutton(right_controls, text="Auto-Contrast Stretch", 
                                       variable=self.auto_contrast_stretch, command=self.update_preview,
                                       bg='#808080', fg='black', selectcolor='#606060',
                                       activebackground='#808080', activeforeground='black',
                                       font=("Arial", 9))
        contrast_check.pack(anchor='w', pady=2)
        
        # Restore defaults button
        ttk.Button(right_controls, text="Restore All Defaults", 
                  command=self.restore_all_defaults).pack(pady=(15, 0))
        
        # Update labels when scales change
        grain_scale.configure(command=self.update_grain_label)
        vignette_scale.configure(command=self.update_vignette_label)
        saturation_scale.configure(command=self.update_saturation_label)
        aberration_scale.configure(command=self.update_aberration_label)
        
        # Preview and processing controls
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(action_frame, text="Load Preview", 
                  command=self.load_preview).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="◀ Previous", 
                  command=self.previous_preview).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="Next ▶", 
                  command=self.next_preview).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="Process All Images", 
                  command=self.process_all_images, 
                  style='Accent.TButton').pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Preview area with side-by-side comparison and synchronized scrolling
        preview_frame = ttk.LabelFrame(main_frame, text="Comparison (1:1 Scale)", padding=10)
        preview_frame.pack(fill='both', expand=True)
        
        # Container for both preview canvases
        canvas_container = tk.Frame(preview_frame, bg='#808080')
        canvas_container.pack(expand=True, fill='both')
        
        # Original image canvas (left side) with scrollbar
        original_frame = tk.Frame(canvas_container, bg='#808080')
        original_frame.pack(side=tk.LEFT, padx=10)
        
        original_label = tk.Label(original_frame, text="Original", 
                                 bg='#808080', fg='black', font=("Arial", 8))
        original_label.pack(pady=2)
        
        # Create frame for canvas and scrollbar
        original_canvas_frame = tk.Frame(original_frame, bg='#808080')
        original_canvas_frame.pack()
        
        self.original_canvas = tk.Canvas(original_canvas_frame, width=720, height=800, 
                                        bg='#1a1a1a', highlightthickness=1, 
                                        highlightbackground='#555555')
        self.original_canvas.pack(side=tk.LEFT)
        
        # Bind mouse wheel scrolling to original canvas
        self.original_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.original_canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.original_canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        
        # Vertical scrollbar for original canvas
        original_scrollbar = tk.Scrollbar(original_canvas_frame, orient='vertical', 
                                         command=self.sync_scroll)
        original_scrollbar.pack(side=tk.RIGHT, fill='y')
        self.original_canvas.configure(yscrollcommand=original_scrollbar.set)
        
        # Processed image canvas (right side) with scrollbar
        processed_frame = tk.Frame(canvas_container, bg='#808080')
        processed_frame.pack(side=tk.LEFT, padx=10)
        
        # Processed label with face detection status
        self.processed_label_text = tk.StringVar(value="Processed")
        processed_label = tk.Label(processed_frame, textvariable=self.processed_label_text, 
                                  bg='#808080', fg='black', font=("Arial", 8))
        processed_label.pack(pady=2)
        
        # Create frame for canvas and scrollbar
        processed_canvas_frame = tk.Frame(processed_frame, bg='#808080')
        processed_canvas_frame.pack()
        
        self.processed_canvas = tk.Canvas(processed_canvas_frame, width=720, height=800, 
                                         bg='#1a1a1a', highlightthickness=1, 
                                         highlightbackground='#555555')
        self.processed_canvas.pack(side=tk.LEFT)
        
        # Bind mouse wheel scrolling to processed canvas
        self.processed_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.processed_canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.processed_canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        
        # Vertical scrollbar for processed canvas
        processed_scrollbar = tk.Scrollbar(processed_canvas_frame, orient='vertical', 
                                          command=self.sync_scroll)
        processed_scrollbar.pack(side=tk.RIGHT, fill='y')
        self.processed_canvas.configure(yscrollcommand=processed_scrollbar.set)
        
        # Store scrollbar references for synchronization
        self.original_scrollbar = original_scrollbar
        self.processed_scrollbar = processed_scrollbar
        
    def update_grain_label(self, value):
        """Update grain intensity label"""
        self.grain_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
    def update_vignette_label(self, value):
        """Update vignette strength label"""
        self.vignette_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
    def update_saturation_label(self, value):
        """Update saturation reduction label"""
        self.saturation_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
    def update_aberration_label(self, value):
        """Update chromatic aberration label"""
        self.aberration_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
    def reset_slider(self, variable, label, default_value):
        """Reset a specific slider to its default value"""
        variable.set(default_value)
        label.config(text=f"{default_value:.2f}")
        self.update_preview()
        
    def restore_all_defaults(self):
        """Restore all slider values to their defaults"""
        self.grain_intensity.set(self.DEFAULT_GRAIN)
        self.vignette_strength.set(self.DEFAULT_VIGNETTE)
        self.saturation_reduction.set(self.DEFAULT_SATURATION)
        self.chromatic_aberration.set(self.DEFAULT_CHROMATIC)
        
        # Update labels
        self.grain_label.config(text=f"{self.DEFAULT_GRAIN:.2f}")
        self.vignette_label.config(text=f"{self.DEFAULT_VIGNETTE:.2f}")
        self.saturation_label.config(text=f"{self.DEFAULT_SATURATION:.2f}")
        self.aberration_label.config(text=f"{self.DEFAULT_CHROMATIC:.2f}")
        
        # Update preview
        self.update_preview()
        
    def detect_face_center(self, image):
        """Detect face center using MediaPipe Face Detection optimized for small faces"""
        if not FACE_DETECTION_AVAILABLE or self.face_detection is None:
            self.face_detected = False
            return (360, 360)  # Default center
            
        try:
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image.copy()
            
            # Ensure RGB format
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                rgb_image = image_np
            else:
                if FACE_DETECTION_AVAILABLE:
                    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                else:
                    # Fallback for grayscale without cv2
                    rgb_image = np.stack([image_np, image_np, image_np], axis=-1)
            
            h, w = rgb_image.shape[:2]
            
            # Use Face Detection (optimized for small faces with long-range model)
            detection_results = self.face_detection.process(rgb_image)
            
            if detection_results.detections and len(detection_results.detections) > 0:
                # Get the first (most confident) detection
                detection = detection_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Calculate face center from bounding box
                center_x = int((bbox.xmin + bbox.width / 2) * w)
                center_y = int((bbox.ymin + bbox.height / 2) * h)
                
                # Calculate face size for reporting
                face_width = int(bbox.width * w)
                face_height = int(bbox.height * h)
                
                print(f"Face detected! Size: {face_width}x{face_height}px, Center: ({center_x}, {center_y})")
                self.face_detected = True
                return (center_x, center_y)
            else:
                print("No face detected with long-range Face Detection model")
                
        except Exception as e:
            print(f"Face detection error: {e}")
            
        # Return default center if face detection fails
        self.face_detected = False
        return (360, 360)
        
    def select_directory(self):
        """Select directory containing images"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.input_directory = directory
            self.scan_images()
            
    def scan_images(self):
        """Scan directory for valid image files"""
        if not self.input_directory:
            return
            
        # Supported image formats
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        self.image_files = []
        for file_path in Path(self.input_directory).iterdir():
            if file_path.suffix.lower() in extensions:
                self.image_files.append(file_path)
        
        self.image_files.sort()  # Sort alphabetically
        
        if self.image_files:
            self.dir_label.config(text=f"{len(self.image_files)} images found in {os.path.basename(self.input_directory)}")
            self.progress_var.set(f"Found {len(self.image_files)} images - Load preview to start")
            self.current_preview_index = 0
        else:
            self.dir_label.config(text="No valid images found")
            self.progress_var.set("No valid images found in directory")
            
    def load_preview(self):
        """Load and display preview image"""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please select a directory with images first")
            return
            
        try:
            image_path = self.image_files[self.current_preview_index]
            self.preview_image = Image.open(image_path)
            
            # Verify image dimensions
            if self.preview_image.size != (720, 1600):
                messagebox.showwarning("Wrong Dimensions", 
                                     f"Image {image_path.name} is {self.preview_image.size}, expected 720x1600")
                return
                
            self.progress_var.set(f"Preview: {image_path.name} ({self.current_preview_index + 1}/{len(self.image_files)})")
            
            # Detect face center for dynamic effect positioning
            self.effect_center = self.detect_face_center(self.preview_image)
            
            # Update processed label with face detection status
            face_status = "✓ Face detected" if self.face_detected else "✗ No face"
            self.processed_label_text.set(f"Processed ({face_status})")
            
            # Set flag to indicate this is a new image load (should reset scroll to top)
            self._is_new_image_load = True
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preview image: {str(e)}")
            
    def previous_preview(self):
        """Load previous image for preview"""
        if self.image_files and self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.load_preview()
            
    def next_preview(self):
        """Load next image for preview"""
        if self.image_files and self.current_preview_index < len(self.image_files) - 1:
            self.current_preview_index += 1
            self.load_preview()
            
    def apply_film_grain_rgb(self, image, intensity):
        # Apply film grain to RGB image as final step
        if intensity == 0.0:
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array
        rgb_array = np.array(image, dtype=np.float32)
        width, height = image.size
        
        # Create uniform grain noise for all RGB channels
        grain = np.random.normal(0, intensity, (height, width, 3))
        
        # Scale grain for RGB (larger range than HSV)
        grain *= 25.0  # Scale for 0-255 RGB range
        
        # Apply grain to RGB channels
        rgb_array += grain
        
        # Clamp to valid RGB range
        rgb_array = np.clip(rgb_array, 0, 255)
        
        # Convert back to PIL Image
        return Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')
        
    def apply_saturation_reduction_rgb(self, image, reduction):
        # Apply saturation reduction to RGB image as final processing step
        if reduction == 0.0:
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to HSV for saturation adjustment
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image, dtype=np.float32)
        
        # Apply saturation reduction
        saturation_factor = 1.0 - reduction
        hsv_array[:, :, 1] *= saturation_factor
        
        # Clamp saturation values
        hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1], 0, 255)
        
        # Convert back to RGB
        hsv_result = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        return hsv_result.convert('RGB')
        
    def create_vignette(self, width, height, strength):
        """Create circular vignette effect"""
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Create circular vignette (normalize distance)
        vignette = distance / max_distance
        
        # Apply smooth falloff
        vignette = np.clip(vignette, 0, 1)
        vignette = 1 - (vignette ** 2 * strength)
        
        return vignette
        
    def apply_chromatic_aberration(self, image, strength):
        """Apply chromatic aberration effect by shifting red/blue channels"""
        if strength == 0.0:
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        center_x, center_y = self.effect_center  # Use detected face center or default
        
        # Split into RGB channels
        r, g, b = image.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)
        
        # Create distance map from center
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create aberration factor based on distance
        # No effect within radius 250, full effect at radius 800+
        aberration_factor = np.zeros_like(distance)
        mask = distance > 250
        aberration_factor[mask] = np.minimum((distance[mask] - 250) / (800 - 250), 1.0)

        # Calculate shift amount (30 pixels at full strength)
        shift_amount = aberration_factor * strength * 30.0
        
        # Create shifted versions of red and blue channels
        r_shifted = r_array.copy()
        b_shifted = b_array.copy()
        
        # Apply horizontal shifts using numpy roll for efficiency
        max_shift = int(np.ceil(np.max(shift_amount)))
        if max_shift > 0:
            # Calculate integer shift for each pixel
            x_shift = np.round(shift_amount).astype(int)
            
            # Create shifted versions by processing different shift amounts
            for shift_val in range(1, max_shift + 1):
                # Find pixels that need this shift amount
                shift_mask = (x_shift == shift_val)
                
                if np.any(shift_mask):
                    # For red channel (shift right)
                    red_rolled = np.roll(r_array, shift_val, axis=1)
                    # Zero out the left edge that wrapped around
                    red_rolled[:, :shift_val] = r_array[:, :shift_val]
                    # Apply only where mask is true
                    r_shifted = np.where(shift_mask, red_rolled, r_shifted)
                    
                    # For blue channel (shift left)  
                    blue_rolled = np.roll(b_array, -shift_val, axis=1)
                    # Zero out the right edge that wrapped around
                    blue_rolled[:, -shift_val:] = b_array[:, -shift_val:]
                    # Apply only where mask is true
                    b_shifted = np.where(shift_mask, blue_rolled, b_shifted)
        
        # Recombine channels
        result = Image.merge('RGB', (
            Image.fromarray(r_shifted.astype(np.uint8)),
            Image.fromarray(g_array.astype(np.uint8)),
            Image.fromarray(b_shifted.astype(np.uint8))
        ))
        
        return result
        
    def apply_filmic_effects(self, image):
        """Apply film grain, vignette, and saturation effects to image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to HSV for better grain control
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image, dtype=np.float32)
        width, height = image.size
        
        # Normalize HSV values (H: 0-360, S: 0-100, V: 0-100)
        hsv_array[:, :, 0] /= 255.0  # H: 0-1 (represents 0-360°)
        hsv_array[:, :, 1] /= 255.0  # S: 0-1 (represents 0-100%)
        hsv_array[:, :, 2] /= 255.0  # V: 0-1 (represents 0-100%)
        
        # Create and apply vignette to brightness only if enabled
        if self.vignette_enabled.get():
            vignette = self.create_vignette(width, height, self.vignette_strength.get())
            hsv_array[:, :, 2] *= vignette
        
        # Clamp HSV values to valid ranges
        hsv_array[:, :, 0] = np.clip(hsv_array[:, :, 0], 0, 1)     # Hue: 0-1
        hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1], 0, 1)     # Saturation: 0-1
        hsv_array[:, :, 2] = np.clip(hsv_array[:, :, 2], 0, 1)     # Value: 0-1
        
        # Convert back to 0-255 range
        hsv_array[:, :, 0] *= 255.0
        hsv_array[:, :, 1] *= 255.0
        hsv_array[:, :, 2] *= 255.0
        
        # Convert back to RGB
        hsv_result = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        rgb_result = hsv_result.convert('RGB')
        
        # Apply effects only if enabled
        current_result = rgb_result
        
        # Apply chromatic aberration if enabled
        if self.chromatic_enabled.get():
            current_result = self.apply_chromatic_aberration(current_result, self.chromatic_aberration.get())
        
        # Apply film grain if enabled
        if self.grain_enabled.get():
            current_result = self.apply_film_grain_rgb(current_result, self.grain_intensity.get())
        
        # Apply saturation reduction if enabled
        if self.saturation_enabled.get():
            current_result = self.apply_saturation_reduction_rgb(current_result, self.saturation_reduction.get())
        
        # Apply unsharp sharpening if enabled
        if self.unsharp_sharpening.get():
            current_result = self.apply_unsharp_sharpening(current_result)
        
        # Apply auto-contrast stretch if enabled
        if self.auto_contrast_stretch.get():
            current_result = self.apply_auto_contrast_stretch(current_result)
        
        # Apply vintage border as the final step
        if self.vintage_border.get():
            current_result = self.create_vintage_border(current_result)
        
        return current_result
        
    def update_preview(self, event=None):
        """Update preview with current effects showing full 1600px images at 1:1 scale with scrolling"""
        if not self.preview_image:
            return
            
        try:
            # Save current scroll position before updating
            current_scroll_pos = self.original_canvas.yview()
            
            # Use full image (720x1600)
            original_full = self.preview_image.copy()
            
            # Apply effects to full image
            processed_full = self.apply_filmic_effects(self.preview_image)
            
            # Show face outline on processed image only if checkbox is enabled
            if self.show_ca_center.get():
                from PIL import ImageDraw
                processed_with_face = processed_full.copy()
                draw = ImageDraw.Draw(processed_with_face)
                
                
                # Draw center point when CA center is enabled
                center_x, center_y = self.effect_center
                radius = 8
                # Draw outer circle (border)
                draw.ellipse([center_x-radius-1, center_y-radius-1, center_x+radius+1, center_y+radius+1], 
                           fill=None, outline=(0, 0, 0), width=2)
                # Draw inner circle (green)
                draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                           fill=(0, 255, 0), outline=None)
                
                processed_full = processed_with_face
            
            self.processed_preview = processed_full
            
            # Convert to PhotoImage for display at 1:1 scale
            original_photo = ImageTk.PhotoImage(original_full)
            processed_photo = ImageTk.PhotoImage(processed_full)
            
            # Clear canvases and display images
            self.original_canvas.delete("all")
            self.processed_canvas.delete("all")
            
            # Display full images at top-left (0,0) for proper scrolling
            self.original_canvas.create_image(0, 0, image=original_photo, anchor='nw')
            self.processed_canvas.create_image(0, 0, image=processed_photo, anchor='nw')
            
            # Configure scroll regions for full image height (1600px)
            self.original_canvas.configure(scrollregion=(0, 0, 720, 1600))
            self.processed_canvas.configure(scrollregion=(0, 0, 720, 1600))
            
            # Keep references to prevent garbage collection
            self.original_photo_ref = original_photo
            self.processed_photo_ref = processed_photo
            
            # Restore previous scroll position (only reset to top if this is a new image load)
            if hasattr(self, '_is_new_image_load') and self._is_new_image_load:
                # Reset to top for new image
                self.original_canvas.yview_moveto(0)
                self.processed_canvas.yview_moveto(0)
                self._is_new_image_load = False
            else:
                # Preserve scroll position for slider updates
                if len(current_scroll_pos) >= 2:
                    self.original_canvas.yview_moveto(current_scroll_pos[0])
                    self.processed_canvas.yview_moveto(current_scroll_pos[0])
            
        except Exception as e:
            print(f"Preview error: {e}")
            
    def process_all_images(self):
        """Process all images in directory"""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please select a directory with images first")
            return
            
        # Confirm processing
        result = messagebox.askyesno("Confirm Processing", 
                                   f"Process {len(self.image_files)} images with filmic effects?\n\n"
                                   f"Images will be saved in 'filmic' subdirectory with a '_filmic' suffix.")
        if not result:
            return
            
        # Start processing in separate thread
        thread = threading.Thread(target=self._process_images_thread)
        thread.daemon = True
        thread.start()
        
    def _process_images_thread(self):
        """Process images in separate thread"""
        try:
            total_images = len(self.image_files)
            processed_count = 0
            
            for i, image_path in enumerate(self.image_files):
                # Update progress
                self.root.after(0, lambda: self.progress_var.set(
                    f"Processing {image_path.name}... ({i+1}/{total_images})"))
                self.root.after(0, lambda: self.progress_bar.configure(
                    maximum=total_images, value=i))
                
                try:
                    # Load image
                    image = Image.open(image_path)
                    
                    # Check dimensions
                    if image.size != (720, 1600):
                        print(f"Skipping {image_path.name} - wrong dimensions: {image.size}")
                        continue
                    
                    # Detect face center for this specific image
                    self.effect_center = self.detect_face_center(image)
                    
                    # Apply effects
                    processed = self.apply_filmic_effects(image)
                    
                    # Create filmic subdirectory if it doesn't exist
                    filmic_dir = image_path.parent / "filmic"
                    filmic_dir.mkdir(exist_ok=True)
                    
                    # Create output filename in filmic subdirectory
                    output_path = filmic_dir / f"{image_path.stem}_filmic{image_path.suffix}"
                    
                    # Save processed image
                    processed.save(output_path, quality=95 if image_path.suffix.lower() in ['.jpg', '.jpeg'] else None)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {image_path.name}: {e}")
                    continue
            
            # Update UI on completion
            self.root.after(0, lambda: self.progress_var.set(
                f"Completed! Processed {processed_count}/{total_images} images"))
            self.root.after(0, lambda: self.progress_bar.configure(value=total_images))
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo("Processing Complete", 
                                                          f"Successfully processed {processed_count} images!\n\n"
                                                          f"Output files saved in 'filmic' subdirectory with '_filmic' suffix."))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", 
                                                           f"An error occurred during processing: {str(e)}"))
    
    def sync_scroll(self, *args):
        """Synchronize scrolling between both preview canvases"""
        # Get the scroll position from whichever scrollbar was moved
        if args[0] == 'scroll':
            # Apply the same scroll to both canvases
            self.original_canvas.yview(*args)
            self.processed_canvas.yview(*args)
        elif args[0] == 'moveto':
            # Move both canvases to the same position
            self.original_canvas.yview_moveto(args[1])
            self.processed_canvas.yview_moveto(args[1])
        
        # Update both scrollbars to show the same position
        pos = self.original_canvas.yview()
        self.original_scrollbar.set(*pos)
        self.processed_scrollbar.set(*pos)
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling for synchronized canvas scrolling"""
        # Calculate scroll amount based on wheel delta
        # On Windows, event.delta is typically 120 or -120
        # On Linux, event.num is 4 (up) or 5 (down)
        
        if event.num == 4 or event.delta > 0:
            # Scroll up
            delta = -1
        elif event.num == 5 or event.delta < 0:
            # Scroll down
            delta = 1
        else:
            return
        
        # Get current scroll position
        current_top, current_bottom = self.original_canvas.yview()
        
        # Calculate new scroll position (scroll 3 units at a time for smooth scrolling)
        scroll_units = 3
        self.original_canvas.yview_scroll(delta * scroll_units, "units")
        self.processed_canvas.yview_scroll(delta * scroll_units, "units")
        
        # Update both scrollbars to show the same position
        pos = self.original_canvas.yview()
        self.original_scrollbar.set(*pos)
        self.processed_scrollbar.set(*pos)
        
        return "break"  # Prevent event from propagating

    def create_vintage_border(self, image):
        """Create a white border with rounded corners to simulate old photograph edges"""
        if not self.vintage_border.get():
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        border_width = 12
        corner_radius = 6  # Half of border width
        
        # Create a copy of the image to work with
        bordered_image = image.copy()
        
        # Convert to numpy array for easier manipulation
        img_array = np.array(bordered_image)
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Start with no border (all False)
        border_mask = np.zeros((height, width), dtype=bool)
        
        # Create the border area (outer rectangle minus inner rectangle with rounded corners)
        # Outer rectangle is the full image (0,0) to (width, height)
        # Inner rectangle with rounded corners
        
        inner_left = border_width
        inner_right = width - border_width
        inner_top = border_width
        inner_bottom = height - border_width
        
        # Create mask for the central rectangular area (will be carved out with rounded corners)
        central_rect = (x >= inner_left) & (x < inner_right) & (y >= inner_top) & (y < inner_bottom)
        
        # Create rounded corner cutouts from the central rectangle
        # Top-left corner cutout
        tl_corner_x, tl_corner_y = inner_left, inner_top
        tl_distance = np.sqrt((x - tl_corner_x)**2 + (y - tl_corner_y)**2)
        tl_cutout = (x >= inner_left) & (x < inner_left + corner_radius) & \
                   (y >= inner_top) & (y < inner_top + corner_radius) & \
                   (tl_distance < corner_radius)
        
        # Top-right corner cutout
        tr_corner_x, tr_corner_y = inner_right, inner_top
        tr_distance = np.sqrt((x - tr_corner_x)**2 + (y - tr_corner_y)**2)
        tr_cutout = (x >= inner_right - corner_radius) & (x < inner_right) & \
                   (y >= inner_top) & (y < inner_top + corner_radius) & \
                   (tr_distance < corner_radius)
        
        # Bottom-left corner cutout
        bl_corner_x, bl_corner_y = inner_left, inner_bottom
        bl_distance = np.sqrt((x - bl_corner_x)**2 + (y - bl_corner_y)**2)
        bl_cutout = (x >= inner_left) & (x < inner_left + corner_radius) & \
                   (y >= inner_bottom - corner_radius) & (y < inner_bottom) & \
                   (bl_distance < corner_radius)
        
        # Bottom-right corner cutout
        br_corner_x, br_corner_y = inner_right, inner_bottom
        br_distance = np.sqrt((x - br_corner_x)**2 + (y - br_corner_y)**2)
        br_cutout = (x >= inner_right - corner_radius) & (x < inner_right) & \
                   (y >= inner_bottom - corner_radius) & (y < inner_bottom) & \
                   (br_distance < corner_radius)
        
        # Create the final inner area with rounded corners
        inner_area_with_rounded_corners = central_rect & ~(tl_cutout | tr_cutout | bl_cutout | br_cutout)
        
        # Border is everything that's NOT the inner area
        border_mask = ~inner_area_with_rounded_corners
        
        # Apply warm very light grey border where mask is True
        img_array[border_mask] = [230, 225, 215]  # Warm very light grey (#EBE1D2)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
    def apply_unsharp_sharpening(self, image):
        """Apply unsharp mask sharpening to the image using internal parameters."""
        try:
            from PIL import ImageFilter
            
            # Use internal parameters (not exposed in GUI)
            radius = self.unsharp_radius  # 2.0
            amount = self.unsharp_amount  # 1.5 (150%)
            threshold = self.unsharp_threshold  # 3
            
            # Apply unsharp mask filter
            sharpened = image.filter(ImageFilter.UnsharpMask(
                radius=int(radius),  # Convert to integer
                percent=int(amount * 100),  # Convert to percentage
                threshold=int(threshold)  # Convert to integer
            ))
            
            return sharpened
            
        except Exception as e:
            print(f"Error applying unsharp sharpening: {e}")
            return image
    
    def apply_auto_contrast_stretch(self, image):
        """Apply automatic contrast stretching using percentile-based approach."""
        try:
            import numpy as np
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Use internal parameter (not exposed in GUI)
            percentile = self.contrast_percentile  # 2.0
            
            # Calculate percentiles for each channel
            low_percentiles = np.percentile(img_array, percentile, axis=(0, 1))
            high_percentiles = np.percentile(img_array, 100 - percentile, axis=(0, 1))
            
            # Stretch contrast for each channel
            stretched_array = np.zeros_like(img_array, dtype=np.float32)
            
            for i in range(img_array.shape[2]):  # For each color channel
                channel = img_array[:, :, i].astype(np.float32)
                low_val = float(low_percentiles[i])
                high_val = float(high_percentiles[i])
                
                # Avoid division by zero
                if high_val > low_val:
                    # Stretch to full 0-255 range
                    stretched_channel = (channel - low_val) * 255.0 / (high_val - low_val)
                    stretched_channel = np.clip(stretched_channel, 0, 255)
                else:
                    stretched_channel = channel
                
                stretched_array[:, :, i] = stretched_channel
            
            # Convert back to uint8 and PIL Image
            stretched_array = stretched_array.astype(np.uint8)
            return Image.fromarray(stretched_array)
            
        except Exception as e:
            print(f"Error applying auto contrast stretch: {e}")
            return image
        
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
        error_msg = f"Missing required packages: {', '.join(missing_packages)}\n\n"
        error_msg += "Please install with:\n"
        error_msg += f"pip install {' '.join(missing_packages)}\n\n"
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
