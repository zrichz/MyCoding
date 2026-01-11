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
        
        # Effect parameters (existing)
        self.grain_intensity = tk.DoubleVar(value=self.DEFAULT_GRAIN)
        self.grain_edge_boost = tk.DoubleVar(value=1.8)
        self.vignette_strength = tk.DoubleVar(value=self.DEFAULT_VIGNETTE)
        self.saturation_reduction = tk.DoubleVar(value=self.DEFAULT_SATURATION)
        self.chromatic_aberration = tk.DoubleVar(value=self.DEFAULT_CHROMATIC)
        
        # Effect enable/disable toggles (existing)
        self.grain_enabled = tk.BooleanVar(value=True)
        self.vignette_enabled = tk.BooleanVar(value=True)
        self.saturation_enabled = tk.BooleanVar(value=True)
        self.chromatic_enabled = tk.BooleanVar(value=True)
        
        # Chromatic aberration center display toggle
        self.show_ca_center = tk.BooleanVar(value=False)
        
        # Other options (existing)
        self.vintage_border = tk.BooleanVar(value=True)
        self.unsharp_sharpening = tk.BooleanVar(value=True)
        self.unsharp_half_strength = tk.BooleanVar(value=False)
        self.auto_contrast_stretch = tk.BooleanVar(value=True)
        
        # NEW: Photographic effects toggles
        self.apply_tone_curve = tk.BooleanVar(value=False)
        self.tone_strength = tk.DoubleVar(value=0.5)
        
        self.apply_split_tone = tk.BooleanVar(value=False)
        
        self.apply_photo_grain = tk.BooleanVar(value=False)
        self.photo_grain_strength = tk.DoubleVar(value=0.04)
        
        self.apply_photo_chromatic = tk.BooleanVar(value=False)
        self.ca_shift = tk.IntVar(value=1)
        
        self.apply_halation = tk.BooleanVar(value=False)
        self.halation_strength = tk.DoubleVar(value=0.2)
        
        self.apply_photo_vignette = tk.BooleanVar(value=False)
        self.photo_vignette_strength = tk.DoubleVar(value=0.5)
        
        self.apply_scan_banding = tk.BooleanVar(value=False)
        self.banding_strength = tk.DoubleVar(value=0.01)
        
        self.apply_rgb_misalignment = tk.BooleanVar(value=False)
        self.misalign_px = tk.IntVar(value=1)
        
        self.apply_dust = tk.BooleanVar(value=False)
        self.dust_amount = tk.DoubleVar(value=0.001)
        self.scratch_amount = tk.DoubleVar(value=0.0002)
        
        self.apply_jpeg_artifacts = tk.BooleanVar(value=False)
        self.jpeg_quality = tk.IntVar(value=70)
        
        self.apply_lens_distortion = tk.BooleanVar(value=False)
        self.distortion_strength = tk.DoubleVar(value=-0.0005)
        
        # Dithering options
        self.apply_dithering = tk.BooleanVar(value=False)
        self.dithering_type = tk.StringVar(value="floyd-steinberg")  # or "bayer"
        self.dithering_colors = tk.IntVar(value=256)  # colors per channel
        
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
    
    def create_large_checkbox(self, parent, text, variable, command=None, font_size=13, bold=False):
        """Helper to create larger, more visible checkboxes"""
        font_style = ("Arial", font_size, "bold") if bold else ("Arial", font_size)
        cb = tk.Checkbutton(parent, text=text, variable=variable, 
                          command=command if command else self.update_preview,
                          bg='#EBE1D2', selectcolor='#707070', font=font_style,
                          borderwidth=2, relief='flat', padx=6, pady=2,
                          cursor='hand2')
        return cb
        
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
        
        # Configure entry style for input boxes
        style.configure('TEntry', fieldbackground='white', foreground='black')
        
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
        
        # Control panel with tabbed interface
        control_frame = ttk.LabelFrame(main_frame, text="Effect Controls", padding=10)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Create notebook (tabs) for organized controls
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill='both', expand=True)
        
        # Tab 1: Original Effects (TWO COLUMNS)
        original_tab = ttk.Frame(notebook, padding=10)
        notebook.add(original_tab, text="Original Effects")
        
        # Create four column frames
        col1 = ttk.Frame(original_tab)
        col1.grid(row=0, column=0, padx=10, sticky='n')
        col2 = ttk.Frame(original_tab)
        col2.grid(row=0, column=1, padx=10, sticky='n')
        col3 = ttk.Frame(original_tab)
        col3.grid(row=0, column=2, padx=10, sticky='n')
        col4 = ttk.Frame(original_tab)
        col4.grid(row=0, column=3, padx=10, sticky='n')
        
        # COLUMN 1
        row = 0
        # Film grain
        self.create_large_checkbox(col1, "Film Grain", self.grain_enabled, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col1, text="Intensity:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        self.grain_entry = ttk.Entry(col1, textvariable=self.grain_intensity, width=10, font=("Arial", 11))
        self.grain_entry.grid(row=row, column=2, padx=5)
        self.grain_entry.bind('<Return>', lambda e: self.update_preview())
        self.grain_entry.bind('<FocusOut>', lambda e: self.update_preview())
        ttk.Button(col1, text="Reset", width=8,
                  command=lambda: self.reset_value(self.grain_intensity, self.DEFAULT_GRAIN)).grid(row=row, column=3, padx=5)
        row += 1
        
        # COLUMN 2
        row = 0
        # Vignette
        self.create_large_checkbox(col2, "Vignette", self.vignette_enabled, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col2, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        self.vignette_entry = ttk.Entry(col2, textvariable=self.vignette_strength, width=10, font=("Arial", 11))
        self.vignette_entry.grid(row=row, column=2, padx=5)
        self.vignette_entry.bind('<Return>', lambda e: self.update_preview())
        self.vignette_entry.bind('<FocusOut>', lambda e: self.update_preview())
        ttk.Button(col2, text="Reset", width=8,
                  command=lambda: self.reset_value(self.vignette_strength, self.DEFAULT_VIGNETTE)).grid(row=row, column=3, padx=5)
        row += 1
        
        # COLUMN 3
        row = 0
        # Saturation
        self.create_large_checkbox(col3, "Saturation Reduction", self.saturation_enabled, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col3, text="Amount:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        self.saturation_entry = ttk.Entry(col3, textvariable=self.saturation_reduction, width=10, font=("Arial", 11))
        self.saturation_entry.grid(row=row, column=2, padx=5)
        self.saturation_entry.bind('<Return>', lambda e: self.update_preview())
        self.saturation_entry.bind('<FocusOut>', lambda e: self.update_preview())
        ttk.Button(col3, text="Reset", width=8,
                  command=lambda: self.reset_value(self.saturation_reduction, self.DEFAULT_SATURATION)).grid(row=row, column=3, padx=5)
        row += 1
        
        # COLUMN 4
        row = 0
        # Chromatic aberration
        self.create_large_checkbox(col4, "Chromatic Aberration", self.chromatic_enabled, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col4, text="Amount:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        self.aberration_entry = ttk.Entry(col4, textvariable=self.chromatic_aberration, width=10, font=("Arial", 11))
        self.aberration_entry.grid(row=row, column=2, padx=5)
        self.aberration_entry.bind('<Return>', lambda e: self.update_preview())
        self.aberration_entry.bind('<FocusOut>', lambda e: self.update_preview())
        ttk.Button(col4, text="Reset", width=8,
                  command=lambda: self.reset_value(self.chromatic_aberration, self.DEFAULT_CHROMATIC)).grid(row=row, column=3, padx=5)
        
        # Additional options in columns below
        row = 1
        self.create_large_checkbox(col1, "Photo Border", self.vintage_border).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        self.create_large_checkbox(col2, "Unsharp Sharpening", self.unsharp_sharpening).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1
        self.create_large_checkbox(col2, "  ↳ Apply at 50%", self.unsharp_half_strength, font_size=11).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row = 1
        self.create_large_checkbox(col3, "Auto-Contrast Stretch", self.auto_contrast_stretch).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        self.create_large_checkbox(col4, "Show Face Center", self.show_ca_center).grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        # Tab 2: Photographic Effects - Part 1 (FOUR COLUMNS)
        photo_tab1 = ttk.Frame(notebook, padding=10)
        notebook.add(photo_tab1, text="Photo Effects 1")
        
        # Create four column frames
        col1_p1 = ttk.Frame(photo_tab1)
        col1_p1.grid(row=0, column=0, padx=10, sticky='n')
        col2_p1 = ttk.Frame(photo_tab1)
        col2_p1.grid(row=0, column=1, padx=10, sticky='n')
        col3_p1 = ttk.Frame(photo_tab1)
        col3_p1.grid(row=0, column=2, padx=10, sticky='n')
        col4_p1 = ttk.Frame(photo_tab1)
        col4_p1.grid(row=0, column=3, padx=10, sticky='n')
        
        # COLUMN 1
        row = 0
        # Tone curve
        self.create_large_checkbox(col1_p1, "Tone Curve (S-Curve)", self.apply_tone_curve, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col1_p1, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col1_p1, textvariable=self.tone_strength, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # COLUMN 2
        row = 0
        # Split toning
        self.create_large_checkbox(col2_p1, "Split Toning", self.apply_split_tone, bold=True).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        row += 1
        ttk.Label(col2_p1, text="(warm shadows/cool)", font=("Arial", 10, "italic")).grid(row=row, column=0, columnspan=3, sticky='w', padx=20, pady=0)
        row += 1
        
        # COLUMN 3
        row = 0
        # Photo grain
        self.create_large_checkbox(col3_p1, "Photo Grain", self.apply_photo_grain, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col3_p1, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col3_p1, textvariable=self.photo_grain_strength, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # COLUMN 4
        row = 0
        # Photo chromatic aberration
        self.create_large_checkbox(col4_p1, "Photo Chromatic Aberr.", self.apply_photo_chromatic, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col4_p1, text="Shift (px):", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col4_p1, textvariable=self.ca_shift, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # Second row of controls
        row = 1
        # Halation in column 1
        self.create_large_checkbox(col1_p1, "Halation (Glow)", self.apply_halation, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col1_p1, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col1_p1, textvariable=self.halation_strength, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # Photo vignette in column 2
        row = 1
        self.create_large_checkbox(col2_p1, "Photo Vignette", self.apply_photo_vignette, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col2_p1, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col2_p1, textvariable=self.photo_vignette_strength, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # Tab 3: Photographic Effects - Part 2 (FOUR COLUMNS)
        photo_tab2 = ttk.Frame(notebook, padding=10)
        notebook.add(photo_tab2, text="Photo Effects 2")
        
        # Create four column frames
        col1_p2 = ttk.Frame(photo_tab2)
        col1_p2.grid(row=0, column=0, padx=10, sticky='n')
        col2_p2 = ttk.Frame(photo_tab2)
        col2_p2.grid(row=0, column=1, padx=10, sticky='n')
        col3_p2 = ttk.Frame(photo_tab2)
        col3_p2.grid(row=0, column=2, padx=10, sticky='n')
        col4_p2 = ttk.Frame(photo_tab2)
        col4_p2.grid(row=0, column=3, padx=10, sticky='n')
        
        # COLUMN 1
        row = 0
        # Scan banding
        self.create_large_checkbox(col1_p2, "Scan Banding", self.apply_scan_banding, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col1_p2, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col1_p2, textvariable=self.banding_strength, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # COLUMN 2
        row = 0
        # RGB misalignment
        self.create_large_checkbox(col2_p2, "RGB Misalignment", self.apply_rgb_misalignment, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col2_p2, text="Shift (px):", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col2_p2, textvariable=self.misalign_px, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # COLUMN 3
        row = 0
        # Dust & scratches
        self.create_large_checkbox(col3_p2, "Dust & Scratches", self.apply_dust, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col3_p2, text="Dust:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col3_p2, textvariable=self.dust_amount, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        ttk.Label(col3_p2, text="Scratches:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col3_p2, textvariable=self.scratch_amount, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        
        # COLUMN 4
        row = 0
        # JPEG artifacts
        self.create_large_checkbox(col4_p2, "JPEG Artifacts", self.apply_jpeg_artifacts, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col4_p2, text="Quality:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col4_p2, textvariable=self.jpeg_quality, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        ttk.Label(col4_p2, text="(1-100)", font=("Arial", 9, "italic")).grid(row=row, column=0, columnspan=3, sticky='w', padx=20, pady=0)
        row += 1
        
        # Second row of controls
        row = 1
        # Lens distortion in column 1
        self.create_large_checkbox(col1_p2, "Lens Distortion", self.apply_lens_distortion, bold=True).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(col1_p2, text="Strength:", font=("Arial", 11)).grid(row=row, column=1, sticky='e', padx=5)
        ttk.Entry(col1_p2, textvariable=self.distortion_strength, width=10, font=("Arial", 11)).grid(row=row, column=2, padx=5)
        row += 1
        ttk.Label(col1_p2, text="(-ve=barrel, +ve=pincushion)", font=("Arial", 9, "italic")).grid(row=row, column=0, columnspan=3, sticky='w', padx=20, pady=0)
        row += 1
        
        # Dithering in column 2
        row = 1
        self.create_large_checkbox(col2_p2, "Color Dithering", self.apply_dithering, bold=True).grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        row += 1
        
        # Dithering type radio buttons
        ttk.Label(col2_p2, text="Type:", font=("Arial", 11)).grid(row=row, column=0, sticky='w', padx=20, pady=2)
        rb1 = tk.Radiobutton(col2_p2, text="Floyd-Steinberg", variable=self.dithering_type, value="floyd-steinberg",
                      command=self.update_preview, bg='#EBE1D2', selectcolor='#606060',
                      font=("Arial", 11), borderwidth=2, pady=3)
        rb1.grid(row=row, column=1, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1
        rb2 = tk.Radiobutton(col2_p2, text="Bayer", variable=self.dithering_type, value="bayer",
                      command=self.update_preview, bg='#EBE1D2', selectcolor='#606060',
                      font=("Arial", 11), borderwidth=2, pady=3)
        rb2.grid(row=row, column=1, columnspan=2, sticky='w', padx=5, pady=2)
        row += 1
        
        ttk.Label(col2_p2, text="Colors/ch:", font=("Arial", 11)).grid(row=row, column=0, sticky='w', padx=20, pady=2)
        ttk.Entry(col2_p2, textvariable=self.dithering_colors, width=10, font=("Arial", 11)).grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
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
        
    def reset_value(self, variable, default_value):
        """Reset a specific value to its default"""
        variable.set(default_value)
        self.update_preview()
        
    def restore_all_defaults(self):
        """Restore all values to their defaults"""
        self.grain_intensity.set(self.DEFAULT_GRAIN)
        self.vignette_strength.set(self.DEFAULT_VIGNETTE)
        self.saturation_reduction.set(self.DEFAULT_SATURATION)
        self.chromatic_aberration.set(self.DEFAULT_CHROMATIC)
        
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
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # Convert RGBA to RGB by removing alpha channel
                rgb_image = image_np[:, :, :3]
            elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
                rgb_image = image_np
            else:
                # Handle grayscale images
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
        
        # Apply new photographic effects if any are enabled
        if (self.apply_tone_curve.get() or self.apply_split_tone.get() or 
            self.apply_photo_grain.get() or self.apply_photo_chromatic.get() or 
            self.apply_halation.get() or self.apply_photo_vignette.get() or 
            self.apply_scan_banding.get() or self.apply_rgb_misalignment.get() or 
            self.apply_dust.get() or self.apply_jpeg_artifacts.get() or 
            self.apply_lens_distortion.get()):
            
            # Convert PIL to numpy/cv2 format (BGR)
            current_result = self.apply_photographic_effects(current_result)
        
        # Apply dithering as final step if enabled
        if self.apply_dithering.get():
            current_result = self.apply_dithering_effect(current_result)
        
        return current_result
    
    def apply_photographic_effects(self, pil_image):
        """Apply photographic effects using the make_image_look_photographed function"""
        try:
            # Convert PIL image to numpy array (RGB)
            img_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            if FACE_DETECTION_AVAILABLE:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                # If cv2 not available, just flip channels manually
                img_bgr = img_array[:, :, ::-1].copy()
            
            # Apply the photographic effects
            result_bgr = make_image_look_photographed(
                img_bgr,
                apply_tone_curve=self.apply_tone_curve.get(),
                tone_strength=self.tone_strength.get(),
                apply_split_tone=self.apply_split_tone.get(),
                apply_grain=self.apply_photo_grain.get(),
                grain_strength=self.photo_grain_strength.get(),
                apply_chromatic_aberration=self.apply_photo_chromatic.get(),
                ca_shift=self.ca_shift.get(),
                apply_halation=self.apply_halation.get(),
                halation_strength=self.halation_strength.get(),
                apply_vignette=self.apply_photo_vignette.get(),
                vignette_strength=self.photo_vignette_strength.get(),
                apply_scan_banding=self.apply_scan_banding.get(),
                banding_strength=self.banding_strength.get(),
                apply_rgb_misalignment=self.apply_rgb_misalignment.get(),
                misalign_px=self.misalign_px.get(),
                apply_dust=self.apply_dust.get(),
                dust_amount=self.dust_amount.get(),
                scratch_amount=self.scratch_amount.get(),
                apply_jpeg_artifacts=self.apply_jpeg_artifacts.get(),
                jpeg_quality=self.jpeg_quality.get(),
                apply_lens_distortion=self.apply_lens_distortion.get(),
                distortion_strength=self.distortion_strength.get(),
                apply_lut=False,  # LUT not implemented in UI yet
                lut=None
            )
            
            # Convert back to RGB for PIL
            if FACE_DETECTION_AVAILABLE:
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            else:
                result_rgb = result_bgr[:, :, ::-1].copy()
            
            # Convert back to PIL Image
            return Image.fromarray(result_rgb)
            
        except Exception as e:
            print(f"Error applying photographic effects: {e}")
            return pil_image
    
    def apply_dithering_effect(self, image):
        """Apply Floyd-Steinberg or Bayer dithering to the image (optimized)"""
        try:
            img_array = np.array(image).astype(np.float32)
            h, w, c = img_array.shape
            colors_per_channel = max(2, min(256, self.dithering_colors.get()))
            
            if self.dithering_type.get() == "floyd-steinberg":
                # Floyd-Steinberg dithering - vectorized per-row
                result = img_array.copy()
                
                # Pre-calculate quantization steps for efficiency
                scale_factor = (colors_per_channel - 1) / 255.0
                inv_scale_factor = 255.0 / (colors_per_channel - 1)
                
                for y in range(h):
                    for x in range(w):
                        old_pixel = result[y, x].copy()
                        
                        # Quantize to reduced color palette
                        new_pixel = np.round(old_pixel * scale_factor) * inv_scale_factor
                        result[y, x] = new_pixel
                        
                        # Calculate quantization error
                        quant_error = old_pixel - new_pixel
                        
                        # Distribute error to neighboring pixels
                        if x + 1 < w:
                            result[y, x + 1] += quant_error * 7/16
                        if y + 1 < h:
                            if x > 0:
                                result[y + 1, x - 1] += quant_error * 3/16
                            result[y + 1, x] += quant_error * 5/16
                            if x + 1 < w:
                                result[y + 1, x + 1] += quant_error * 1/16
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            else:  # Bayer dithering - FULLY VECTORIZED
                # 8x8 Bayer matrix
                bayer_matrix = np.array([
                    [ 0, 32,  8, 40,  2, 34, 10, 42],
                    [48, 16, 56, 24, 50, 18, 58, 26],
                    [12, 44,  4, 36, 14, 46,  6, 38],
                    [60, 28, 52, 20, 62, 30, 54, 22],
                    [ 3, 35, 11, 43,  1, 33,  9, 41],
                    [51, 19, 59, 27, 49, 17, 57, 25],
                    [15, 47,  7, 39, 13, 45,  5, 37],
                    [63, 31, 55, 23, 61, 29, 53, 21]
                ]) / 64.0  # Normalize to 0-1
                
                # Tile the Bayer matrix to cover the entire image
                tile_h = (h + 7) // 8  # Ceiling division
                tile_w = (w + 7) // 8
                tiled_bayer = np.tile(bayer_matrix, (tile_h, tile_w))[:h, :w]
                
                # Add channel dimension for broadcasting
                tiled_bayer = tiled_bayer[:, :, np.newaxis]
                
                # Apply threshold and quantize all pixels at once
                threshold = tiled_bayer * (255.0 / colors_per_channel)
                adjusted = img_array + threshold - 127.5 / colors_per_channel
                
                # Vectorized quantization
                scale_factor = (colors_per_channel - 1) / 255.0
                inv_scale_factor = 255.0 / (colors_per_channel - 1)
                result = np.round(adjusted * scale_factor) * inv_scale_factor
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"Error applying dithering: {e}")
            return image
        
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
            from PIL import ImageFilter, ImageChops
            
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
            
            # If half-strength is enabled, blend at 50% with original
            if self.unsharp_half_strength.get():
                # Blend: 50% original + 50% sharpened
                return Image.blend(image, sharpened, 0.5)
            else:
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


def make_image_look_photographed(
    img,

    # tonal shaping
    apply_tone_curve=True,
    tone_strength=0.5,

    # colour shaping
    apply_split_tone=True,
    shadow_tint=(1.05, 1.0, 0.95),
    highlight_tint=(0.95, 1.0, 1.05),

    # grain
    apply_grain=True,
    grain_strength=0.04,

    # lens imperfections
    apply_chromatic_aberration=True,
    ca_shift=1,

    apply_halation=True,
    halation_strength=0.2,

    apply_vignette=True,
    vignette_strength=0.5,

    # scan / print artifacts
    apply_scan_banding=True,
    banding_strength=0.01,

    apply_rgb_misalignment=True,
    misalign_px=1,

    # texture overlay
    paper_texture=None,
    texture_strength=0.1,

    # NEW: dust & scratches
    apply_dust=True,
    dust_amount=0.001,
    scratch_amount=0.0002,

    # NEW: JPEG compression artifacts
    apply_jpeg_artifacts=True,
    jpeg_quality=70,

    # NEW: lens distortion
    apply_lens_distortion=True,
    distortion_strength=-0.0005,  # negative = barrel, positive = pincushion

    # NEW: LUT (film stock)
    apply_lut=False,
    lut=None,  # 256x1x3 LUT or 3D LUT converted to 1D
):
    """
    Takes an image (NumPy array, BGR 0–255) and applies a stack of
    realism-enhancing effects to simulate a photographed print/scan.
    """

    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # -------------------------------
    # 1. Tone curve
    # -------------------------------
    if apply_tone_curve:
        def s_curve(x, s=tone_strength):
            return 1 / (1 + np.exp(-s * (x - 0.5)))
        img = s_curve(img)

    # -------------------------------
    # 2. Split toning
    # -------------------------------
    if apply_split_tone:
        shadows = np.clip(img * shadow_tint, 0, 1)
        highlights = np.clip(img * highlight_tint, 0, 1)
        mask = img > 0.5
        img = np.where(mask, highlights, shadows)

    # -------------------------------
    # 3. Film grain
    # -------------------------------
    if apply_grain:
        grain = np.random.normal(0, grain_strength, (h, w, 1))
        grain = np.repeat(grain, 3, axis=2)
        luma = img.mean(axis=2, keepdims=True)
        grain *= (0.5 - np.abs(luma - 0.5)) * 2
        img = np.clip(img + grain, 0, 1)

    # -------------------------------
    # 4. Chromatic aberration
    # -------------------------------
    if apply_chromatic_aberration:
        def shift_channel(ch, dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(ch, M, (w, h))

        b, g, r = cv2.split(img)
        r = shift_channel(r, ca_shift, 0)
        b = shift_channel(b, -ca_shift, 0)
        img = cv2.merge([b, g, r])

    # -------------------------------
    # 5. Halation
    # -------------------------------
    if apply_halation:
        blur = cv2.GaussianBlur(img, (0, 0), 5)
        img = np.clip(img + blur * halation_strength, 0, 1)

    # -------------------------------
    # 6. Vignette
    # -------------------------------
    if apply_vignette:
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        vignette = 1 - (dist / dist.max())**2 * vignette_strength
        vignette = vignette[..., None]
        img *= vignette

    # -------------------------------
    # 7. Scan banding
    # -------------------------------
    if apply_scan_banding:
        bands = (np.sin(np.linspace(0, 50, h)) * banding_strength).reshape(h, 1, 1)
        img = np.clip(img + bands, 0, 1)

    # -------------------------------
    # 8. RGB misalignment
    # -------------------------------
    if apply_rgb_misalignment:
        def shift(ch, dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(ch, M, (w, h))

        b, g, r = cv2.split(img)
        r = shift(r, misalign_px, 0)
        g = shift(g, 0, -misalign_px)
        b = shift(b, -misalign_px, misalign_px)
        img = cv2.merge([b, g, r])

    # -------------------------------
    # 9. Paper texture
    # -------------------------------
    if paper_texture is not None:
        tex = cv2.resize(paper_texture.astype(np.float32) / 255.0, (w, h))
        img = img * (1 - texture_strength) + tex * texture_strength

    # -------------------------------
    # 10. Dust & scratches (IMPROVED)
    # -------------------------------
    if apply_dust and (dust_amount > 0 or scratch_amount > 0):
        # Create dust (dark spots)
        if dust_amount > 0:
            num_dust = int(h * w * dust_amount)
            if num_dust > 0:
                dust_y = np.random.randint(0, h, num_dust)
                dust_x = np.random.randint(0, w, num_dust)
                dust_sizes = np.random.randint(1, 4, num_dust)  # Small spots
                
                for y, x, size in zip(dust_y, dust_x, dust_sizes):
                    y1, y2 = max(0, y-size), min(h, y+size)
                    x1, x2 = max(0, x-size), min(w, x+size)
                    # Darken the spot slightly
                    img[y1:y2, x1:x2] *= 0.7
        
        # Create scratches (light lines)
        if scratch_amount > 0:
            num_scratches = int(h * scratch_amount * 10)  # Adjust multiplier for visible effect
            if num_scratches > 0:
                for _ in range(num_scratches):
                    x = np.random.randint(0, w)
                    y_start = np.random.randint(0, h // 2)
                    y_end = y_start + np.random.randint(h // 4, h)
                    y_end = min(y_end, h)
                    thickness = np.random.randint(1, 2)
                    
                    # Draw a light vertical scratch
                    x1, x2 = max(0, x-thickness), min(w, x+thickness)
                    img[y_start:y_end, x1:x2] = np.clip(img[y_start:y_end, x1:x2] + 0.15, 0, 1)

    # -------------------------------
    # 11. JPEG artifacts
    # -------------------------------
    if apply_jpeg_artifacts:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, enc = cv2.imencode(".jpg", (img * 255).astype(np.uint8), encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # -------------------------------
    # 12. Lens distortion
    # -------------------------------
    if apply_lens_distortion:
        K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        D = np.array([distortion_strength, 0, 0, 0], dtype=np.float32)
        img = cv2.undistort(img, K, D)

    # -------------------------------
    # 13. LUT (film stock)
    # -------------------------------
    if apply_lut and lut is not None:
        img_uint = (img * 255).astype(np.uint8)
        img = cv2.LUT(img_uint, lut).astype(np.float32) / 255.0

    return (img * 255).astype(np.uint8)

        
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
