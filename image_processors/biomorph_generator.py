#!/usr/bin/env python3
"""
Clifford Pickover Biomorphs Generator - Interactive fractal generator with GUI controls
Based on the classic DOS BASIC radiolarian biomorph algorithm
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime

class BiomorphGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Clifford Pickover Biomorphs Generator")
        self.root.geometry("1600x900")  # Large window for full HD screen
        self.root.configure(bg='#2c2c2c')  # Dark background
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Variables
        self.fractal_image = None
        self.display_image = None
        self.is_generating = False
        self.generation_thread = None
        
        # Fractal parameters
        self.const_real = tk.DoubleVar(value=0.5)
        self.const_imag = tk.DoubleVar(value=0.0)
        self.zoom = tk.DoubleVar(value=2.5)
        self.center_x = tk.DoubleVar(value=0.0)
        self.center_y = tk.DoubleVar(value=0.0)
        self.max_iterations = tk.IntVar(value=100)
        self.escape_radius = tk.DoubleVar(value=10.0)
        self.image_width = tk.IntVar(value=1200)
        self.image_height = tk.IntVar(value=800)
        
        # Color scheme
        self.invert_colors = tk.BooleanVar(value=False)
        self.use_color_palette = tk.BooleanVar(value=True)
        self.color_palette = tk.StringVar(value="vibrant")
        
        # Auto-generation control
        self.auto_generate_enabled = tk.BooleanVar(value=False)  # Disabled by default
        self.pending_auto_generate = None  # Track pending auto-generation
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with dark theme
        main_frame = tk.Frame(self.root, bg='#2c2c2c')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left column - Controls (fixed width)
        left_column = tk.Frame(main_frame, bg='#2c2c2c', width=350)
        left_column.pack(side=tk.LEFT, fill='y', padx=(0, 10))
        left_column.pack_propagate(False)
        
        # Title
        title_label = tk.Label(left_column, text="🦠 Biomorphs Generator", 
                              font=("Arial", 16, "bold"), fg='white', bg='#2c2c2c')
        title_label.pack(pady=(0, 15))
        
        # Fractal Parameters
        params_frame = tk.LabelFrame(left_column, text="Fractal Parameters", 
                                   font=("Arial", 12, "bold"), fg='white', bg='#3c3c3c', 
                                   relief='ridge', bd=2)
        params_frame.pack(fill='x', pady=5)
        
        # Constant Real part
        self.create_slider(params_frame, "Constant (Real)", self.const_real, 
                          -2.0, 2.0, 0.01, "Controls the real part of the constant added each iteration")
        
        # Constant Imaginary part  
        self.create_slider(params_frame, "Constant (Imaginary)", self.const_imag, 
                          -2.0, 2.0, 0.01, "Controls the imaginary part of the constant")
        
        # Zoom level
        self.create_slider(params_frame, "Zoom Level", self.zoom, 
                          0.1, 10.0, 0.1, "Controls how zoomed in the view is")
        
        # Center X
        self.create_slider(params_frame, "Center X", self.center_x, 
                          -5.0, 5.0, 0.1, "Horizontal center of the view")
        
        # Center Y  
        self.create_slider(params_frame, "Center Y", self.center_y, 
                          -5.0, 5.0, 0.1, "Vertical center of the view")
        
        # Max Iterations
        self.create_slider(params_frame, "Max Iterations", self.max_iterations, 
                          10, 1000, 1, "Maximum iterations before considering a point stable", is_int=True)
        
        # Escape Radius
        self.create_slider(params_frame, "Escape Radius", self.escape_radius, 
                          2.0, 50.0, 0.5, "Radius beyond which points are considered to escape")
        
        # Image dimensions
        dims_frame = tk.Frame(params_frame, bg='#3c3c3c')
        dims_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(dims_frame, text="Image Size:", fg='white', bg='#3c3c3c', 
                font=("Arial", 10, "bold")).pack(anchor='w')
        
        # Fixed to 1200x800
        size_label = tk.Label(dims_frame, text="1200 x 800 pixels", fg='yellow', bg='#3c3c3c', 
                             font=("Arial", 10, "bold"))
        size_label.pack(anchor='w', padx=10)
        
        # Display Options
        display_frame = tk.LabelFrame(left_column, text="Display Options", 
                                    font=("Arial", 12, "bold"), fg='white', bg='#3c3c3c', 
                                    relief='ridge', bd=2)
        display_frame.pack(fill='x', pady=10)
        
        # Color palette options
        color_frame = tk.Frame(display_frame, bg='#3c3c3c')
        color_frame.pack(fill='x', padx=10, pady=5)
        
        use_color_check = tk.Checkbutton(color_frame, text="Use Color Palette", 
                                        variable=self.use_color_palette, command=self.on_parameter_change,
                                        fg='white', bg='#3c3c3c', selectcolor='#4c4c4c',
                                        activeforeground='white', activebackground='#3c3c3c')
        use_color_check.pack(anchor='w', pady=2)
        
        # Color palette selection
        palette_frame = tk.Frame(color_frame, bg='#3c3c3c')
        palette_frame.pack(fill='x', pady=2)
        
        tk.Label(palette_frame, text="Palette:", fg='white', bg='#3c3c3c', 
                font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        palette_combo = ttk.Combobox(palette_frame, textvariable=self.color_palette,
                                    values=["vibrant", "rainbow", "fire", "ocean", "plasma", "sunset", "forest"], 
                                    state="readonly", width=12)
        palette_combo.pack(side=tk.RIGHT, padx=5)
        palette_combo.bind('<<ComboboxSelected>>', self.on_parameter_change)
        
        # Color inversion
        invert_check = tk.Checkbutton(display_frame, text="Invert Colors", 
                                     variable=self.invert_colors, command=self.on_parameter_change,
                                     fg='white', bg='#3c3c3c', selectcolor='#4c4c4c',
                                     activeforeground='white', activebackground='#3c3c3c')
        invert_check.pack(anchor='w', padx=10, pady=5)
        
        # Auto-generation control
        auto_gen_check = tk.Checkbutton(display_frame, text="Auto-generate on parameter change", 
                                       variable=self.auto_generate_enabled,
                                       fg='white', bg='#3c3c3c', selectcolor='#4c4c4c',
                                       activeforeground='white', activebackground='#3c3c3c')
        auto_gen_check.pack(anchor='w', padx=10, pady=5)
        
        # Preset buttons
        preset_frame = tk.LabelFrame(left_column, text="Presets", 
                                   font=("Arial", 12, "bold"), fg='white', bg='#3c3c3c', 
                                   relief='ridge', bd=2)
        preset_frame.pack(fill='x', pady=10)
        
        presets = [
            ("Classic Radiolarian", {"real": 0.5, "imag": 0.0}),
            ("Spiral Biomorph", {"real": 0.7, "imag": 0.2}),
            ("Complex Branch", {"real": -0.3, "imag": 0.8}),
            ("Delicate Web", {"real": 0.1, "imag": -0.6}),
            ("Dense Forest", {"real": -0.8, "imag": 0.3}),
            ("Radial Pattern", {"real": 0.0, "imag": 1.0})
        ]
        
        for i, (name, params) in enumerate(presets):
            btn = tk.Button(preset_frame, text=name, 
                          command=lambda p=params: self.load_preset(p),
                          bg='#4c4c4c', fg='white', font=("Arial", 9),
                          relief='raised', bd=1, padx=5, pady=2)
            btn.pack(fill='x', padx=5, pady=2)
        
        # Control buttons
        button_frame = tk.Frame(left_column, bg='#2c2c2c')
        button_frame.pack(fill='x', pady=15)
        
        # Generate button
        self.generate_btn = tk.Button(button_frame, text="🎯 Generate Fractal", 
                                     command=self.generate_fractal, 
                                     bg='#0d7377', fg='white', font=("Arial", 11, "bold"),
                                     relief='raised', bd=2, padx=10, pady=5)
        self.generate_btn.pack(fill='x', pady=2)
        
        # Save button
        self.save_btn = tk.Button(button_frame, text="💾 Save Fractal", 
                                 command=self.save_fractal, state='disabled',
                                 bg='#14a085', fg='white', font=("Arial", 11, "bold"),
                                 relief='raised', bd=2, padx=10, pady=5)
        self.save_btn.pack(fill='x', pady=2)
        
        # Stop button
        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop Generation", 
                                 command=self.stop_generation, state='disabled',
                                 bg='#c44536', fg='white', font=("Arial", 11, "bold"),
                                 relief='raised', bd=2, padx=10, pady=5)
        self.stop_btn.pack(fill='x', pady=2)
        
        # Progress
        self.progress_var = tk.StringVar(value="Ready to generate fractal")
        progress_label = tk.Label(left_column, textvariable=self.progress_var, 
                                 fg='white', bg='#2c2c2c', font=("Arial", 9))
        progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(left_column, length=320, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        
        # Right column - Fractal Display
        right_column = tk.Frame(main_frame, bg='#2c2c2c')
        right_column.pack(side=tk.RIGHT, fill='both', expand=True, padx=(10, 0))
        
        # Display frame
        display_frame = tk.LabelFrame(right_column, text="Biomorph Fractal", 
                                    font=("Arial", 14, "bold"), fg='white', bg='#3c3c3c', 
                                    relief='ridge', bd=2)
        display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Canvas for fractal display
        canvas_frame = tk.Frame(display_frame, bg='#3c3c3c')
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg='black', width=720, height=480)
        
        # Scrollbars for large images
        v_scrollbar = tk.Scrollbar(canvas_frame, orient='vertical', command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient='horizontal', command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill='y')
        h_scrollbar.pack(side=tk.BOTTOM, fill='x')
        self.canvas.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Click to zoom functionality
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Info label
        info_label = tk.Label(right_column, 
                             text="Click on the fractal to center view at that point", 
                             fg='#888888', bg='#2c2c2c', font=("Arial", 10))
        info_label.pack(pady=5)
        
    def create_slider(self, parent, label, variable, min_val, max_val, resolution, tooltip, is_int=False):
        """Create a labeled slider with tooltip"""
        frame = tk.Frame(parent, bg='#3c3c3c')
        frame.pack(fill='x', padx=5, pady=3)
        
        # Label
        label_widget = tk.Label(frame, text=f"{label}:", fg='white', bg='#3c3c3c', 
                               font=("Arial", 10, "bold"), width=18, anchor='w')
        label_widget.pack(side=tk.LEFT)
        
        # Value display
        if is_int:
            value_label = tk.Label(frame, text=f"{variable.get():.0f}", fg='yellow', bg='#3c3c3c', 
                                  font=("Arial", 10), width=8)
        else:
            value_label = tk.Label(frame, text=f"{variable.get():.2f}", fg='yellow', bg='#3c3c3c', 
                                  font=("Arial", 10), width=8)
        value_label.pack(side=tk.RIGHT)
        
        # Slider
        slider = tk.Scale(frame, from_=min_val, to=max_val, resolution=resolution,
                         variable=variable, orient='horizontal', 
                         bg='#4c4c4c', fg='white', troughcolor='#6c6c6c',
                         highlightbackground='#3c3c3c', showvalue=False,
                         command=lambda val, vl=value_label, is_i=is_int: self.update_value_label(vl, val, is_i))
        slider.pack(fill='x', padx=5)
        
        # Bind parameter change
        variable.trace('w', self.on_parameter_change)
        
        # Tooltip (simple version)
        self.create_tooltip(slider, tooltip)
        
        return slider
        
    def create_tooltip(self, widget, text):
        """Simple tooltip implementation"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background='#ffffe0', 
                           relief='solid', borderwidth=1, font=("Arial", 8))
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        
    def update_value_label(self, label, value, is_int):
        """Update the value display label"""
        if is_int:
            label.config(text=f"{float(value):.0f}")
        else:
            label.config(text=f"{float(value):.2f}")
            
    def on_parameter_change(self, *args):
        """Called when any parameter changes"""
        # Only auto-generate if enabled and not currently generating
        if self.auto_generate_enabled.get() and not self.is_generating:
            # Cancel any pending auto-generation
            if self.pending_auto_generate:
                self.root.after_cancel(self.pending_auto_generate)
            # Schedule new auto-generation with delay
            self.pending_auto_generate = self.root.after(500, self.auto_generate)
            
    def auto_generate(self):
        """Auto-generate fractal after parameter changes"""
        self.pending_auto_generate = None  # Clear pending flag
        if not self.is_generating and self.auto_generate_enabled.get():
            self.generate_fractal()
            
    def load_preset(self, params):
        """Load a preset configuration"""
        # Temporarily disable auto-generation during preset loading
        was_auto_enabled = self.auto_generate_enabled.get()
        self.auto_generate_enabled.set(False)
        
        # Load preset values
        self.const_real.set(params["real"])
        self.const_imag.set(params["imag"])
        
        # Restore auto-generation setting
        self.auto_generate_enabled.set(was_auto_enabled)
        
        # Generate once after loading preset
        if not self.is_generating:
            self.generate_fractal()
        
    def on_canvas_click(self, event):
        """Handle clicks on canvas to center view"""
        if self.fractal_image:
            # Convert canvas coordinates to fractal coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Get current view parameters
            zoom = self.zoom.get()
            center_x = self.center_x.get()
            center_y = self.center_y.get()
            width = self.image_width.get()
            height = self.image_height.get()
            
            # Calculate world coordinates
            aspect_ratio = width / height
            x_range = zoom * 2
            y_range = zoom * 2 / aspect_ratio
            
            # Convert to world coordinates
            world_x = center_x + (canvas_x - width/2) * x_range / width
            world_y = center_y + (canvas_y - height/2) * y_range / height
            
            # Update center
            self.center_x.set(world_x)
            self.center_y.set(world_y)
            
    def generate_biomorph(self, width, height, const_real, const_imag, zoom, center_x, center_y, 
                         max_iter, escape_radius, progress_callback=None):
        """Generate the biomorph fractal"""
        
        # Calculate coordinate bounds with proper aspect ratio
        aspect_ratio = width / height
        ymax_val = zoom  # Use zoom as the y-range
        ymin_val = -ymax_val
        # Adjust x-range based on aspect ratio
        xmax_val = ymax_val * aspect_ratio  
        xmin_val = -xmax_val
        
        # Apply center offset
        xmin = center_x + xmin_val
        xmax = center_x + xmax_val
        ymin = center_y + ymin_val
        ymax = center_y + ymax_val
        
        # Create result array for iteration counts
        result = np.zeros((height, width), dtype=np.int32)
        
        for i in range(height):
            if progress_callback and i % 10 == 0:
                progress = (i / height) * 100
                progress_callback(progress)
                
            if not self.is_generating:  # Check for stop request
                return None
                
            for j in range(width):
                # Map pixel to complex plane (matching BASIC: x0 = xmin + (xmax - xmin) * j / jlimit)
                x0 = xmin + (xmax - xmin) * j / (width - 1)
                y0 = -ymin - (ymax - ymin) * i / (height - 1)  # Match BASIC: y0 = -ymin - (ymax - ymin) * i / ilimit
                
                x, y = x0, y0
                
                # Iterate the biomorph equation (matching original BASIC)
                iteration_count = 0
                for n in range(1, max_iter + 1):
                    # Compute z^3 + c (the core biomorph equation)
                    xx = x * (x * x - 3 * y * y) + const_real
                    yy = y * (3 * x * x - y * y) + const_imag
                    
                    x, y = xx, yy
                    iteration_count = n
                    
                    # Check escape condition (matching original: ABS(x) > 10 OR ABS(y) > 10 OR x*x + y*y > 10^2)
                    if abs(x) > escape_radius or abs(y) > escape_radius or x*x + y*y > escape_radius*escape_radius:
                        break
                        
                # Store iteration count (0 means it never escaped - the biomorph itself)
                if abs(x) < escape_radius and abs(y) < escape_radius:
                    result[i, j] = 0  # Stable point - the biomorph structure
                else:
                    result[i, j] = iteration_count  # Escaped - store iteration count
                    
        return result
        
    def generate_fractal(self):
        """Generate fractal in a separate thread"""
        if self.is_generating:
            return
            
        self.is_generating = True
        self.generate_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.save_btn.config(state='disabled')
        self.progress_bar.start()
        
        def generation_worker():
            try:
                def update_progress(progress):
                    self.root.after(0, lambda: self.progress_var.set(f"Generating... {progress:.1f}%"))
                
                self.progress_var.set("Starting generation...")
                
                # Get parameters
                width = self.image_width.get()
                height = self.image_height.get()
                const_real = self.const_real.get()
                const_imag = self.const_imag.get()
                zoom = self.zoom.get()
                center_x = self.center_x.get()
                center_y = self.center_y.get()
                max_iter = self.max_iterations.get()
                escape_radius = self.escape_radius.get()
                
                # Debug: Print parameters
                print(f"Generating with: const_real={const_real}, const_imag={const_imag}")
                print(f"zoom={zoom}, center=({center_x},{center_y})")
                print(f"max_iter={max_iter}, escape_radius={escape_radius}, width={width}, height={height}")
                
                # Generate fractal
                fractal_data = self.generate_biomorph(
                    width, height, const_real, const_imag, zoom, center_x, center_y,
                    max_iter, escape_radius, update_progress)
                
                if fractal_data is not None and self.is_generating:
                    # Debug: Check iteration counts and distribution
                    biomorph_pixels = np.sum(fractal_data == 0)
                    escaped_pixels = np.sum(fractal_data > 0)
                    max_iterations_found = np.max(fractal_data)
                    
                    # Analyze iteration distribution
                    if escaped_pixels > 0:
                        escaped_data = fractal_data[fractal_data > 0]
                        mean_iter = np.mean(escaped_data)
                        median_iter = np.median(escaped_data)
                        unique_iters = len(np.unique(escaped_data))
                        print(f"Result: {biomorph_pixels} biomorph pixels, {escaped_pixels} escaped pixels")
                        print(f"Max iterations: {max_iterations_found}, Mean: {mean_iter:.1f}, Median: {median_iter:.1f}")
                        print(f"Unique iteration values: {unique_iters}")
                    else:
                        print(f"Result: {biomorph_pixels} biomorph pixels, {escaped_pixels} escaped pixels")
                        print(f"Max iterations found: {max_iterations_found}")
                    
                    # Convert iteration data to color image
                    if self.use_color_palette.get():
                        # Generate larger color palette for cycling
                        palette_size = 256  # Fixed palette size for smooth gradients
                        palette = self.generate_color_palette(self.color_palette.get(), palette_size)
                        palette_array = np.array(palette, dtype=np.uint8)
                        
                        # Create RGB image using vectorized operations
                        rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        # Handle biomorph structure (iteration count = 0)
                        biomorph_mask = (fractal_data == 0)
                        if self.invert_colors.get():
                            rgb_data[biomorph_mask] = [255, 255, 255]  # White biomorph
                        else:
                            rgb_data[biomorph_mask] = [0, 0, 0]  # Black biomorph
                        
                        # Handle escaped points (iteration count > 0)
                        escaped_mask = (fractal_data > 0)
                        if np.any(escaped_mask):
                            escaped_iterations = fractal_data[escaped_mask]
                            
                            # Ultra-aggressive color mapping for maximum visual variety
                            # Designed to create beautiful gradients even when median iteration = 1
                            
                            # Strategy 1: Extreme stretching for low values (1-10 iterations)
                            # Map iterations 1-10 across 80% of the palette
                            stretched_values = np.clip(escaped_iterations, 1, 10)
                            stretched_indices = ((stretched_values - 1) * palette_size * 0.8 / 9).astype(int)
                            
                            # Strategy 2: High-frequency cycling with prime modulo for organic patterns
                            # Use different prime numbers for R, G, B-like cycling
                            cycle1 = ((escaped_iterations * 47) % palette_size)  # Prime 47
                            cycle2 = ((escaped_iterations * 73) % palette_size)  # Prime 73
                            cycle3 = ((escaped_iterations * 31) % palette_size)  # Prime 31
                            multi_cycle = (cycle1 + cycle2 + cycle3) // 3
                            
                            # Strategy 3: Position-based variation for spatial coherence
                            # Create smooth spatial gradients based on actual 2D pixel coordinates
                            # Get the 2D coordinates of escaped pixels
                            y_coords, x_coords = np.where(escaped_mask)
                            
                            # Create non-linear spatial waves to avoid banding
                            # Use sine/cosine functions with different phases and frequencies
                            import math
                            
                            # Convert coordinates to normalized range [0,1]
                            norm_x = x_coords / width
                            norm_y = y_coords / height
                            
                            # Create multiple non-linear wave patterns with different frequencies
                            wave1 = np.sin(norm_x * 2 * math.pi * 3.7 + norm_y * 2 * math.pi * 2.3) * 0.5 + 0.5
                            wave2 = np.cos(norm_x * 2 * math.pi * 1.9 + norm_y * 2 * math.pi * 4.1) * 0.5 + 0.5
                            wave3 = np.sin(norm_x * 2 * math.pi * 5.1 - norm_y * 2 * math.pi * 1.7) * 0.5 + 0.5
                            
                            # Combine waves with smooth blending
                            spatial_blend = ((wave1 + wave2 + wave3) / 3 * palette_size).astype(int) % palette_size
                            
                            # Strategy 4: Fractal-inspired recursive coloring
                            # Use the iteration value itself to create nested patterns
                            fractal_indices = ((escaped_iterations * escaped_iterations * 13) % palette_size)
                            
                            # Strategy 5: Smooth interpolation between discrete values
                            # Add sub-pixel precision based on actual coordinates
                            coord_hash = (x_coords * 73 + y_coords * 101) % 256  # Pseudo-random but deterministic
                            smooth_offset = coord_hash / 256.0 * 5  # Subtle variation
                            smooth_indices = ((escaped_iterations - 1 + smooth_offset) * palette_size / 10).astype(int) % palette_size
                            
                            # Combine all strategies with optimized weights for maximum color variation
                            final_indices = (
                                stretched_indices * 0.4 +     # 40% extreme stretching for low values
                                multi_cycle * 0.25 +          # 25% multi-prime cycling
                                spatial_blend * 0.15 +        # 15% spatial coherence
                                fractal_indices * 0.1 +       # 10% fractal patterns
                                smooth_indices * 0.1          # 10% smooth interpolation
                            ).astype(int) % palette_size
                            
                            colors = palette_array[final_indices]
                            
                            if self.invert_colors.get():
                                colors = 255 - colors  # Invert colors
                            
                            rgb_data[escaped_mask] = colors
                        
                        # Create image from RGB data
                        self.fractal_image = Image.fromarray(rgb_data)
                    else:
                        # Convert to grayscale for traditional biomorph look
                        gray_data = np.zeros((height, width), dtype=np.uint8)
                        
                        for i in range(height):
                            for j in range(width):
                                if fractal_data[i, j] == 0:
                                    gray_data[i, j] = 0 if not self.invert_colors.get() else 255
                                else:
                                    gray_data[i, j] = 255 if not self.invert_colors.get() else 0
                        
                        # Create grayscale image
                        self.fractal_image = Image.fromarray(gray_data)
                    
                    # Update display
                    self.root.after(0, self.display_fractal)
                    self.root.after(0, lambda: self.progress_var.set("Generation complete!"))
                else:
                    self.root.after(0, lambda: self.progress_var.set("Generation stopped"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.progress_var.set(f"Error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Generation Error", f"Error generating fractal: {str(e)}"))
            finally:
                self.root.after(0, self.generation_finished)
                
        self.generation_thread = threading.Thread(target=generation_worker, daemon=True)
        self.generation_thread.start()
        
    def stop_generation(self):
        """Stop the current generation"""
        self.is_generating = False
        
    def generation_finished(self):
        """Called when generation is complete"""
        self.is_generating = False
        self.generate_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_bar.stop()
        
        if self.fractal_image:
            self.save_btn.config(state='normal')
            
    def display_fractal(self):
        """Display the generated fractal"""
        if self.fractal_image:
            # Create display version (may be scaled)
            display_img = self.fractal_image.copy()
            
            # Convert to RGB for display
            if display_img.mode != 'RGB':
                display_img = display_img.convert('RGB')
                
            # Create PhotoImage
            self.display_image = ImageTk.PhotoImage(display_img)
            
            # Clear canvas and display
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.display_image, anchor='nw')
            
            # Update scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
    def save_fractal(self):
        """Save the generated fractal with resolution options"""
        if not self.fractal_image:
            messagebox.showwarning("No Fractal", "Please generate a fractal first")
            return
            
        # Create save options dialog
        save_dialog = tk.Toplevel(self.root)
        save_dialog.title("Save Fractal Options")
        save_dialog.geometry("400x300")
        save_dialog.configure(bg='#2c2c2c')
        save_dialog.transient(self.root)
        save_dialog.grab_set()
        
        # Center the dialog
        save_dialog.update_idletasks()
        x = (save_dialog.winfo_screenwidth() // 2) - (save_dialog.winfo_width() // 2)
        y = (save_dialog.winfo_screenheight() // 2) - (save_dialog.winfo_height() // 2)
        save_dialog.geometry(f"+{x}+{y}")
        
        # Title
        title_label = tk.Label(save_dialog, text="💾 Save Fractal", 
                              font=("Arial", 16, "bold"), fg='white', bg='#2c2c2c')
        title_label.pack(pady=15)
        
        # Resolution options
        res_frame = tk.LabelFrame(save_dialog, text="Resolution Options", 
                                 font=("Arial", 12, "bold"), fg='white', bg='#3c3c3c', 
                                 relief='ridge', bd=2)
        res_frame.pack(fill='x', padx=20, pady=10)
        
        resolution_choice = tk.StringVar(value="1x")
        
        # 1x option
        res1x = tk.Radiobutton(res_frame, text="1x - Standard (1200x800)", 
                              variable=resolution_choice, value="1x",
                              fg='white', bg='#3c3c3c', selectcolor='#4c4c4c',
                              activeforeground='white', activebackground='#3c3c3c',
                              font=("Arial", 11))
        res1x.pack(anchor='w', padx=10, pady=5)
        
        # 2x option
        res2x = tk.Radiobutton(res_frame, text="2x - High Resolution (2400x1600)", 
                              variable=resolution_choice, value="2x",
                              fg='white', bg='#3c3c3c', selectcolor='#4c4c4c',
                              activeforeground='white', activebackground='#3c3c3c',
                              font=("Arial", 11))
        res2x.pack(anchor='w', padx=10, pady=5)
        
        # Info label
        info_label = tk.Label(res_frame, text="2x resolution requires regeneration and takes longer", 
                             fg='#888888', bg='#3c3c3c', font=("Arial", 9))
        info_label.pack(anchor='w', padx=10, pady=2)
        
        # Additional info
        time_label = tk.Label(res_frame, text="Estimated time: 30-60 seconds for complex fractals", 
                             fg='#888888', bg='#3c3c3c', font=("Arial", 8))
        time_label.pack(anchor='w', padx=10, pady=1)
        
        # Button frame
        btn_frame = tk.Frame(save_dialog, bg='#2c2c2c')
        btn_frame.pack(fill='x', padx=20, pady=15)
        
        def do_save():
            save_dialog.destroy()
            resolution = resolution_choice.get()
            self._perform_save(resolution)
        
        def cancel_save():
            save_dialog.destroy()
        
        # Save button
        save_btn = tk.Button(btn_frame, text="💾 Save", command=do_save,
                            bg='#0d7377', fg='white', font=("Arial", 11, "bold"),
                            relief='raised', bd=2, padx=20, pady=5)
        save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cancel button
        cancel_btn = tk.Button(btn_frame, text="❌ Cancel", command=cancel_save,
                              bg='#c44536', fg='white', font=("Arial", 11, "bold"),
                              relief='raised', bd=2, padx=20, pady=5)
        cancel_btn.pack(side=tk.RIGHT)
        
    def _perform_save(self, resolution):
        """Perform the actual save operation"""
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_suffix = "_2x" if resolution == "2x" else ""
        default_name = f"biomorph_{timestamp}{res_suffix}.png"
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Save Biomorph Fractal",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        try:
            if resolution == "1x":
                # Save current fractal
                if self.fractal_image:
                    save_image = self.fractal_image.copy()
                    if save_image.mode == 'L':
                        save_image = save_image.convert('RGB')
                    save_image.save(filepath)
                messagebox.showinfo("Success", f"Fractal saved as: {filepath}")
                
            elif resolution == "2x":
                # Generate high-resolution version
                self._save_high_resolution(filepath)
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not save fractal: {str(e)}")
            
    def _save_high_resolution(self, filepath):
        """Generate and save a 2x resolution version"""
        # Show progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Generating High Resolution")
        progress_dialog.geometry("400x150")
        progress_dialog.configure(bg='#2c2c2c')
        progress_dialog.transient(self.root)
        progress_dialog.grab_set()
        
        # Center the dialog
        progress_dialog.update_idletasks()
        x = (progress_dialog.winfo_screenwidth() // 2) - (progress_dialog.winfo_width() // 2)
        y = (progress_dialog.winfo_screenheight() // 2) - (progress_dialog.winfo_height() // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Progress info
        progress_label = tk.Label(progress_dialog, text="🎯 Generating 2400x1600 high-resolution image...", 
                                 font=("Arial", 12, "bold"), fg='white', bg='#2c2c2c')
        progress_label.pack(pady=20)
        
        progress_bar_hires = ttk.Progressbar(progress_dialog, length=300, mode='determinate')
        progress_bar_hires.pack(pady=10)
        
        status_label = tk.Label(progress_dialog, text="Starting...", 
                               fg='#888888', bg='#2c2c2c', font=("Arial", 10))
        status_label.pack(pady=5)
        
        def generate_hires():
            try:
                def update_hires_progress(progress):
                    progress_bar_hires['value'] = progress
                    status_label.config(text=f"Processing... {progress:.1f}%")
                    progress_dialog.update()
                
                # Get current parameters
                const_real = self.const_real.get()
                const_imag = self.const_imag.get()
                zoom = self.zoom.get()
                center_x = self.center_x.get()
                center_y = self.center_y.get()
                max_iter = self.max_iterations.get()
                escape_radius = self.escape_radius.get()
                
                # Generate 2x resolution fractal (2400x1600) without cancellation checks
                hires_data = self._generate_biomorph_uncancellable(
                    2400, 1600, const_real, const_imag, zoom, center_x, center_y,
                    max_iter, escape_radius, update_hires_progress)
                
                if hires_data is not None:
                    status_label.config(text="Converting to color image...")
                    progress_dialog.update()
                    
                    # Apply same color processing as normal resolution
                    if self.use_color_palette.get():
                        palette_size = 256
                        palette = self.generate_color_palette(self.color_palette.get(), palette_size)
                        palette_array = np.array(palette, dtype=np.uint8)
                        
                        rgb_data = np.zeros((1600, 2400, 3), dtype=np.uint8)
                        
                        # Handle biomorph structure
                        biomorph_mask = (hires_data == 0)
                        if self.invert_colors.get():
                            rgb_data[biomorph_mask] = [255, 255, 255]
                        else:
                            rgb_data[biomorph_mask] = [0, 0, 0]
                        
                        # Handle escaped points with same aggressive coloring
                        escaped_mask = (hires_data > 0)
                        if np.any(escaped_mask):
                            import math  # Import math for trigonometric functions
                            escaped_iterations = hires_data[escaped_mask]
                            
                            # Apply same ultra-aggressive color mapping
                            stretched_values = np.clip(escaped_iterations, 1, 10)
                            stretched_indices = ((stretched_values - 1) * palette_size * 0.8 / 9).astype(int)
                            
                            cycle1 = ((escaped_iterations * 47) % palette_size)
                            cycle2 = ((escaped_iterations * 73) % palette_size)
                            cycle3 = ((escaped_iterations * 31) % palette_size)
                            multi_cycle = (cycle1 + cycle2 + cycle3) // 3
                            
                            # Get 2D coordinates for high-res version too
                            y_coords_hires, x_coords_hires = np.where(escaped_mask)
                            
                            # Use same non-linear spatial waves for high-res version
                            norm_x_hires = x_coords_hires / 2400  # High-res width
                            norm_y_hires = y_coords_hires / 1600  # High-res height
                            
                            wave1_hires = np.sin(norm_x_hires * 2 * math.pi * 3.7 + norm_y_hires * 2 * math.pi * 2.3) * 0.5 + 0.5
                            wave2_hires = np.cos(norm_x_hires * 2 * math.pi * 1.9 + norm_y_hires * 2 * math.pi * 4.1) * 0.5 + 0.5
                            wave3_hires = np.sin(norm_x_hires * 2 * math.pi * 5.1 - norm_y_hires * 2 * math.pi * 1.7) * 0.5 + 0.5
                            
                            spatial_blend = ((wave1_hires + wave2_hires + wave3_hires) / 3 * palette_size).astype(int) % palette_size
                            
                            fractal_indices = ((escaped_iterations * escaped_iterations * 13) % palette_size)
                            coord_hash_hires = (x_coords_hires * 73 + y_coords_hires * 101) % 256
                            smooth_offset = coord_hash_hires / 256.0 * 5
                            smooth_indices = ((escaped_iterations - 1 + smooth_offset) * palette_size / 10).astype(int) % palette_size
                            
                            final_indices = (
                                stretched_indices * 0.4 +
                                multi_cycle * 0.25 +
                                spatial_blend * 0.15 +
                                fractal_indices * 0.1 +
                                smooth_indices * 0.1
                            ).astype(int) % palette_size
                            
                            colors = palette_array[final_indices]
                            
                            if self.invert_colors.get():
                                colors = 255 - colors
                            
                            rgb_data[escaped_mask] = colors
                        
                        hires_image = Image.fromarray(rgb_data)
                    else:
                        # Grayscale version
                        gray_data = np.zeros((1600, 2400), dtype=np.uint8)
                        gray_data[hires_data == 0] = 0 if not self.invert_colors.get() else 255
                        gray_data[hires_data > 0] = 255 if not self.invert_colors.get() else 0
                        hires_image = Image.fromarray(gray_data)
                    
                    # Save the image
                    status_label.config(text="Saving high-resolution image...")
                    progress_dialog.update()
                    
                    if hires_image.mode == 'L':
                        hires_image = hires_image.convert('RGB')
                    
                    hires_image.save(filepath)
                    progress_dialog.destroy()
                    messagebox.showinfo("Success", f"High-resolution fractal saved as: {filepath}\nResolution: 2400x1600 pixels")
                    
                else:
                    progress_dialog.destroy()
                    messagebox.showwarning("Cancelled", "High-resolution generation was cancelled")
                    
            except Exception as e:
                progress_dialog.destroy()
                messagebox.showerror("Error", f"Could not generate high-resolution fractal: {str(e)}")
        
        # Start generation in thread
        threading.Thread(target=generate_hires, daemon=True).start()
    
    def _generate_biomorph_uncancellable(self, width, height, const_real, const_imag, zoom, center_x, center_y, 
                                        max_iter, escape_radius, progress_callback=None):
        """Generate the biomorph fractal without cancellation checks (for high-res saves)"""
        
        # Calculate coordinate bounds with proper aspect ratio
        aspect_ratio = width / height
        ymax_val = zoom  # Use zoom as the y-range
        ymin_val = -ymax_val
        # Adjust x-range based on aspect ratio
        xmax_val = ymax_val * aspect_ratio  
        xmin_val = -xmax_val
        
        # Apply center offset
        xmin = center_x + xmin_val
        xmax = center_x + xmax_val
        ymin = center_y + ymin_val
        ymax = center_y + ymax_val
        
        # Create result array for iteration counts
        result = np.zeros((height, width), dtype=np.int32)
        
        for i in range(height):
            if progress_callback and i % 10 == 0:
                progress = (i / height) * 100
                progress_callback(progress)
                
            # No cancellation check - always complete the generation
            
            for j in range(width):
                # Map pixel to complex plane (matching BASIC: x0 = xmin + (xmax - xmin) * j / jlimit)
                x0 = xmin + (xmax - xmin) * j / (width - 1)
                y0 = -ymin - (ymax - ymin) * i / (height - 1)  # Match BASIC: y0 = -ymin - (ymax - ymin) * i / ilimit
                
                x, y = x0, y0
                
                # Iterate the biomorph equation (matching original BASIC)
                iteration_count = 0
                for n in range(1, max_iter + 1):
                    # Compute z^3 + c (the core biomorph equation)
                    xx = x * (x * x - 3 * y * y) + const_real
                    yy = y * (3 * x * x - y * y) + const_imag
                    
                    x, y = xx, yy
                    iteration_count = n
                    
                    # Check escape condition (matching original: ABS(x) > 10 OR ABS(y) > 10 OR x*x + y*y > 10^2)
                    if abs(x) > escape_radius or abs(y) > escape_radius or x*x + y*y > escape_radius*escape_radius:
                        break
                        
                # Store iteration count (0 means it never escaped - the biomorph itself)
                if abs(x) < escape_radius and abs(y) < escape_radius:
                    result[i, j] = 0  # Stable point - the biomorph structure
                else:
                    result[i, j] = iteration_count  # Escaped - store iteration count
                    
        return result
    
    def generate_color_palette(self, palette_name, num_colors):
        """Generate a smooth color palette"""
        colors = []
        
        if palette_name == "vibrant":
            # Vibrant cycling palette: Designed for multiple cycles with high contrast
            cycle_colors = [
                (255, 0, 100),    # Hot Pink
                (0, 255, 200),    # Cyan
                (255, 150, 0),    # Orange
                (100, 0, 255),    # Purple
                (255, 255, 0),    # Yellow
                (0, 255, 0),      # Green
                (255, 0, 0),      # Red
                (0, 150, 255),    # Sky Blue
            ]
            
            # Interpolate between cycle colors
            for i in range(num_colors):
                # Map to position in cycle
                cycle_pos = (i / num_colors) * len(cycle_colors)
                idx1 = int(cycle_pos) % len(cycle_colors)
                idx2 = (idx1 + 1) % len(cycle_colors)
                t = cycle_pos - int(cycle_pos)
                
                # Interpolate between colors
                c1 = cycle_colors[idx1]
                c2 = cycle_colors[idx2]
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
                colors.append((r, g, b))
                
        elif palette_name == "rainbow":
            # Rainbow: Red -> Orange -> Yellow -> Green -> Blue -> Purple
            for i in range(num_colors):
                hue = i / num_colors * 300  # 0 to 300 degrees (avoiding red repeat)
                sat = 1.0
                val = 1.0
                r, g, b = self.hsv_to_rgb(hue/360, sat, val)
                colors.append((int(r*255), int(g*255), int(b*255)))
                
        elif palette_name == "fire":
            # Fire: Black -> Red -> Orange -> Yellow -> White
            for i in range(num_colors):
                t = i / (num_colors - 1)
                if t < 0.25:
                    # Black to Red
                    r = int(255 * (t * 4))
                    g = 0
                    b = 0
                elif t < 0.5:
                    # Red to Orange
                    r = 255
                    g = int(255 * ((t - 0.25) * 4))
                    b = 0
                elif t < 0.75:
                    # Orange to Yellow
                    r = 255
                    g = 255
                    b = int(255 * ((t - 0.5) * 4))
                else:
                    # Yellow to White
                    r = 255
                    g = 255
                    b = 255
                colors.append((r, g, b))
                
        elif palette_name == "ocean":
            # Ocean: Dark Blue -> Light Blue -> Cyan -> White
            for i in range(num_colors):
                t = i / (num_colors - 1)
                r = int(255 * t * 0.3)
                g = int(255 * (0.3 + t * 0.7))
                b = int(255 * (0.5 + t * 0.5))
                colors.append((r, g, b))
                
        elif palette_name == "plasma":
            # Enhanced Plasma: Deep Purple -> Magenta -> Orange -> Bright Yellow
            for i in range(num_colors):
                t = i / (num_colors - 1)
                if t < 0.2:
                    # Deep Purple to Magenta
                    r = int(255 * (0.1 + t * 4))
                    g = int(255 * t * 0.5)
                    b = int(255 * (0.8 + t))
                elif t < 0.5:
                    # Magenta to Red-Orange
                    tt = (t - 0.2) / 0.3
                    r = 255
                    g = int(255 * (0.1 + tt * 0.7))
                    b = int(255 * (0.9 - tt * 0.9))
                elif t < 0.8:
                    # Red-Orange to Orange
                    tt = (t - 0.5) / 0.3
                    r = 255
                    g = int(255 * (0.8 + tt * 0.2))
                    b = int(255 * tt * 0.3)
                else:
                    # Orange to Bright Yellow
                    tt = (t - 0.8) / 0.2
                    r = 255
                    g = 255
                    b = int(255 * tt * 0.8)
                colors.append((r, g, b))
                
        elif palette_name == "sunset":
            # Sunset: Dark Purple -> Orange -> Yellow -> Light Blue
            for i in range(num_colors):
                t = i / (num_colors - 1)
                if t < 0.3:
                    # Dark Purple to Orange
                    r = int(255 * (0.3 + t * 2.3))
                    g = int(255 * t * 2)
                    b = int(255 * (0.5 - t))
                elif t < 0.7:
                    # Orange to Yellow
                    r = 255
                    g = int(255 * (0.6 + (t - 0.3) * 1))
                    b = 0
                else:
                    # Yellow to Light Blue
                    r = int(255 * (1 - (t - 0.7) * 2))
                    g = int(255 * (1 - (t - 0.7) * 0.5))
                    b = int(255 * ((t - 0.7) * 3))
                colors.append((r, g, b))
                
        elif palette_name == "forest":
            # Forest: Dark Green -> Light Green -> Yellow -> Orange
            for i in range(num_colors):
                t = i / (num_colors - 1)
                if t < 0.5:
                    # Dark Green to Light Green
                    r = int(255 * t * 0.2)
                    g = int(255 * (0.3 + t * 0.7))
                    b = int(255 * t * 0.1)
                else:
                    # Light Green to Yellow to Orange
                    r = int(255 * ((t - 0.5) * 2))
                    g = 255
                    b = 0
                colors.append((r, g, b))
                
        return colors
        
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        import math
        
        if s == 0:
            return v, v, v
            
        h *= 6.0
        i = int(h)
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q

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
        error_msg += "Or use the launcher script which will install automatically."
        
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
    app = BiomorphGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
