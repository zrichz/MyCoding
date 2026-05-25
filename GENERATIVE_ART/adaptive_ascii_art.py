"""
Adaptive ASCII Art Generator - Creates ASCII art with variable character sizes based on luminance variance
Uses adaptive subdivision to create smaller chunks in high-variance areas and larger chunks in uniform areas
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np
import os
import math

class AdaptiveAsciiArt:
    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive ASCII Art Generator")
        self.root.geometry("1800x950")  # Larger window for enhanced controls
        self.root.configure(bg='#808080')  # Mid-grey background
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Variables
        self.input_image = None
        self.ascii_image = None
        
        # Individual thresholds for different chunk sizes
        self.threshold_64 = 1000
        self.threshold_32 = 750
        self.threshold_16 = 500
        
        # Background color toggle
        self.show_background_colors = True
        
        # Font scaling factor
        self.font_scale_factor = 1.0
        
        # Text color (True = white, False = black)
        self.white_text = False
        
        # Show ASCII text toggle
        self.show_text = True
        
        # ASCII character mapping presets (ordered from darkest to lightest)
        self.ascii_presets = {
            "Dense (Extended)": " .'^`,:-;~-=<>+*oahkbdpqRPGUOA$QHMWBb8%#&@",
            "Classic": " .:=-+*#%@",
            "Simple": " ▏▎▍▌▋▊▉█",
            "Dots": " ○◕◔◓◒◑◐●"
        }
        
        # Current ASCII mapping
        self.ascii_chars = self.ascii_presets["Dense (Extended)"]
        self.current_preset = "Dense (Extended)"
        
        # Custom ASCII mapping
        self.custom_ascii = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=15)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill='x', pady=5)
        
        ttk.Label(file_frame, text="Select Image (will resize to 768x768):", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar(value="No image selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var, foreground="gray")
        file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(file_frame, text="Browse", command=self.select_image).pack(side=tk.RIGHT)
        
        # Individual threshold controls
        thresholds_frame = ttk.LabelFrame(control_frame, text="Variance Thresholds by Chunk Size", padding=15)
        thresholds_frame.pack(fill='x', pady=10)
        
        # 64x64 threshold
        threshold_64_frame = ttk.Frame(thresholds_frame)
        threshold_64_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_64_frame, text="64×64px Threshold:", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        self.threshold_64_var = tk.IntVar(value=1000)
        threshold_64_scale = ttk.Scale(threshold_64_frame, from_=0, to=5000, variable=self.threshold_64_var, 
                                      orient='horizontal', length=600)  # Much wider slider
        threshold_64_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.threshold_64_label = ttk.Label(threshold_64_frame, text="1000", width=6)
        self.threshold_64_label.pack(side=tk.RIGHT, padx=5)
        
        # 32x32 threshold
        threshold_32_frame = ttk.Frame(thresholds_frame)
        threshold_32_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_32_frame, text="32×32px Threshold:", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        self.threshold_32_var = tk.IntVar(value=750)
        threshold_32_scale = ttk.Scale(threshold_32_frame, from_=0, to=3000, variable=self.threshold_32_var, 
                                      orient='horizontal', length=600)
        threshold_32_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.threshold_32_label = ttk.Label(threshold_32_frame, text="750", width=6)
        self.threshold_32_label.pack(side=tk.RIGHT, padx=5)
        
        # 16x16 threshold
        threshold_16_frame = ttk.Frame(thresholds_frame)
        threshold_16_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_16_frame, text="16×16px Threshold:", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        self.threshold_16_var = tk.IntVar(value=500)
        threshold_16_scale = ttk.Scale(threshold_16_frame, from_=0, to=2000, variable=self.threshold_16_var, 
                                      orient='horizontal', length=600)
        threshold_16_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.threshold_16_label = ttk.Label(threshold_16_frame, text="500", width=6)
        self.threshold_16_label.pack(side=tk.RIGHT, padx=5)
        
        # Update threshold labels when scales change
        threshold_64_scale.configure(command=self.update_threshold_64_label)
        threshold_32_scale.configure(command=self.update_threshold_32_label)
        threshold_16_scale.configure(command=self.update_threshold_16_label)
        
        # Background color toggle
        background_frame = ttk.Frame(control_frame)
        background_frame.pack(fill='x', pady=10)
        
        self.show_background_var = tk.BooleanVar(value=True)
        background_checkbox = ttk.Checkbutton(background_frame, 
                                             text="Show colored background rectangles", 
                                             variable=self.show_background_var,
                                             command=self.update_background_setting)
        background_checkbox.pack(side=tk.LEFT)
        
        # Text color toggle
        text_color_frame = ttk.Frame(control_frame)
        text_color_frame.pack(fill='x', pady=5)
        
        self.white_text_var = tk.BooleanVar(value=False)
        text_color_checkbox = ttk.Checkbutton(text_color_frame, 
                                             text="Use white text (instead of black)", 
                                             variable=self.white_text_var,
                                             command=self.update_text_color_setting)
        text_color_checkbox.pack(side=tk.LEFT)
        
        # Show text toggle
        show_text_frame = ttk.Frame(control_frame)
        show_text_frame.pack(fill='x', pady=5)
        
        self.show_text_var = tk.BooleanVar(value=True)
        show_text_checkbox = ttk.Checkbutton(show_text_frame, 
                                            text="Show ASCII text (uncheck for solid rectangles only)", 
                                            variable=self.show_text_var,
                                            command=self.update_show_text_setting)
        show_text_checkbox.pack(side=tk.LEFT)
        
        # Font scaling control
        font_scale_frame = ttk.Frame(control_frame)
        font_scale_frame.pack(fill='x', pady=10)
        
        ttk.Label(font_scale_frame, text="Font Scale Factor:", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        self.font_scale_var = tk.DoubleVar(value=1.0)
        font_scale_scale = ttk.Scale(font_scale_frame, from_=0.5, to=3.0, variable=self.font_scale_var, 
                                    orient='horizontal', length=400)
        font_scale_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.font_scale_label = ttk.Label(font_scale_frame, text="1.0", width=6)
        self.font_scale_label.pack(side=tk.RIGHT, padx=5)
        
        # Update font scale label when scale changes
        font_scale_scale.configure(command=self.update_font_scale_label)
        
        # ASCII mapping selection
        ascii_frame = ttk.LabelFrame(control_frame, text="ASCII Character Mapping", padding=15)
        ascii_frame.pack(fill='x', pady=10)
        
        # Preset selection
        preset_frame = ttk.Frame(ascii_frame)
        preset_frame.pack(fill='x', pady=5)
        
        ttk.Label(preset_frame, text="Preset:", font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT)
        self.ascii_preset_var = tk.StringVar(value="Dense (Extended)")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.ascii_preset_var, 
                                   values=list(self.ascii_presets.keys()), 
                                   state="readonly", width=20)
        preset_combo.pack(side=tk.LEFT, padx=10)
        preset_combo.bind('<<ComboboxSelected>>', self.update_ascii_preset)
        
        # Custom mapping input
        custom_frame = ttk.Frame(ascii_frame)
        custom_frame.pack(fill='x', pady=5)
        
        ttk.Label(custom_frame, text="Custom:", font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT)
        self.custom_ascii_var = tk.StringVar()
        custom_entry = ttk.Entry(custom_frame, textvariable=self.custom_ascii_var, width=40)
        custom_entry.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        ttk.Button(custom_frame, text="Apply Custom", 
                  command=self.apply_custom_ascii, width=12).pack(side=tk.RIGHT, padx=5)
        
        # Current mapping display
        current_frame = ttk.Frame(ascii_frame)
        current_frame.pack(fill='x', pady=5)
        
        ttk.Label(current_frame, text="Current:", font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT)
        self.current_ascii_var = tk.StringVar(value=self.ascii_chars)
        current_label = ttk.Label(current_frame, textvariable=self.current_ascii_var, 
                                 foreground="blue", font=("Courier", 9))
        current_label.pack(side=tk.LEFT, padx=10)
        
        # Generate button with enhanced styling
        style = ttk.Style()
        style.configure('Generate.TButton', 
                       background='orange', 
                       foreground='black',
                       font=('Arial', 10, 'bold'))
        
        self.generate_btn = ttk.Button(control_frame, text="🎨 Generate ASCII Art", 
                                      command=self.generate_ascii, state='disabled',
                                      style='Generate.TButton')
        self.generate_btn.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Select an image to begin")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var)
        progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(control_frame, length=300, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        
        # Image display area
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill='both', expand=True)
        
        # Original image panel
        original_frame = ttk.LabelFrame(image_frame, text="Original Image (768x768)", padding=10)
        original_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, width=768, height=768, bg='white')
        self.original_canvas.pack()
        
        # ASCII art panel
        ascii_frame = ttk.LabelFrame(image_frame, text="Adaptive ASCII Art", padding=10)
        ascii_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=(5, 0))
        
        # Create canvas with scrollbars for ASCII art
        canvas_frame = ttk.Frame(ascii_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        self.ascii_canvas = tk.Canvas(canvas_frame, width=768, height=768, bg='white')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.ascii_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal', command=self.ascii_canvas.xview)
        
        self.ascii_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill='y')
        h_scrollbar.pack(side=tk.BOTTOM, fill='x')
        self.ascii_canvas.pack(side=tk.LEFT, fill='both', expand=True)
        
    def update_threshold_64_label(self, value):
        """Update 64x64 threshold label when scale changes"""
        self.threshold_64_label.config(text=str(int(float(value))))
        self.threshold_64 = int(float(value))
        
    def update_threshold_32_label(self, value):
        """Update 32x32 threshold label when scale changes"""
        self.threshold_32_label.config(text=str(int(float(value))))
        self.threshold_32 = int(float(value))
        
    def update_threshold_16_label(self, value):
        """Update 16x16 threshold label when scale changes"""
        self.threshold_16_label.config(text=str(int(float(value))))
        self.threshold_16 = int(float(value))
        
    def update_background_setting(self):
        """Update background color setting"""
        self.show_background_colors = self.show_background_var.get()
        
    def update_text_color_setting(self):
        """Update text color setting"""
        self.white_text = self.white_text_var.get()
        
    def update_show_text_setting(self):
        """Update show text setting"""
        self.show_text = self.show_text_var.get()
        
    def update_font_scale_label(self, value):
        """Update font scale label when scale changes"""
        self.font_scale_label.config(text=f"{float(value):.1f}")
        self.font_scale_factor = float(value)
        
    def update_ascii_preset(self, event=None):
        """Update ASCII characters when preset is selected"""
        selected = self.ascii_preset_var.get()
        if selected in self.ascii_presets:
            self.ascii_chars = self.ascii_presets[selected]
            self.current_preset = selected
            self.current_ascii_var.set(self.ascii_chars)
            
    def apply_custom_ascii(self):
        """Apply custom ASCII character mapping"""
        custom = self.custom_ascii_var.get()  # Don't strip - preserve leading/trailing spaces
        if custom:
            if len(custom) < 2:
                messagebox.showwarning("Invalid Mapping", "Custom mapping must have at least 2 characters")
                return
            self.ascii_chars = custom
            self.current_preset = "Custom"
            self.ascii_preset_var.set("Custom")
            self.current_ascii_var.set(custom)
            messagebox.showinfo("Success", f"Applied custom mapping with {len(custom)} characters")
        else:
            messagebox.showwarning("Empty Mapping", "Please enter a custom character mapping")
        
    def select_image(self):
        """Select input image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select 512x512 Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Load and validate image
                img = Image.open(filepath)
                
                # Always resize to 768x768 for consistency
                original_size = img.size
                if img.size != (768, 768):
                    img = img.resize((768, 768), Image.Resampling.LANCZOS)
                    if original_size != (768, 768):
                        messagebox.showinfo("Image Resized", 
                                          f"Image resized from {original_size} to 768x768")
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                self.input_image = img
                
                # Display original image
                self.display_original_image()
                
                # Update UI
                filename = os.path.basename(filepath)
                self.file_path_var.set(filename)
                self.generate_btn.configure(state='normal')
                self.progress_var.set("Image loaded - ready to generate ASCII art")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                
    def display_original_image(self):
        """Display the original image on canvas"""
        if self.input_image:
            # Create PhotoImage for display
            photo = ImageTk.PhotoImage(self.input_image)
            
            # Clear canvas and display image
            self.original_canvas.delete("all")
            self.original_canvas.create_image(384, 384, image=photo)
            
            # Keep a reference to prevent garbage collection
            self.original_photo_ref = photo
            
    def calculate_luma_variance(self, img_array):
        """Calculate luminance variance for an image chunk"""
        # Convert RGB to luminance using standard weights
        luma = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        return np.var(luma)
        
    def calculate_average_luma(self, img_array):
        """Calculate average luminance for an image chunk"""
        luma = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        return np.mean(luma)
        
    def calculate_average_color(self, img_array):
        """Calculate average RGB color for an image chunk"""
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        return (int(avg_r), int(avg_g), int(avg_b))
        
    def luma_to_ascii(self, luma, min_luma=None, max_luma=None):
        """Convert luminance value to ASCII character with dynamic range mapping"""
        if min_luma is None or max_luma is None:
            # Fallback to basic mapping if range not provided
            index = min(int((luma / 255.0) * len(self.ascii_chars)), len(self.ascii_chars) - 1)
        else:
            # Dynamic range mapping to ensure full character usage
            if max_luma == min_luma:
                # Avoid division by zero
                normalized_luma = 0.5
            else:
                normalized_luma = (luma - min_luma) / (max_luma - min_luma)
            
            # Clamp to valid range and map to character index
            normalized_luma = max(0.0, min(1.0, normalized_luma))
            index = min(int(normalized_luma * len(self.ascii_chars)), len(self.ascii_chars) - 1)
        
        # Adjust mapping based on text color setting
        # Our strings are ordered light to dark: " .'^`,:-;~-=<>+*oahkbdpqRPGUOA$QHMWBb8%#&@"
        if self.white_text:
            # White text: dark luma (index high) should use dark chars (end of string)
            return self.ascii_chars[index] 
        else:
            # Black text: dark luma (index high) should use light chars (start of string) for contrast
            return self.ascii_chars[-(index + 1)]
        
    def subdivide_chunk(self, img_array, x, y, size, threshold, chunks):
        """Recursively subdivide image chunks based on variance threshold"""
        variance = self.calculate_luma_variance(img_array)
        
        # Determine appropriate threshold based on current chunk size
        # Only allow subdivision if we can get to the next valid size: 64→32→16→8
        if size == 64:
            current_threshold = self.threshold_64
            next_size = 32
        elif size == 32:
            current_threshold = self.threshold_32
            next_size = 16
        elif size == 16:
            current_threshold = self.threshold_16
            next_size = 8
        else:
            # For any other size (should be 8 or smaller), don't subdivide further
            current_threshold = 0  # Force stop subdivision
            next_size = 0
        
        # If variance is below threshold or minimum size reached, add to chunks
        if variance <= current_threshold or size <= 8:
            avg_luma = self.calculate_average_luma(img_array)
            avg_color = self.calculate_average_color(img_array)
            # Store luma for later dynamic range mapping - don't assign char yet
            chunks.append({
                'x': x,  # Top-left position for rectangle
                'y': y,
                'center_x': x + size // 2,  # Center position for text
                'center_y': y + size // 2,
                'size': size,
                'luma': avg_luma,  # Store for dynamic range mapping
                'char': None,  # Will be assigned after range analysis
                'variance': variance,
                'color': avg_color
            })
            return
            
        half_size = next_size # Otherwise, subdivide into 4 quadrants of the next valid size
        
        # Top-left, top-right, bottom-left, bottom-right
        self.subdivide_chunk(img_array[:half_size, :half_size], x, y, half_size, current_threshold, chunks)
        self.subdivide_chunk(img_array[:half_size, half_size:], x + half_size, y, half_size, current_threshold, chunks)
        self.subdivide_chunk(img_array[half_size:, :half_size], x, y + half_size, half_size, current_threshold, chunks)
        self.subdivide_chunk(img_array[half_size:, half_size:], x + half_size, y + half_size, half_size, current_threshold, chunks)
        
    def generate_ascii(self):
        """Generate adaptive ASCII art"""
        if not self.input_image:
            return
            
        try:
            # Show progress
            self.progress_var.set("Generating ASCII art...")
            self.progress_bar.start()
            self.root.update()
            
            # Convert image to numpy array
            img_array = np.array(self.input_image)
            
            # Start subdivision process - divide 768x768 image into 64x64 chunks first
            chunks = []
            # Divide the 768x768 image into 12x12 grid of 64x64 chunks
            for row in range(0, 768, 64):
                for col in range(0, 768, 64):
                    # Extract 64x64 chunk
                    chunk_array = img_array[row:row+64, col:col+64]
                    # Process this 64x64 chunk through subdivision
                    self.subdivide_chunk(chunk_array, col, row, 64, self.threshold_64, chunks)
            
            # Apply dynamic range mapping to ensure full character usage
            if chunks:
                # Collect all luma values
                luma_values = [chunk['luma'] for chunk in chunks]
                min_luma = min(luma_values)
                max_luma = max(luma_values)
                
                # Assign ASCII characters using dynamic range mapping
                for chunk in chunks:
                    chunk['char'] = self.luma_to_ascii(chunk['luma'], min_luma, max_luma)
            
            # Create ASCII art image (PIL automatically handles pixel-perfect rendering for bitmap fonts)
            ascii_img = Image.new('RGB', (768, 768), 'white')
            draw = ImageDraw.Draw(ascii_img)
            
            # Try to find IBM EGA font for authentic ASCII art - check multiple locations
            ibm_ega_locations = [
                "C:/Windows/Fonts/Ac437_IBM_EGA_8x8.ttf",  # System fonts
                os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts/Ac437_IBM_EGA_8x8.ttf"),  # User fonts
                "C:/Windows/System32/Fonts/Ac437_IBM_EGA_8x8.ttf"  # Alternative system location
            ]
            
            # Fallback fonts if IBM EGA is not available
            fallback_fonts = [
                "C:/Windows/Fonts/consola.ttf",  # Consolas (monospace)
                "C:/Windows/Fonts/cour.ttf",     # Courier New
                "C:/Windows/Fonts/lucon.ttf",    # Lucida Console
                "C:/Windows/Fonts/arial.ttf"     # Arial (not monospace but widely available)
            ]
            
            base_font_path = None
            font_name = "Unknown"
            
            # First try to find IBM EGA font
            for font_path in ibm_ega_locations:
                if os.path.exists(font_path):
                    base_font_path = font_path
                    font_name = "IBM EGA 8x8"
                    break
            
            # If IBM EGA not found, try fallback fonts
            if not base_font_path:
                for font_path in fallback_fonts:
                    if os.path.exists(font_path):
                        base_font_path = font_path
                        if "consola" in font_path:
                            font_name = "Consolas"
                        elif "cour" in font_path:
                            font_name = "Courier New"
                        elif "lucon" in font_path:
                            font_name = "Lucida Console"
                        elif "arial" in font_path:
                            font_name = "Arial"
                        break
            
            # If no fonts found at all, use PIL's default font
            if not base_font_path:
                font_name = "PIL Default"
                print(f"Warning: Using PIL default font - ASCII art quality may be reduced")
            else:
                print(f"Using font: {font_name} ({base_font_path})")
            
            # Draw chunks with optional colored backgrounds and ASCII characters
            for chunk in chunks:
                # Draw background rectangle with appropriate colors
                rect_coords = [
                    chunk['x'], chunk['y'],  # Top-left
                    chunk['x'] + chunk['size'], chunk['y'] + chunk['size']  # Bottom-right
                ]
                
                # Handle background color based on settings
                if self.show_background_colors:
                    draw.rectangle(rect_coords, fill=chunk['color'])
                    text_color = 'white' if self.white_text else 'black'
                else:
                    # If no colored background, use appropriate contrast
                    if self.white_text:
                        draw.rectangle(rect_coords, fill='black')  # Black background for white text
                        text_color = 'white'
                    else:
                        draw.rectangle(rect_coords, fill='white')  # White background for black text
                        text_color = 'black'
                
                # Skip text drawing if show_text is disabled
                if not self.show_text:
                    continue
                
                # Calculate font size with pixel-perfect integer scaling for retro aesthetic
                # Base EGA font is 8x8, scale in integer multiples: 1x, 2x, 4x, 8x
                if chunk['size'] == 8:
                    base_multiplier = 1  # 8px font
                elif chunk['size'] == 16:
                    base_multiplier = 2  # 16px font (2x scale)
                elif chunk['size'] == 32:
                    base_multiplier = 4  # 32px font (4x scale)
                elif chunk['size'] == 64:
                    base_multiplier = 8  # 64px font (8x scale)
                else:
                    # Fallback for any unexpected sizes
                    base_multiplier = max(1, chunk['size'] // 8)
                
                # Apply user scaling factor as integer multiplier
                final_multiplier = max(1, int(base_multiplier * self.font_scale_factor))
                font_size = 8 * final_multiplier  # EGA base size (8) × multiplier
                
                # Ensure font size is reasonable (but keep integer multiples)
                font_size = max(8, min(128, font_size))  # Minimum 8px (1x EGA), max 128px (16x EGA)
                
                # Debug: Print font sizes being used (remove this in production)
                # print(f"Chunk {chunk['size']}px -> Base multiplier: {base_multiplier}x -> Final: {final_multiplier}x -> Font: {font_size}px")
                
                try:
                    # Load font with exact size if available, otherwise use default
                    if base_font_path:
                        font = ImageFont.truetype(base_font_path, font_size)
                    else:
                        font = ImageFont.load_default()
                except Exception as e:
                    # If font loading fails, use default font
                    print(f"Warning: Failed to load font at size {font_size}: {str(e)}")
                    font = ImageFont.load_default()
                
                # Draw character at chunk center with chosen text color
                draw.text(
                    (chunk['center_x'], chunk['center_y']), 
                    chunk['char'], 
                    font=font, 
                    fill=text_color,
                    anchor='mm'  # Center the text
                )
            
            # Display ASCII art
            self.ascii_image = ascii_img
            self.display_ascii_image()
            
            # Update progress with more details
            self.progress_bar.stop()
            
            # Calculate chunk size distribution for info
            size_counts = {}
            for chunk in chunks:
                size = chunk['size']
                size_counts[size] = size_counts.get(size, 0) + 1
            
            size_info = ", ".join([f"{count}×{size}px" for size, count in sorted(size_counts.items(), reverse=True)])
            
            # Show dynamic range and font info
            luma_values = [chunk['luma'] for chunk in chunks]
            min_luma = min(luma_values)
            max_luma = max(luma_values)
            
            # IBM EGA font is always used (we error if not available)
            self.progress_var.set(f"Generated {len(chunks)} chunks: {size_info} | Luma: {min_luma:.0f}-{max_luma:.0f} | Font: IBM EGA")
            
        except Exception as e:
            self.progress_bar.stop()
            self.progress_var.set("Error generating ASCII art")
            messagebox.showerror("Error", f"Could not generate ASCII art: {str(e)}")
            
    def display_ascii_image(self):
        """Display ASCII art on canvas"""
        if self.ascii_image:
            # Create PhotoImage for display
            photo = ImageTk.PhotoImage(self.ascii_image)
            
            # Clear canvas and display image
            self.ascii_canvas.delete("all")
            self.ascii_canvas.create_image(384, 384, image=photo)
            
            # Configure scroll region
            self.ascii_canvas.configure(scrollregion=self.ascii_canvas.bbox("all"))
            
            # Keep a reference to prevent garbage collection
            self.ascii_photo_ref = photo

def main():
    # Check if required packages are available
    missing_packages = []
    
    try:
        import numpy as np
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        from PIL import Image, ImageTk, ImageFont, ImageDraw
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
    app = AdaptiveAsciiArt(root)
    root.mainloop()

if __name__ == "__main__":
    main()