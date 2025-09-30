#!/usr/bin/env python3
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
        
        # Extended ASCII character mapping (from darkest to lightest) - 20+ levels
        self.ascii_chars = "@&#%8BWMHQ$AOUGPRqpdbkhao*+<>=~-;:,'^`'. "
        
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
        
        ttk.Label(file_frame, text="Select 512x512 Image:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
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
        threshold_64_scale = ttk.Scale(threshold_64_frame, from_=100, to=5000, variable=self.threshold_64_var, 
                                      orient='horizontal', length=600)  # Much wider slider
        threshold_64_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.threshold_64_label = ttk.Label(threshold_64_frame, text="1000", width=6)
        self.threshold_64_label.pack(side=tk.RIGHT, padx=5)
        
        # 32x32 threshold
        threshold_32_frame = ttk.Frame(thresholds_frame)
        threshold_32_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_32_frame, text="32×32px Threshold:", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        self.threshold_32_var = tk.IntVar(value=750)
        threshold_32_scale = ttk.Scale(threshold_32_frame, from_=50, to=3000, variable=self.threshold_32_var, 
                                      orient='horizontal', length=600)
        threshold_32_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.threshold_32_label = ttk.Label(threshold_32_frame, text="750", width=6)
        self.threshold_32_label.pack(side=tk.RIGHT, padx=5)
        
        # 16x16 threshold
        threshold_16_frame = ttk.Frame(thresholds_frame)
        threshold_16_frame.pack(fill='x', pady=5)
        
        ttk.Label(threshold_16_frame, text="16×16px Threshold:", font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        self.threshold_16_var = tk.IntVar(value=500)
        threshold_16_scale = ttk.Scale(threshold_16_frame, from_=25, to=2000, variable=self.threshold_16_var, 
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
        
        # Generate button
        self.generate_btn = ttk.Button(control_frame, text="🎨 Generate ASCII Art", 
                                      command=self.generate_ascii, state='disabled')
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
        original_frame = ttk.LabelFrame(image_frame, text="Original Image (512x512)", padding=10)
        original_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, width=512, height=512, bg='white')
        self.original_canvas.pack()
        
        # ASCII art panel
        ascii_frame = ttk.LabelFrame(image_frame, text="Adaptive ASCII Art", padding=10)
        ascii_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=(5, 0))
        
        # Create canvas with scrollbars for ASCII art
        canvas_frame = ttk.Frame(ascii_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        self.ascii_canvas = tk.Canvas(canvas_frame, width=512, height=512, bg='white')
        
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
        
    def update_font_scale_label(self, value):
        """Update font scale label when scale changes"""
        self.font_scale_label.config(text=f"{float(value):.1f}")
        self.font_scale_factor = float(value)
        
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
                
                # Resize to 512x512 if needed
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    messagebox.showinfo("Image Resized", 
                                      f"Image resized from {Image.open(filepath).size} to 512x512")
                
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
            self.original_canvas.create_image(256, 256, image=photo)
            
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
        
    def luma_to_ascii(self, luma):
        """Convert luminance value to ASCII character"""
        # Map luminance (0-255) to ASCII character index
        index = int((luma / 255.0) * (len(self.ascii_chars) - 1))
        return self.ascii_chars[-(index + 1)]  # Reverse for dark-to-light mapping
        
    def subdivide_chunk(self, img_array, x, y, size, threshold, chunks):
        """Recursively subdivide image chunks based on variance threshold"""
        variance = self.calculate_luma_variance(img_array)
        
        # Determine appropriate threshold based on current chunk size
        if size >= 64:
            current_threshold = self.threshold_64
        elif size >= 32:
            current_threshold = self.threshold_32
        elif size >= 16:
            current_threshold = self.threshold_16
        else:
            current_threshold = threshold  # Use passed threshold for smaller sizes
        
        # If variance is below threshold or minimum size reached, add to chunks
        if variance <= current_threshold or size <= 8:
            avg_luma = self.calculate_average_luma(img_array)
            avg_color = self.calculate_average_color(img_array)
            ascii_char = self.luma_to_ascii(avg_luma)
            chunks.append({
                'x': x,  # Top-left position for rectangle
                'y': y,
                'center_x': x + size // 2,  # Center position for text
                'center_y': y + size // 2,
                'size': size,
                'char': ascii_char,
                'variance': variance,
                'color': avg_color
            })
            return
            
        # Otherwise, subdivide into 4 quadrants
        half_size = size // 2
        new_threshold = threshold / 4
        
        # Top-left
        self.subdivide_chunk(
            img_array[:half_size, :half_size], 
            x, y, half_size, new_threshold, chunks
        )
        
        # Top-right
        self.subdivide_chunk(
            img_array[:half_size, half_size:], 
            x + half_size, y, half_size, new_threshold, chunks
        )
        
        # Bottom-left
        self.subdivide_chunk(
            img_array[half_size:, :half_size], 
            x, y + half_size, half_size, new_threshold, chunks
        )
        
        # Bottom-right
        self.subdivide_chunk(
            img_array[half_size:, half_size:], 
            x + half_size, y + half_size, half_size, new_threshold, chunks
        )
        
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
            
            # Start subdivision process with 64px threshold
            chunks = []
            self.subdivide_chunk(img_array, 0, 0, 512, self.threshold_64, chunks)
            
            # Create ASCII art image
            ascii_img = Image.new('RGB', (512, 512), 'white')
            draw = ImageDraw.Draw(ascii_img)
            
            # Try to use a monospace font, fallback to default
            try:
                # Try common monospace fonts
                font_paths = [
                    "C:/Windows/Fonts/consola.ttf",  # Consolas
                    "C:/Windows/Fonts/cour.ttf",     # Courier New
                    "C:/Windows/Fonts/arial.ttf"     # Arial as fallback
                ]
                
                base_font_path = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        base_font_path = font_path
                        break
                        
            except Exception:
                base_font_path = None
            
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
                
                # Calculate font size based on chunk size with user-adjustable scaling
                # Use more aggressive scaling: larger chunks get much larger fonts
                if chunk['size'] >= 64:
                    base_font_size = min(64, chunk['size'] // 1.5)  # Even larger base fonts
                elif chunk['size'] >= 32:
                    base_font_size = min(40, chunk['size'] // 1.2)
                elif chunk['size'] >= 16:
                    base_font_size = min(24, chunk['size'] // 1.0)
                else:
                    base_font_size = max(8, chunk['size'] // 1.0)  # Minimum readable size
                
                # Apply user scaling factor
                font_size = int(base_font_size * self.font_scale_factor)
                
                # Ensure font size is reasonable
                font_size = max(4, min(128, font_size))
                
                try:
                    if base_font_path:
                        font = ImageFont.truetype(base_font_path, font_size)
                    else:
                        # Try to get a default font with size
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = ImageFont.load_default()
                except Exception:
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
            self.progress_var.set(f"Generated {len(chunks)} chunks: {size_info}")
            
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
            self.ascii_canvas.create_image(256, 256, image=photo)
            
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