#!/usr/bin/env python3
"""
Font Similarity ASCII Art Generator - Creates ASCII art using least-squares error similarity
Renders each character as 8x8 bitmap and finds best match for each image chunk using LSE
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np
import os
import string

class FontSimilarityAscii:
    def __init__(self, root):
        self.root = root
        self.root.title("Font Similarity ASCII Art Generator")
        self.root.geometry("1800x950")  # Large window for 1920x1080 screen
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
        self.grid_size = 16  # Default grid size
        self.font_bitmaps = {}  # Cache for character bitmaps
        self.current_font_path = None
        
        # Character sets to choose from
        self.character_sets = {
            "Printable ASCII": string.printable[:95],  # Standard printable characters
            "Letters + Numbers": string.ascii_letters + string.digits,
            "Letters Only": string.ascii_letters,
            "Numbers + Symbols": string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?",
            "Extended ASCII": ''.join(chr(i) for i in range(32, 127)),
            "Custom": ""
        }
        
        self.current_character_set = "Printable ASCII"
        self.characters = self.character_sets[self.current_character_set]
        
        # Font scaling for output
        self.output_scale_factor = 1.0
        
        # Text and background colors
        self.text_color = 'black'
        self.background_color = 'white'
        
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
        
        ttk.Label(file_frame, text="Select Image (will resize to 512x512):", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.file_path_var = tk.StringVar(value="No image selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_path_var, foreground="gray")
        file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(file_frame, text="Browse", command=self.select_image).pack(side=tk.RIGHT)
        
        # Grid size selection
        grid_frame = ttk.LabelFrame(control_frame, text="Grid Size (N×N pixels per character)", padding=15)
        grid_frame.pack(fill='x', pady=10)
        
        self.grid_size_var = tk.IntVar(value=16)
        grid_options = [8, 16, 32, 64]
        
        for i, size in enumerate(grid_options):
            rb = ttk.Radiobutton(grid_frame, text=f"{size}×{size} pixels", 
                                variable=self.grid_size_var, value=size,
                                command=self.update_grid_size)
            rb.pack(side=tk.LEFT, padx=20)
        
        # Font selection
        font_frame = ttk.LabelFrame(control_frame, text="Font Selection", padding=15)
        font_frame.pack(fill='x', pady=10)
        
        font_select_frame = ttk.Frame(font_frame)
        font_select_frame.pack(fill='x', pady=5)
        
        ttk.Label(font_select_frame, text="Font File:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.font_path_var = tk.StringVar(value="Click Browse to select font...")
        font_label = ttk.Label(font_select_frame, textvariable=self.font_path_var, 
                              foreground="gray", width=50)
        font_label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(font_select_frame, text="Browse Font", 
                  command=self.select_font).pack(side=tk.RIGHT)
        
        # Character set selection
        charset_frame = ttk.LabelFrame(control_frame, text="Character Set", padding=15)
        charset_frame.pack(fill='x', pady=10)
        
        # Preset selection
        preset_frame = ttk.Frame(charset_frame)
        preset_frame.pack(fill='x', pady=5)
        
        ttk.Label(preset_frame, text="Preset:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.charset_var = tk.StringVar(value="Printable ASCII")
        charset_combo = ttk.Combobox(preset_frame, textvariable=self.charset_var,
                                    values=list(self.character_sets.keys()),
                                    state="readonly", width=25)
        charset_combo.pack(side=tk.LEFT, padx=10)
        charset_combo.bind('<<ComboboxSelected>>', self.update_character_set)
        
        # Custom character set
        custom_frame = ttk.Frame(charset_frame)
        custom_frame.pack(fill='x', pady=5)
        
        ttk.Label(custom_frame, text="Custom:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.custom_chars_var = tk.StringVar()
        custom_entry = ttk.Entry(custom_frame, textvariable=self.custom_chars_var, width=50)
        custom_entry.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        ttk.Button(custom_frame, text="Apply Custom", 
                  command=self.apply_custom_charset).pack(side=tk.RIGHT, padx=5)
        
        # Current character set display
        current_charset_frame = ttk.Frame(charset_frame)
        current_charset_frame.pack(fill='x', pady=5)
        
        ttk.Label(current_charset_frame, text="Current:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.current_chars_var = tk.StringVar(value=self.characters[:50] + "..." if len(self.characters) > 50 else self.characters)
        current_label = ttk.Label(current_charset_frame, textvariable=self.current_chars_var,
                                 foreground="blue", font=("Courier", 9))
        current_label.pack(side=tk.LEFT, padx=10)
        
        # Output settings
        output_frame = ttk.LabelFrame(control_frame, text="Output Settings", padding=15)
        output_frame.pack(fill='x', pady=10)
        
        # Scale factor
        scale_frame = ttk.Frame(output_frame)
        scale_frame.pack(fill='x', pady=5)
        
        ttk.Label(scale_frame, text="Output Scale:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_scale = ttk.Scale(scale_frame, from_=0.5, to=4.0, variable=self.scale_var,
                               orient='horizontal', length=300)
        scale_scale.pack(side=tk.LEFT, padx=10)
        
        self.scale_label = ttk.Label(scale_frame, text="1.0x")
        self.scale_label.pack(side=tk.LEFT, padx=5)
        
        scale_scale.configure(command=self.update_scale_label)
        
        # Color settings
        color_frame = ttk.Frame(output_frame)
        color_frame.pack(fill='x', pady=5)
        
        ttk.Label(color_frame, text="Colors:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        
        self.invert_colors_var = tk.BooleanVar(value=False)
        invert_check = ttk.Checkbutton(color_frame, text="White text on black background",
                                      variable=self.invert_colors_var,
                                      command=self.update_colors)
        invert_check.pack(side=tk.LEFT, padx=20)
        
        # Generate button
        self.generate_btn = ttk.Button(control_frame, text="🎯 Generate Font-Matched ASCII Art",
                                      command=self.generate_ascii, state='disabled')
        self.generate_btn.pack(pady=15)
        
        # Progress
        self.progress_var = tk.StringVar(value="Select an image and font to begin")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var, font=("Arial", 10))
        progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(control_frame, length=400, mode='indeterminate')
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
        ascii_frame = ttk.LabelFrame(image_frame, text="Font-Matched ASCII Art", padding=10)
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
        
        # Save button
        save_btn = ttk.Button(ascii_frame, text="Save ASCII Art", 
                             command=self.save_ascii_image)
        save_btn.pack(pady=5)
        
    def update_grid_size(self):
        """Update grid size setting"""
        self.grid_size = self.grid_size_var.get()
        # Clear font bitmaps cache when grid size changes
        self.font_bitmaps.clear()
        
    def update_scale_label(self, value):
        """Update scale factor label"""
        scale_val = float(value)
        self.scale_label.configure(text=f"{scale_val:.1f}x")
        self.output_scale_factor = scale_val
        
    def update_colors(self):
        """Update text and background colors"""
        if self.invert_colors_var.get():
            self.text_color = 'white'
            self.background_color = 'black'
        else:
            self.text_color = 'black'
            self.background_color = 'white'
            
    def update_character_set(self, event=None):
        """Update character set when preset is selected"""
        selected = self.charset_var.get()
        if selected in self.character_sets:
            self.characters = self.character_sets[selected]
            self.current_character_set = selected
            # Update display
            display_chars = self.characters[:50] + "..." if len(self.characters) > 50 else self.characters
            self.current_chars_var.set(display_chars)
            # Clear font bitmaps cache
            self.font_bitmaps.clear()
            
    def apply_custom_charset(self):
        """Apply custom character set"""
        custom_chars = self.custom_chars_var.get()
        if custom_chars:
            if len(custom_chars) < 2:
                messagebox.showwarning("Invalid Character Set", 
                                     "Custom character set must have at least 2 characters")
                return
            self.characters = custom_chars
            self.current_character_set = "Custom"
            self.charset_var.set("Custom")
            display_chars = custom_chars[:50] + "..." if len(custom_chars) > 50 else custom_chars
            self.current_chars_var.set(display_chars)
            self.font_bitmaps.clear()
            messagebox.showinfo("Success", f"Applied custom character set with {len(custom_chars)} characters")
        else:
            messagebox.showwarning("Empty Character Set", "Please enter custom characters")
            
    def select_image(self):
        """Select input image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Load and validate image
                img = Image.open(filepath)
                
                # Resize to 512x512 for consistency
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                self.input_image = img
                self.display_original_image()
                
                # Update UI
                filename = os.path.basename(filepath)
                self.file_path_var.set(filename)
                self.check_ready_to_generate()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                
    def select_font(self):
        """Select font file"""
        filetypes = [
            ("Font files", "*.ttf *.otf"),
            ("TrueType fonts", "*.ttf"),
            ("OpenType fonts", "*.otf"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Font File",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Test loading the font
                test_font = ImageFont.truetype(filepath, 8)
                
                self.current_font_path = filepath
                filename = os.path.basename(filepath)
                self.font_path_var.set(filename)
                
                # Clear bitmap cache
                self.font_bitmaps.clear()
                self.check_ready_to_generate()
                
            except Exception as e:
                messagebox.showerror("Font Error", f"Could not load font: {str(e)}")
                
    def check_ready_to_generate(self):
        """Check if ready to generate and update button state"""
        if self.input_image and self.current_font_path:
            self.generate_btn.configure(state='normal')
            self.progress_var.set("Ready to generate ASCII art")
        else:
            self.generate_btn.configure(state='disabled')
            missing = []
            if not self.input_image:
                missing.append("image")
            if not self.current_font_path:
                missing.append("font")
            self.progress_var.set(f"Please select: {', '.join(missing)}")
            
    def display_original_image(self):
        """Display the original image on canvas"""
        if self.input_image:
            # Create PhotoImage for display
            photo = ImageTk.PhotoImage(self.input_image)
            
            # Clear canvas and display image
            self.original_canvas.delete("all")
            self.original_canvas.create_image(256, 256, image=photo)
            
            # Keep reference
            self.original_photo_ref = photo
            
    def generate_character_bitmaps(self):
        """Generate 8x8 bitmaps for each character in the character set"""
        if not self.current_font_path:
            return
            
        self.progress_var.set("Generating character bitmaps...")
        self.root.update()
        
        # Clear existing bitmaps
        self.font_bitmaps.clear()
        
        try:
            # Load font at size 8 (for 8x8 bitmaps)
            font = ImageFont.truetype(self.current_font_path, 8)
            
            for char in self.characters:
                # Create 8x8 image for character
                char_img = Image.new('L', (8, 8), 255)  # Grayscale, white background
                draw = ImageDraw.Draw(char_img)
                
                # Draw character centered in 8x8 space
                draw.text((4, 4), char, font=font, fill=0, anchor='mm')  # Black text
                
                # Convert to numpy array and normalize to 0-1 range
                char_array = np.array(char_img, dtype=np.float32) / 255.0
                
                # Store the bitmap (invert so 0=white, 1=black for easier comparison)
                self.font_bitmaps[char] = 1.0 - char_array
                
        except Exception as e:
            messagebox.showerror("Font Error", f"Error generating character bitmaps: {str(e)}")
            
    def calculate_chunk_luma(self, img_chunk):
        """Convert RGB chunk to grayscale using standard luminance weights"""
        # img_chunk shape: (height, width, 3)
        luma = 0.299 * img_chunk[:, :, 0] + 0.587 * img_chunk[:, :, 1] + 0.114 * img_chunk[:, :, 2]
        return luma / 255.0  # Normalize to 0-1 range
        
    def find_best_matching_character(self, img_chunk):
        """Find character with lowest least-squares error compared to image chunk"""
        # Convert image chunk to grayscale
        chunk_luma = self.calculate_chunk_luma(img_chunk)
        
        # Resize chunk to 8x8 for comparison
        chunk_resized = np.array(Image.fromarray((chunk_luma * 255).astype(np.uint8)).resize((8, 8)))
        chunk_normalized = chunk_resized.astype(np.float32) / 255.0
        
        # Invert so 0=white, 1=black to match character bitmaps
        chunk_normalized = 1.0 - chunk_normalized
        
        best_char = ' '
        min_error = float('inf')
        
        # Compare with each character bitmap
        for char, char_bitmap in self.font_bitmaps.items():
            # Calculate least-squares error
            error = np.sum((chunk_normalized - char_bitmap) ** 2)
            
            if error < min_error:
                min_error = error
                best_char = char
                
        return best_char, min_error
        
    def generate_ascii(self):
        """Generate ASCII art using font similarity matching"""
        if not self.input_image or not self.current_font_path:
            return
            
        try:
            # Show progress
            self.progress_var.set("Generating ASCII art...")
            self.progress_bar.start()
            self.root.update()
            
            # Generate character bitmaps if not cached
            if not self.font_bitmaps:
                self.generate_character_bitmaps()
                
            # Convert image to numpy array
            img_array = np.array(self.input_image)
            
            # Calculate grid dimensions
            grid_cols = 512 // self.grid_size
            grid_rows = 512 // self.grid_size
            
            self.progress_var.set(f"Processing {grid_rows}×{grid_cols} grid...")
            self.root.update()
            
            # Process each grid cell
            ascii_chars = []
            total_chunks = grid_rows * grid_cols
            processed = 0
            
            for row in range(grid_rows):
                char_row = []
                for col in range(grid_cols):
                    # Extract chunk
                    y_start = row * self.grid_size
                    y_end = y_start + self.grid_size
                    x_start = col * self.grid_size
                    x_end = x_start + self.grid_size
                    
                    chunk = img_array[y_start:y_end, x_start:x_end]
                    
                    # Find best matching character
                    best_char, error = self.find_best_matching_character(chunk)
                    char_row.append(best_char)
                    
                    processed += 1
                    if processed % 10 == 0:  # Update progress every 10 chunks
                        progress_pct = int((processed / total_chunks) * 100)
                        self.progress_var.set(f"Processing... {progress_pct}%")
                        self.root.update()
                        
                ascii_chars.append(char_row)
            
            # Create ASCII art image
            self.progress_var.set("Rendering ASCII art...")
            self.root.update()
            
            # Calculate output dimensions
            font_size = max(8, int(8 * self.output_scale_factor))
            char_width = font_size
            char_height = font_size
            
            output_width = grid_cols * char_width
            output_height = grid_rows * char_height
            
            # Create output image
            ascii_img = Image.new('RGB', (output_width, output_height), self.background_color)
            draw = ImageDraw.Draw(ascii_img)
            
            # Load font for rendering
            try:
                render_font = ImageFont.truetype(self.current_font_path, font_size)
            except:
                render_font = ImageFont.load_default()
            
            # Draw characters
            for row in range(grid_rows):
                for col in range(grid_cols):
                    char = ascii_chars[row][col]
                    x = col * char_width + char_width // 2
                    y = row * char_height + char_height // 2
                    
                    draw.text((x, y), char, font=render_font, fill=self.text_color, anchor='mm')
            
            # Store and display result
            self.ascii_image = ascii_img
            self.display_ascii_image()
            
            # Update progress
            self.progress_bar.stop()
            avg_chars = len(set(''.join(''.join(row) for row in ascii_chars)))
            self.progress_var.set(f"Generated {grid_rows}×{grid_cols} ASCII art using {avg_chars} unique characters")
            
        except Exception as e:
            self.progress_bar.stop()
            self.progress_var.set("Error generating ASCII art")
            messagebox.showerror("Error", f"Could not generate ASCII art: {str(e)}")
            
    def display_ascii_image(self):
        """Display ASCII art on canvas with scaling for large images"""
        if self.ascii_image:
            # Scale image to fit in canvas if necessary
            display_img = self.ascii_image
            
            # If image is larger than canvas, scale it down for display
            canvas_size = 512
            if display_img.width > canvas_size or display_img.height > canvas_size:
                display_img = display_img.copy()
                display_img.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)
            
            # Create PhotoImage for display
            photo = ImageTk.PhotoImage(display_img)
            
            # Clear canvas and display image
            self.ascii_canvas.delete("all")
            
            # Center the image
            x_pos = canvas_size // 2
            y_pos = canvas_size // 2
            self.ascii_canvas.create_image(x_pos, y_pos, image=photo)
            
            # Configure scroll region for full-size image
            self.ascii_canvas.configure(scrollregion=(0, 0, self.ascii_image.width, self.ascii_image.height))
            
            # Keep reference
            self.ascii_photo_ref = photo
            
    def save_ascii_image(self):
        """Save the ASCII art image"""
        if not self.ascii_image:
            messagebox.showwarning("No ASCII Art", "Please generate ASCII art first")
            return
            
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Save ASCII Art",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                self.ascii_image.save(filepath)
                messagebox.showinfo("Success", f"ASCII art saved as: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")

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
    app = FontSimilarityAscii(root)
    root.mainloop()

if __name__ == "__main__":
    main()
