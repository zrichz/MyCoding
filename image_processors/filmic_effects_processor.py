#!/usr/bin/env python3
"""
Filmic Effects Processor - Applies film grain and vignette effects to 720x1600 images
Creates cinematic effects with edge-enhanced grain and circular vignetting
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageChops
import numpy as np
import os
import threading
from pathlib import Path

class FilmicEffectsProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Filmic Effects Processor - 720x1600")
        self.root.geometry("2500x1400")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.input_directory = None
        self.preview_image = None
        self.processed_preview = None
        self.image_files = []
        self.current_preview_index = 0
        
        # Image references for canvas display (prevent garbage collection)
        self.original_photo_ref = None
        self.processed_photo_ref = None
        
        # Effect parameters
        self.grain_intensity = tk.DoubleVar(value=0.03)
        self.grain_edge_boost = tk.DoubleVar(value=1.8)
        self.vignette_strength = tk.DoubleVar(value=0.10)
        self.saturation_reduction = tk.DoubleVar(value=0.0)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill='both', expand=True)
        
        # Configure style for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='white')
        style.configure('Dark.TButton', background='#404040', foreground='white')
        
        # Directory selection
        dir_frame = ttk.LabelFrame(main_frame, text="Image Directory", padding=10)
        dir_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(dir_frame, text="📁 Select Directory", 
                  command=self.select_directory).pack(side=tk.LEFT, padx=(0, 10))
        
        self.dir_label = ttk.Label(dir_frame, text="No directory selected", 
                                  foreground="gray")
        self.dir_label.pack(side=tk.LEFT)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Effect Controls", padding=15)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Film grain controls
        grain_frame = ttk.Frame(control_frame)
        grain_frame.pack(fill='x', pady=5)
        
        ttk.Label(grain_frame, text="Film Grain Intensity:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        grain_scale = ttk.Scale(grain_frame, from_=0.0, to=0.07, 
                               variable=self.grain_intensity, orient='horizontal', 
                               length=300, command=self.update_preview)
        grain_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.grain_label = ttk.Label(grain_frame, text="0.03", width=6)
        self.grain_label.pack(side=tk.RIGHT, padx=5)
        
        # Edge boost controls
        edge_frame = ttk.Frame(control_frame)
        edge_frame.pack(fill='x', pady=5)
        
        ttk.Label(edge_frame, text="Edge Grain Boost:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        edge_scale = ttk.Scale(edge_frame, from_=1.0, to=3.0, 
                              variable=self.grain_edge_boost, orient='horizontal', 
                              length=300, command=self.update_preview)
        edge_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.edge_label = ttk.Label(edge_frame, text="1.8", width=6)
        self.edge_label.pack(side=tk.RIGHT, padx=5)
        
        # Vignette controls
        vignette_frame = ttk.Frame(control_frame)
        vignette_frame.pack(fill='x', pady=5)
        
        ttk.Label(vignette_frame, text="Vignette Strength:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        vignette_scale = ttk.Scale(vignette_frame, from_=0.0, to=0.375, 
                                  variable=self.vignette_strength, orient='horizontal', 
                                  length=300, command=self.update_preview)
        vignette_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.vignette_label = ttk.Label(vignette_frame, text="0.10", width=6)
        self.vignette_label.pack(side=tk.RIGHT, padx=5)
        
        # Saturation controls
        saturation_frame = ttk.Frame(control_frame)
        saturation_frame.pack(fill='x', pady=5)
        
        ttk.Label(saturation_frame, text="Saturation Reduction:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        saturation_scale = ttk.Scale(saturation_frame, from_=0.0, to=0.40, 
                                    variable=self.saturation_reduction, orient='horizontal', 
                                    length=300, command=self.update_preview)
        saturation_scale.pack(side=tk.LEFT, padx=10, fill='x', expand=True)
        
        self.saturation_label = ttk.Label(saturation_frame, text="0.00", width=6)
        self.saturation_label.pack(side=tk.RIGHT, padx=5)
        
        # Update labels when scales change
        grain_scale.configure(command=self.update_grain_label)
        edge_scale.configure(command=self.update_edge_label)
        vignette_scale.configure(command=self.update_vignette_label)
        saturation_scale.configure(command=self.update_saturation_label)
        
        # Preview and processing controls
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(action_frame, text="🔍 Load Preview", 
                  command=self.load_preview).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="◀ Previous", 
                  command=self.previous_preview).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="Next ▶", 
                  command=self.next_preview).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(action_frame, text="🎬 Process All Images", 
                  command=self.process_all_images, 
                  style='Accent.TButton').pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Select a directory to begin")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Preview area with side-by-side comparison
        preview_frame = ttk.LabelFrame(main_frame, text="Preview Comparison (Top Half 1:1)", padding=10)
        preview_frame.pack(fill='both', expand=True)
        
        # Container for both preview canvases
        canvas_container = tk.Frame(preview_frame, bg='#2b2b2b')
        canvas_container.pack(expand=True, fill='both')
        
        # Original image canvas (left side)
        original_frame = tk.Frame(canvas_container, bg='#2b2b2b')
        original_frame.pack(side=tk.LEFT, padx=10)
        
        original_label = tk.Label(original_frame, text="Original (Top Half)", 
                                 bg='#2b2b2b', fg='white', font=("Arial", 12, "bold"))
        original_label.pack(pady=5)
        
        self.original_canvas = tk.Canvas(original_frame, width=720, height=800, 
                                        bg='#1a1a1a', highlightthickness=1, 
                                        highlightbackground='#555555')
        self.original_canvas.pack()
        
        # Processed image canvas (right side)
        processed_frame = tk.Frame(canvas_container, bg='#2b2b2b')
        processed_frame.pack(side=tk.LEFT, padx=10)
        
        processed_label = tk.Label(processed_frame, text="With Filmic Effects", 
                                  bg='#2b2b2b', fg='white', font=("Arial", 12, "bold"))
        processed_label.pack(pady=5)
        
        self.processed_canvas = tk.Canvas(processed_frame, width=720, height=800, 
                                         bg='#1a1a1a', highlightthickness=1, 
                                         highlightbackground='#555555')
        self.processed_canvas.pack()
        
    def update_grain_label(self, value):
        """Update grain intensity label"""
        self.grain_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
    def update_edge_label(self, value):
        """Update edge boost label"""
        self.edge_label.config(text=f"{float(value):.1f}")
        self.update_preview()
        
    def update_vignette_label(self, value):
        """Update vignette strength label"""
        self.vignette_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
    def update_saturation_label(self, value):
        """Update saturation reduction label"""
        self.saturation_label.config(text=f"{float(value):.2f}")
        self.update_preview()
        
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
            
    def create_film_grain(self, width, height, intensity, edge_boost):
        """Create film grain effect with edge enhancement for saturation and brightness only"""
        # Create base noise for saturation and brightness (2 channels)
        noise = np.random.normal(0, intensity, (height, width, 2))
        
        # Create edge mask for enhanced grain at edges
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        # Distance from center (normalized)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        edge_factor = (distance / max_distance) ** 0.8
        
        # Apply edge enhancement to both saturation and brightness noise
        for i in range(2):  # Saturation and brightness channels
            noise[:, :, i] *= (1 + edge_factor * (edge_boost - 1))
        
        # Clamp values (smaller range for HSV adjustments)
        noise = np.clip(noise, -0.3, 0.3)
        
        return noise
        
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
        
        # Create and apply film grain to saturation and brightness only
        grain = self.create_film_grain(width, height, 
                                     self.grain_intensity.get(), 
                                     self.grain_edge_boost.get())
        
        # Apply grain to saturation (channel 1) and brightness (channel 2)
        # Hue (channel 0) remains unchanged
        hsv_array[:, :, 1] += grain[:, :, 0]  # Saturation noise
        hsv_array[:, :, 2] += grain[:, :, 1]  # Brightness noise
        
        # Apply saturation reduction
        saturation_factor = 1.0 - self.saturation_reduction.get()
        hsv_array[:, :, 1] *= saturation_factor
        
        # Create and apply vignette to brightness only
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
        return hsv_result.convert('RGB')
        
    def update_preview(self, event=None):
        """Update preview with current effects showing top half at 1:1 scale"""
        if not self.preview_image:
            return
            
        try:
            # Crop to top half (720x800 from 720x1600)
            top_half_original = self.preview_image.crop((0, 0, 720, 800))
            
            # Apply effects to full image first, then crop
            processed_full = self.apply_filmic_effects(self.preview_image)
            top_half_processed = processed_full.crop((0, 0, 720, 800))
            
            self.processed_preview = processed_full
            
            # Convert to PhotoImage for display at 1:1 scale
            original_photo = ImageTk.PhotoImage(top_half_original)
            processed_photo = ImageTk.PhotoImage(top_half_processed)
            
            # Clear canvases and display images
            self.original_canvas.delete("all")
            self.processed_canvas.delete("all")
            
            # Display images centered in canvases
            self.original_canvas.create_image(360, 400, image=original_photo)
            self.processed_canvas.create_image(360, 400, image=processed_photo)
            
            # Keep references to prevent garbage collection
            self.original_photo_ref = original_photo
            self.processed_photo_ref = processed_photo
            
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
                                   f"Images will be saved with '_filmic' suffix.")
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
                    
                    # Apply effects
                    processed = self.apply_filmic_effects(image)
                    
                    # Create output filename
                    output_path = image_path.parent / f"{image_path.stem}_filmic{image_path.suffix}"
                    
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
                                                          f"Output files saved with '_filmic' suffix."))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", 
                                                           f"An error occurred during processing: {str(e)}"))

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