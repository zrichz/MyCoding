#!/usr/bin/env python3
"""
Fil        # Effect parameters
        self.grain_intensity = tk.DoubleVar(value=0.03)
        self.vignette_strength = tk.DoubleVar(value=0.10)
        self.saturation_reduction = tk.DoubleVar(value=0.0)
        self.chromatic_aberration = tk.DoubleVar(value=0.0)fects Processor - Applies film grain and vignette effects to 720x1600 images
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
        self.root.geometry("1600x1400")
        self.root.configure(bg='#808080')
        
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
        self.grain_intensity = tk.DoubleVar(value=0.20)
        self.grain_edge_boost = tk.DoubleVar(value=1.8)
        self.vignette_strength = tk.DoubleVar(value=0.3)
        self.saturation_reduction = tk.DoubleVar(value=0.3)
        self.chromatic_aberration = tk.DoubleVar(value=0.08)
        
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
        grain_scale = ttk.Scale(grain_frame, from_=0.0, to=0.56, 
                               variable=self.grain_intensity, orient='horizontal', 
                               length=800, style='Blue.Horizontal.TScale')
        grain_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.grain_label = ttk.Label(grain_frame, text="0.20", width=8,
                                    font=("Arial", 11, "bold"))
        self.grain_label.pack(side=tk.LEFT)
        
        # Vignette controls
        vignette_frame = ttk.Frame(control_frame)
        vignette_frame.pack(fill='x', pady=5)
        
        ttk.Label(vignette_frame, text="Vignette Strength:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        vignette_scale = ttk.Scale(vignette_frame, from_=0.0, to=0.75, 
                                  variable=self.vignette_strength, orient='horizontal', 
                                  length=800, style='Blue.Horizontal.TScale')
        vignette_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.vignette_label = ttk.Label(vignette_frame, text="0.30", width=8,
                                       font=("Arial", 11, "bold"))
        self.vignette_label.pack(side=tk.LEFT)
        
        # Saturation controls
        saturation_frame = ttk.Frame(control_frame)
        saturation_frame.pack(fill='x', pady=5)
        
        ttk.Label(saturation_frame, text="Saturation Reduction:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        saturation_scale = ttk.Scale(saturation_frame, from_=0.0, to=1.0, 
                                    variable=self.saturation_reduction, orient='horizontal', 
                                    length=800, style='Blue.Horizontal.TScale')
        saturation_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.saturation_label = ttk.Label(saturation_frame, text="0.30", width=8,
                                         font=("Arial", 11, "bold"))
        self.saturation_label.pack(side=tk.LEFT)
        
        # Chromatic aberration controls
        aberration_frame = ttk.Frame(control_frame)
        aberration_frame.pack(fill='x', pady=5)
        
        ttk.Label(aberration_frame, text="Chromatic Aberration:", 
                 font=("Arial", 10, "bold"), width=20).pack(side=tk.LEFT)
        aberration_scale = ttk.Scale(aberration_frame, from_=0.0, to=1.0, 
                                    variable=self.chromatic_aberration, orient='horizontal', 
                                    length=800, style='Blue.Horizontal.TScale')
        aberration_scale.pack(side=tk.LEFT, padx=(10, 15))
        
        self.aberration_label = ttk.Label(aberration_frame, text="0.08", width=8,
                                         font=("Arial", 11, "bold"))
        self.aberration_label.pack(side=tk.LEFT)
        
        # Update labels when scales change
        grain_scale.configure(command=self.update_grain_label)
        vignette_scale.configure(command=self.update_vignette_label)
        saturation_scale.configure(command=self.update_saturation_label)
        aberration_scale.configure(command=self.update_aberration_label)
        
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
        
        # Preview area with side-by-side comparison and synchronized scrolling
        preview_frame = ttk.LabelFrame(main_frame, text="Preview Comparison (1:1 Scale - Scroll to see full image)", padding=10)
        preview_frame.pack(fill='both', expand=True)
        
        # Container for both preview canvases
        canvas_container = tk.Frame(preview_frame, bg='#808080')
        canvas_container.pack(expand=True, fill='both')
        
        # Original image canvas (left side) with scrollbar
        original_frame = tk.Frame(canvas_container, bg='#808080')
        original_frame.pack(side=tk.LEFT, padx=10)
        
        original_label = tk.Label(original_frame, text="Original (720x1600)", 
                                 bg='#808080', fg='black', font=("Arial", 12, "bold"))
        original_label.pack(pady=5)
        
        # Create frame for canvas and scrollbar
        original_canvas_frame = tk.Frame(original_frame, bg='#808080')
        original_canvas_frame.pack()
        
        self.original_canvas = tk.Canvas(original_canvas_frame, width=720, height=800, 
                                        bg='#1a1a1a', highlightthickness=1, 
                                        highlightbackground='#555555')
        self.original_canvas.pack(side=tk.LEFT)
        
        # Vertical scrollbar for original canvas
        original_scrollbar = tk.Scrollbar(original_canvas_frame, orient='vertical', 
                                         command=self.sync_scroll)
        original_scrollbar.pack(side=tk.RIGHT, fill='y')
        self.original_canvas.configure(yscrollcommand=original_scrollbar.set)
        
        # Processed image canvas (right side) with scrollbar
        processed_frame = tk.Frame(canvas_container, bg='#808080')
        processed_frame.pack(side=tk.LEFT, padx=10)
        
        processed_label = tk.Label(processed_frame, text="With Filmic Effects", 
                                  bg='#808080', fg='black', font=("Arial", 12, "bold"))
        processed_label.pack(pady=5)
        
        # Create frame for canvas and scrollbar
        processed_canvas_frame = tk.Frame(processed_frame, bg='#808080')
        processed_canvas_frame.pack()
        
        self.processed_canvas = tk.Canvas(processed_canvas_frame, width=720, height=800, 
                                         bg='#1a1a1a', highlightthickness=1, 
                                         highlightbackground='#555555')
        self.processed_canvas.pack(side=tk.LEFT)
        
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
        """Apply film grain to RGB image as final processing step"""
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
        """Apply saturation reduction to RGB image as final processing step"""
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
        center_x, center_y = width // 2, height // 2
        
        # Split into RGB channels
        r, g, b = image.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)
        
        # Create distance map from center
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create aberration factor based on distance
        # No effect within radius 300, full effect at radius 800+
        aberration_factor = np.zeros_like(distance)
        mask = distance > 300
        aberration_factor[mask] = np.minimum((distance[mask] - 300) / (800 - 300), 1.0)

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
        
        # Create and apply vignette to brightness only (no saturation reduction here anymore)
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
        
        # Apply chromatic aberration
        aberration_result = self.apply_chromatic_aberration(rgb_result, self.chromatic_aberration.get())
        
        # Apply film grain
        grain_result = self.apply_film_grain_rgb(aberration_result, 
                                               self.grain_intensity.get())
        
        # Apply saturation reduction as the final processing step
        final_result = self.apply_saturation_reduction_rgb(grain_result, self.saturation_reduction.get())
        
        return final_result
        
    def update_preview(self, event=None):
        """Update preview with current effects showing full 1600px images at 1:1 scale with scrolling"""
        if not self.preview_image:
            return
            
        try:
            # Save current scroll position before updating
            current_scroll_pos = self.original_canvas.yview()
            
            # Use full image (720x1600)
            original_full = self.preview_image
            
            # Apply effects to full image
            processed_full = self.apply_filmic_effects(self.preview_image)
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
                                   f"Images will be saved in 'filmic' subdirectory with '_filmic' suffix.")
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