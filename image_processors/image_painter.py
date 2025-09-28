#!/usr/bin/env python3
"""
Image Painter - Load image and analyze RGB gradient directions
Shows original image thumbnail (512x512) and blurred gradient directions for R, G, B channels (256x256 each)
Designed for 1920x1080 screen resolution.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import os

class ImagePainter:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Painter - Gradient Analysis")
        self.root.geometry("1800x900")
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Store image data
        self.original_image = None
        self.thumbnail_image = None
        self.gradient_images = {'R': None, 'G': None, 'B': None}
        self.painted_image = None
        self.gradient_data = {'R': None, 'G': None, 'B': None}  # Store raw gradient data
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="Image Painter - Gradient Analysis", 
                              font=("Arial", 20, "bold"))
        title_label.pack(pady=15)
        
        # Load button
        load_frame = ttk.Frame(self.root)
        load_frame.pack(pady=10)
        
        load_btn = ttk.Button(load_frame, text="Load Image", 
                             command=self.load_image, width=15)
        load_btn.pack()
        
        # Blur amount control
        blur_frame = ttk.Frame(self.root)
        blur_frame.pack(pady=10)
        
        blur_label = tk.Label(blur_frame, text="Blur Amount (pixels):", font=("Arial", 12))
        blur_label.pack(side=tk.LEFT, padx=5)
        
        self.blur_amount = tk.IntVar(value=2)
        blur_spinbox = tk.Spinbox(blur_frame, from_=0, to=10, textvariable=self.blur_amount, 
                                 width=5, font=("Arial", 11))
        blur_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Update button to reprocess with new blur amount
        update_btn = ttk.Button(blur_frame, text="Update Gradients", 
                               command=self.update_gradients, width=15)
        update_btn.pack(side=tk.LEFT, padx=10)
        
        # Paint button to create brushstroke version
        paint_btn = ttk.Button(blur_frame, text="Paint Brushstrokes", 
                              command=self.paint_brushstrokes, width=15)
        paint_btn.pack(side=tk.LEFT, padx=10)
        
        # Save button for brushstroke painting
        save_btn = ttk.Button(blur_frame, text="Save Painting", 
                             command=self.save_painted_image, width=15)
        save_btn.pack(side=tk.LEFT, padx=10)
        
        # Main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(expand=True, fill='both', padx=30, pady=20)
        
        # Left side - Original image thumbnail (512x512)
        left_frame = ttk.LabelFrame(content_frame, text="Original Image (512x512)", padding=20)
        left_frame.pack(side=tk.LEFT, padx=20, pady=10, fill='y', expand=False)
        left_frame.configure(width=550, height=550)  # Set minimum size for the frame
        
        self.thumbnail_label = tk.Label(left_frame, text="No image loaded", 
                                       bg="lightgray", font=("Arial", 12))
        self.thumbnail_label.pack(padx=10, pady=10)
        
        # Right side - Brushstroke painting display
        right_frame = ttk.LabelFrame(content_frame, text="RGB Brushstroke Painting", padding=20)
        right_frame.pack(side=tk.LEFT, padx=20, pady=10, fill='both', expand=True)
        
        # Brushstroke explanation
        explanation = tk.Label(right_frame, 
                              text="Pure RGB brushstrokes based on gradient directions\nRed, Green, Blue channels painted with 50% transparency",
                              font=("Arial", 11), justify=tk.CENTER)
        explanation.pack(pady=10)
        
        # Brushstroke display area
        brushstroke_frame = ttk.Frame(right_frame)
        brushstroke_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        self.brushstroke_label = tk.Label(brushstroke_frame, text="Load image and click 'Paint Brushstrokes'", 
                                         bg="white", font=("Arial", 12))
        self.brushstroke_label.pack(expand=True, fill='both')
        
        # Status label
        self.status_label = tk.Label(self.root, text="Click 'Load Image' to begin analysis", 
                                    font=("Arial", 11))
        self.status_label.pack(pady=15)
        
    def load_image(self):
        """Load an image file and process it"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=filetypes
        )
        
        if filename:
            try:
                self.status_label.configure(text="Loading and processing image...")
                self.root.update()
                
                # Load image
                self.original_image = Image.open(filename)
                
                # Convert to RGB if needed
                if self.original_image.mode != 'RGB':
                    self.original_image = self.original_image.convert('RGB')
                
                # Create thumbnail that fits in 512x512
                self.create_thumbnail()
                
                # Generate gradient analysis (use original image, not thumbnail)
                self.analyze_gradients()
                
                # Update displays
                self.update_displays()
                
                filename_short = os.path.basename(filename)
                if len(filename_short) > 50:
                    filename_short = filename_short[:47] + "..."
                
                self.status_label.configure(text=f"Loaded: {filename_short}")
                
            except Exception as e:
                messagebox.showerror("Error Loading Image", f"Could not load image: {str(e)}")
                self.status_label.configure(text="Error loading image")
    
    def create_thumbnail(self):
        """Create a thumbnail that fits within 512x512"""
        if not self.original_image:
            return
        
        # Get original dimensions
        orig_width, orig_height = self.original_image.size
        
        # Calculate thumbnail size maintaining aspect ratio
        img_copy = self.original_image.copy()
        
        # Ensure minimum size - if image is very small, resize it up
        if orig_width < 100 or orig_height < 100:
            # Scale up small images
            scale_factor = max(200 / orig_width, 200 / orig_height)
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)
            img_copy = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)

        
        # Now create thumbnail that fits in 512x512
        img_copy.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Get final thumbnail dimensions
        thumb_width, thumb_height = img_copy.size
        

        
        self.thumbnail_image = img_copy
    
    def analyze_gradients(self):
        """Analyze gradient directions for each RGB channel"""
        if not self.original_image:
            return
        
        # Convert to numpy array for gradient calculation
        img_array = np.array(self.original_image)
        height, width = img_array.shape[:2]
        
        # Process each RGB channel
        channels = ['R', 'G', 'B']
        gradient_results = {}
        raw_gradient_data = {}
        
        for i, channel in enumerate(channels):
            # Extract channel
            channel_data = img_array[:, :, i].astype(np.float32)
            
            # Apply blur to the input channel BEFORE gradient calculation
            blur_radius = self.blur_amount.get()
            if blur_radius > 0:
                # Convert to PIL Image for blurring
                channel_img = Image.fromarray(channel_data.astype(np.uint8))
                channel_img = channel_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                channel_data = np.array(channel_img).astype(np.float32)
            
            # Calculate gradients using numpy on the (possibly blurred) channel data
            grad_x = np.gradient(channel_data, axis=1)  # Horizontal gradient
            grad_y = np.gradient(channel_data, axis=0)  # Vertical gradient
            
            # Calculate gradient magnitude and direction
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_direction = np.arctan2(grad_y, grad_x)
            
            # Store raw gradient data for brushstroke painting
            raw_gradient_data[channel] = {
                'magnitude': gradient_magnitude,
                'direction': gradient_direction,
                'grad_x': grad_x,
                'grad_y': grad_y
            }
            
            # Normalize gradient direction to 0-255 range for visualization
            # Convert from [-π, π] to [0, 255]
            gradient_direction_norm = ((gradient_direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Apply gradient magnitude as alpha/intensity
            gradient_magnitude_norm = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
            
            # Create visualization combining direction and magnitude
            # Use HSV-like representation: direction as hue, magnitude as value
            visualization = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Map direction to color channels for better visualization
            if channel == 'R':
                visualization[:, :, 0] = gradient_magnitude_norm  # Red intensity
                visualization[:, :, 1] = gradient_direction_norm // 2  # Some green
                visualization[:, :, 2] = 255 - gradient_direction_norm  # Inverse blue
            elif channel == 'G':
                visualization[:, :, 0] = 255 - gradient_direction_norm  # Inverse red
                visualization[:, :, 1] = gradient_magnitude_norm  # Green intensity
                visualization[:, :, 2] = gradient_direction_norm // 2  # Some blue
            else:  # Blue channel
                visualization[:, :, 0] = gradient_direction_norm // 2  # Some red
                visualization[:, :, 1] = 255 - gradient_direction_norm  # Inverse green
                visualization[:, :, 2] = gradient_magnitude_norm  # Blue intensity
            
            # Convert back to PIL Image
            gradient_img = Image.fromarray(visualization)
            
            # Apply histogram equalization to improve contrast
            gradient_img = self.equalize_image(gradient_img)
            
            # Resize to 300x300 for display (no additional blur needed since input was blurred)
            gradient_img = gradient_img.resize((300, 300), Image.Resampling.LANCZOS)
            
            gradient_results[channel] = gradient_img
        
        # Create combined gradient visualization
        combined_array = np.zeros((height, width, 3), dtype=np.float32)
        
        for i, channel in enumerate(channels):
            channel_data = img_array[:, :, i].astype(np.float32)
            
            # Apply blur to the input channel BEFORE gradient calculation for combined gradient too
            blur_radius = self.blur_amount.get()
            if blur_radius > 0:
                # Convert to PIL Image for blurring
                channel_img = Image.fromarray(channel_data.astype(np.uint8))
                channel_img = channel_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                channel_data = np.array(channel_img).astype(np.float32)
            
            grad_x = np.gradient(channel_data, axis=1)
            grad_y = np.gradient(channel_data, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize and add to combined
            if gradient_magnitude.max() > 0:
                combined_array[:, :, i] = gradient_magnitude / gradient_magnitude.max()
        
        # Convert combined to image
        combined_norm = (combined_array * 255).astype(np.uint8)
        combined_img = Image.fromarray(combined_norm)
        
        # Apply histogram equalization to improve contrast
        combined_img = self.equalize_image(combined_img)
        
        # Resize to 300x300 for display (no additional blur needed since input was already blurred)
        combined_img = combined_img.resize((300, 300), Image.Resampling.LANCZOS)
        
        gradient_results['Combined'] = combined_img
        self.gradient_images = gradient_results
        self.gradient_data = raw_gradient_data
    
    def update_displays(self):
        """Update all display elements"""
        # Update thumbnail display
        if self.thumbnail_image:

            thumbnail_photo = ImageTk.PhotoImage(self.thumbnail_image)
            self.thumbnail_label.configure(image=thumbnail_photo, text="")
            self.thumbnail_label.image = thumbnail_photo  # Keep reference
        
        # Update brushstroke display (will be updated when painting is done)
        pass

    def equalize_image(self, img):
        """Apply histogram equalization to improve contrast"""
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Apply histogram equalization to each channel separately
        equalized_channels = []
        
        for i in range(img_array.shape[2]):  # For each RGB channel
            channel = img_array[:, :, i]
            
            # Calculate histogram
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            
            # Calculate cumulative distribution function (CDF)
            cdf = hist.cumsum()
            
            # Normalize CDF to range [0, 255]
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf_normalized = cdf_normalized.astype(np.uint8)
            
            # Apply equalization using the CDF as a lookup table
            equalized_channel = cdf_normalized[channel]
            equalized_channels.append(equalized_channel)
        
        # Combine channels back into RGB image
        equalized_array = np.stack(equalized_channels, axis=2)
        
        # Convert back to PIL Image
        return Image.fromarray(equalized_array)

    def update_gradients(self):
        """Update gradient analysis with current blur settings"""
        if not self.original_image:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        self.status_label.configure(text="Updating gradients with new blur amount...")
        self.root.update()
        
        # Regenerate gradient analysis
        self.analyze_gradients()
        
        # Update displays
        self.update_displays()
        
        self.status_label.configure(text=f"Gradients updated with blur radius: {self.blur_amount.get()} pixels")

    def paint_brushstrokes(self):
        """Create RGB brushstroke painting using gradient directions"""
        if not self.original_image or not self.gradient_data:
            messagebox.showwarning("No Data", "Please load an image and calculate gradients first")
            return
        
        self.status_label.configure(text="Creating RGB brushstroke painting...")
        self.root.update()
        
        # Convert original image to numpy array
        img_array = np.array(self.original_image)
        height, width = img_array.shape[:2]
        
        # Create output canvas with alpha channel (RGBA)
        painted_array = np.ones((height, width, 4), dtype=np.float32) * 255  # Start with white background
        painted_array[:, :, 3] = 255  # Full alpha for background
        
        # Brushstroke parameters
        brush_length = 8   # Length of each brushstroke
        brush_width = 4    # Width of each brushstroke (4 pixels as requested)
        stride = 4         # Spacing between brushstroke centers
        alpha = 0.1        # 10% transparency for all strokes
        
        # Paint each RGB channel separately with blended colors
        channel_colors = {
            'R': [255, 0, 0],    # Pure red
            'G': [0, 255, 0],    # Pure green  
            'B': [0, 0, 255]     # Pure blue
        }
        
        for channel_name, pure_color in channel_colors.items():
            if channel_name not in self.gradient_data:
                continue
                
            grad_data = self.gradient_data[channel_name]
            
            # Paint brushstrokes for this channel
            for y in range(0, height - brush_length, stride):
                for x in range(0, width - brush_length, stride):
                    # Sample gradient at brush center
                    center_y = min(y + brush_width//2, height-1)
                    center_x = min(x + brush_width//2, width-1)
                    
                    direction = grad_data['direction'][center_y, center_x]
                    magnitude = grad_data['magnitude'][center_y, center_x]
                    
                    # Skip if gradient is too weak
                    if magnitude < 1.0:
                        continue
                    
                    # Sample original color at this position for blending
                    original_color = img_array[center_y, center_x]
                    
                    # Blend original color (25%) with pure color (75%)
                    blended_color = [
                        original_color[0] * 0.25 + pure_color[0] * 0.75,
                        original_color[1] * 0.25 + pure_color[1] * 0.75,
                        original_color[2] * 0.25 + pure_color[2] * 0.75
                    ]
                    
                    # Calculate brushstroke direction (perpendicular to gradient)
                    brush_direction = direction + np.pi/2
                    brush_dx = np.cos(brush_direction) * brush_length
                    brush_dy = np.sin(brush_direction) * brush_length
                    
                    # Draw brushstroke line
                    steps = max(abs(int(brush_dx)), abs(int(brush_dy)), 1)
                    
                    for step in range(steps):
                        # Calculate position along the brushstroke
                        t = step / steps
                        brush_y = int(y + brush_width//2 + t * brush_dy)
                        brush_x = int(x + brush_width//2 + t * brush_dx)
                        
                        # Ensure we're within bounds
                        if 0 <= brush_y < height and 0 <= brush_x < width:
                            # Paint with brush width
                            brush_radius = brush_width // 2
                            
                            for dy in range(-brush_radius, brush_radius + 1):
                                for dx in range(-brush_radius, brush_radius + 1):
                                    py = brush_y + dy
                                    px = brush_x + dx
                                    
                                    if 0 <= py < height and 0 <= px < width:
                                        # Distance from brush center
                                        dist = np.sqrt(dy*dy + dx*dx)
                                        if dist <= brush_radius:
                                            # Alpha blend the blended color
                                            for c in range(3):  # RGB channels
                                                painted_array[py, px, c] = (
                                                    painted_array[py, px, c] * (1 - alpha) + 
                                                    blended_color[c] * alpha
                                                )
        
        # Convert back to RGB (remove alpha channel for display)
        painted_rgb = painted_array[:, :, :3].astype(np.uint8)
        self.painted_image = Image.fromarray(painted_rgb)
        
        # Display the painted result directly in the GUI
        self.display_brushstroke_result()
        
        self.status_label.configure(text="RGB brushstroke painting complete!")
    
    def display_brushstroke_result(self):
        """Display the brushstroke result in the main GUI"""
        if not self.painted_image:
            return
        
        # Create thumbnail that fits in the display area (about 600x400)
        painted_copy = self.painted_image.copy()
        painted_copy.thumbnail((600, 400), Image.Resampling.LANCZOS)
        
        # Display the image in the brushstroke label
        painted_photo = ImageTk.PhotoImage(painted_copy)
        self.brushstroke_label.configure(image=painted_photo, text="")
        self.brushstroke_label.image = painted_photo  # Keep reference
    
    def save_painted_image(self):
        """Save the painted image to file"""
        if not self.painted_image:
            return
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Painted Image",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filename:
            try:
                self.painted_image.save(filename)
                messagebox.showinfo("Success", f"Painted image saved as: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")

def main():
    root = tk.Tk()
    app = ImagePainter(root)
    root.mainloop()

if __name__ == "__main__":
    main()
