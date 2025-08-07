"""
Image Vertical Expander - GUI Application

This script provides a graphical interface for expanding images vertically with customizable effects.

Key Features:
- Load images via file dialog with thumbnail preview (max 800x800 display)
- Expand images vertically by adding pixels at top and bottom edges
- Configurable expansion sizes: 120px, 240px, or 360px on each side
- Progressive horizontal blur applied to expanded regions with selectable intensity:
  * Blur options: 20px, 40px, 80px, or 160px maximum
  * Linear gradient from 0 blur at original edge to maximum at outermost expansion
- Progressive luminance reduction (darkening) for natural fade effect:
  * Luminance drop options: 0%, 25%, 50%, 75%, or 100%
  * Linear gradient from original brightness to maximum reduction
- Real-time preview of processed image
- Save results in common image formats (PNG, JPEG)
- Status feedback showing current settings and processing results

Technical Implementation:
- Uses scipy.ndimage for horizontal-only blur to preserve colors
- Combines blur and luminance effects for realistic expansion transitions
- Maintains original image quality and color fidelity
"""


from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy import ndimage

class ImageExpander:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Vertical Expander")
        self.root.geometry("900x750")
        
        self.original_image = None
        self.processed_image = None
        
        # Default settings
        self.blur_amount = tk.IntVar(value=20)
        self.expansion_size = tk.IntVar(value=120)
        self.luminance_drop = tk.IntVar(value=0)
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # Title
        title_label = tk.Label(main_frame, text="Image Vertical Expander", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Settings frame
        settings_frame = tk.Frame(main_frame)
        settings_frame.pack(pady=(0, 20))
        
        # Blur amount selection
        blur_frame = tk.LabelFrame(settings_frame, text="Horizontal Blur Amount (pixels)", 
                                  font=("Arial", 10, "bold"))
        blur_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Radiobutton(blur_frame, text="20px", variable=self.blur_amount, value=20,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(blur_frame, text="40px", variable=self.blur_amount, value=40,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(blur_frame, text="80px", variable=self.blur_amount, value=80,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(blur_frame, text="160px", variable=self.blur_amount, value=160,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        
        # Expansion size selection
        expansion_frame = tk.LabelFrame(settings_frame, text="Expansion Size (pixels top and bottom)", 
                                       font=("Arial", 10, "bold"))
        expansion_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Radiobutton(expansion_frame, text="120px", variable=self.expansion_size, value=120,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(expansion_frame, text="240px", variable=self.expansion_size, value=240,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(expansion_frame, text="360px", variable=self.expansion_size, value=360,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        
        # Luminance drop selection
        luminance_frame = tk.LabelFrame(settings_frame, text="Luminance Drop (%)", 
                                       font=("Arial", 10, "bold"))
        luminance_frame.pack(side=tk.LEFT)
        
        tk.Radiobutton(luminance_frame, text="0%", variable=self.luminance_drop, value=0,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(luminance_frame, text="25%", variable=self.luminance_drop, value=25,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(luminance_frame, text="50%", variable=self.luminance_drop, value=50,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(luminance_frame, text="75%", variable=self.luminance_drop, value=75,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        tk.Radiobutton(luminance_frame, text="100%", variable=self.luminance_drop, value=100,
                      font=("Arial", 9)).pack(anchor='w', padx=10, pady=2)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(0, 20))
        
        # Load image button
        load_button = tk.Button(button_frame, text="Load Image", 
                               command=self.load_image, bg="#4CAF50", fg="white",
                               font=("Arial", 10), padx=20, pady=10)
        load_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        self.process_button = tk.Button(button_frame, text="Expand Image", 
                                       command=self.process_image, bg="#2196F3", fg="white",
                                       font=("Arial", 10), padx=20, pady=10, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save button
        self.save_button = tk.Button(button_frame, text="Save Result", 
                                    command=self.save_image, bg="#FF9800", fg="white",
                                    font=("Arial", 10), padx=20, pady=10, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)
        
        # Image display frame
        self.image_frame = tk.Frame(main_frame, bg="lightgray", relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(expand=True, fill='both')
        
        # Image label
        self.image_label = tk.Label(self.image_frame, text="No image loaded", 
                                   bg="lightgray", fg="gray", font=("Arial", 12))
        self.image_label.pack(expand=True)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready", 
                                    font=("Arial", 9), fg="gray")
        self.status_label.pack(pady=(10, 0))
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image)
                self.process_button.config(state=tk.NORMAL)
                self.status_label.config(text=f"Loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image(self, image):
        # Create thumbnail for display (max 800x800)
        display_image = image.copy()
        display_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for tkinter
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def apply_horizontal_blur(self, line_array, blur_amount):
        """Apply horizontal-only blur to preserve colors"""
        if blur_amount <= 0:
            return line_array
        
        # Create horizontal Gaussian kernel
        kernel_size = int(blur_amount * 2) * 2 + 1  # Ensure odd size
        sigma = blur_amount / 3.0  # Convert blur_amount to sigma
        
        # Create 1D horizontal Gaussian kernel
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Apply horizontal convolution to each color channel
        if len(line_array.shape) == 2:  # Color image (width, channels)
            blurred = np.zeros_like(line_array, dtype=np.float32)
            for channel in range(line_array.shape[1]):
                # Use mode='nearest' to handle edges properly
                blurred[:, channel] = ndimage.convolve1d(
                    line_array[:, channel].astype(np.float32), 
                    kernel, 
                    axis=0, 
                    mode='nearest'
                )
            return np.clip(blurred, 0, 255).astype(np.uint8)
        else:  # Grayscale
            blurred = ndimage.convolve1d(
                line_array.astype(np.float32), 
                kernel, 
                axis=0, 
                mode='nearest'
            )
            return np.clip(blurred, 0, 255).astype(np.uint8)
    
    def apply_luminance_reduction(self, line_array, luminance_factor):
        """Apply luminance reduction to darken the line"""
        if luminance_factor <= 0:
            return line_array
        
        # Convert to float for calculation
        line_float = line_array.astype(np.float32)
        
        # Apply luminance reduction (multiply by (1 - factor))
        reduction_multiplier = 1.0 - (luminance_factor / 100.0)
        reduced = line_float * reduction_multiplier
        
        return np.clip(reduced, 0, 255).astype(np.uint8)
    
    def process_image(self):
        if not self.original_image:
            return
        
        try:
            self.status_label.config(text="Processing image...")
            self.root.update()
            
            # Convert image to numpy array
            img_array = np.array(self.original_image)
            height, width = img_array.shape[:2]
            
            # Get selected settings
            expansion_pixels = self.expansion_size.get()
            max_blur = self.blur_amount.get()
            max_luminance_drop = self.luminance_drop.get()
            
            # Create expanded image array
            new_height = height + (2 * expansion_pixels)  # expansion_pixels top + expansion_pixels bottom
            if len(img_array.shape) == 3:  # Color image
                expanded_array = np.zeros((new_height, width, img_array.shape[2]), dtype=img_array.dtype)
            else:  # Grayscale
                expanded_array = np.zeros((new_height, width), dtype=img_array.dtype)
            
            # Copy original image to center
            expanded_array[expansion_pixels:expansion_pixels+height] = img_array
            
            # Process top expansion
            top_line = img_array[0]  # First line of original image
            for i in range(expansion_pixels):
                # Calculate blur amount (0 to max_blur pixels)
                blur_amount = (expansion_pixels - 1 - i) * max_blur / (expansion_pixels - 1) if expansion_pixels > 1 else 0
                
                # Calculate luminance reduction (0 to max_luminance_drop %)
                luminance_reduction = (expansion_pixels - 1 - i) * max_luminance_drop / (expansion_pixels - 1) if expansion_pixels > 1 else 0
                
                # Apply horizontal blur to the top line
                blurred_line = self.apply_horizontal_blur(top_line, blur_amount)
                
                # Apply luminance reduction
                final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                expanded_array[i] = final_line
            
            # Process bottom expansion
            bottom_line = img_array[-1]  # Last line of original image
            for i in range(expansion_pixels):
                # Calculate blur amount (0 to max_blur pixels)
                blur_amount = i * max_blur / (expansion_pixels - 1) if expansion_pixels > 1 else 0
                
                # Calculate luminance reduction (0 to max_luminance_drop %)
                luminance_reduction = i * max_luminance_drop / (expansion_pixels - 1) if expansion_pixels > 1 else 0
                
                # Apply horizontal blur to the bottom line
                blurred_line = self.apply_horizontal_blur(bottom_line, blur_amount)
                
                # Apply luminance reduction
                final_line = self.apply_luminance_reduction(blurred_line, luminance_reduction)
                expanded_array[expansion_pixels + height + i] = final_line
            
            # Convert back to PIL Image
            self.processed_image = Image.fromarray(expanded_array.astype('uint8'))
            
            # Display the result
            self.display_image(self.processed_image)
            self.save_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Image expanded successfully! (+{expansion_pixels}px each side, {max_blur}px max blur, {max_luminance_drop}% luminance drop)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
            self.status_label.config(text="Error processing image")
    
    def save_image(self):
        if not self.processed_image:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Expanded Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.processed_image.save(file_path)
                self.status_label.config(text=f"Saved: {file_path}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageExpander()
    app.run()