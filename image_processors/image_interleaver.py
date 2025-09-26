#!/usr/bin/env python3
"""
Image Interleaver - Combines 4 images by interleaving chunks in a user-defined pattern
Each chunk in the output is divided into 4 quadrants, each containing a sub-chunk from one of the 4 input images:
- Top-left quadrant: Image 1
- Top-right quadrant: Image 2  
- Bottom-left quadrant: Image 3
- Bottom-right quadrant: Image 4

For example, with 32x32 chunks, each quadrant would be 16x16 pixels.
This version uses PIL only, no numpy dependency required.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os

class ImageInterleaver:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Interleaver - Configurable Chunk Combiner")
        self.root.geometry("1400x900")
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Store selected images
        self.image_paths = [None, None, None, None]
        self.images = [None, None, None, None]
        self.preview_images = [None, None, None, None]
        
        # Default chunk size
        self.chunk_size = tk.IntVar(value=16)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="Image Interleaver", 
                              font=("Arial", 24, "bold"))
        title_label.pack(pady=15)
        
        # Chunk size selection
        chunk_frame = ttk.Frame(self.root)
        chunk_frame.pack(pady=15)
        
        chunk_label = tk.Label(chunk_frame, text="Chunk Size:", font=("Arial", 14, "bold"))
        chunk_label.pack(side=tk.LEFT, padx=8)
        
        chunk_sizes = ["2", "4", "8", "16", "32", "64", "128"]
        chunk_combo = ttk.Combobox(chunk_frame, textvariable=self.chunk_size, 
                                  values=chunk_sizes, state="readonly", width=12, font=("Arial", 12))
        chunk_combo.pack(side=tk.LEFT, padx=8)
        
        chunk_info = tk.Label(chunk_frame, 
                             text="(Each quadrant will be chunk_size/2 × chunk_size/2 pixels)",
                             font=("Arial", 11, "italic"))
        chunk_info.pack(side=tk.LEFT, padx=15)
        
        # Frame for image selection buttons and previews
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=40, pady=20)
        
        # Create 2x2 grid for image selection
        self.image_frames = []
        self.image_labels = []
        self.select_buttons = []
        self.preview_labels = []
        
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        labels = ["Image 1 (Top-Left Quadrant)", "Image 2 (Top-Right Quadrant)", 
                 "Image 3 (Bottom-Left Quadrant)", "Image 4 (Bottom-Right Quadrant)"]
        
        for i, ((row, col), label) in enumerate(zip(positions, labels)):
            frame = ttk.LabelFrame(main_frame, text=label, padding=20)
            frame.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")
            
            # Configure grid weights for centering
            main_frame.grid_rowconfigure(row, weight=1)
            main_frame.grid_columnconfigure(col, weight=1)
            
            # Select button
            btn = ttk.Button(frame, text=f"Select Image {i+1}",
                           command=lambda idx=i: self.select_image(idx))
            btn.configure(width=20)  # Make buttons wider
            btn.pack(pady=8)
            
            # Preview label
            preview = tk.Label(frame, text="No image selected", 
                             width=35, height=18, bg="lightgray", font=("Arial", 10))
            preview.pack(pady=10)
            
            # File path label
            path_label = tk.Label(frame, text="", wraplength=250, 
                                font=("Arial", 10))
            path_label.pack()
            
            self.image_frames.append(frame)
            self.select_buttons.append(btn)
            self.preview_labels.append(preview)
            self.image_labels.append(path_label)
        
        # Control buttons frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=25)
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="Combine Images",
                                    command=self.process_images, state="disabled")
        self.process_btn.configure(width=20, padding=(10, 5))  # Make button larger
        self.process_btn.pack(side=tk.LEFT, padx=15)
        
        # Clear button
        clear_btn = ttk.Button(control_frame, text="Clear All",
                              command=self.clear_all)
        clear_btn.configure(width=15, padding=(10, 5))  # Make button larger
        clear_btn.pack(side=tk.LEFT, padx=15)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Select 4 images to begin",
                                   font=("Arial", 12))
        self.status_label.pack(pady=15)
        
    def select_image(self, index):
        """Select an image file for the given index"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title=f"Select Image {index + 1}",
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load and validate image
                img = Image.open(filename)
                self.images[index] = img
                self.image_paths[index] = filename
                
                # Update UI
                self.update_preview(index, img, filename)
                self.update_status()
                
                # Check if we can enable processing
                self.check_ready_to_process()
                
            except Exception as e:
                messagebox.showerror(f"Error loading image {index + 1}",
                                   f"Could not load image: {str(e)}")
    
    def update_preview(self, index, img, filename):
        """Update the preview for the selected image"""
        # Create thumbnail for preview
        img_copy = img.copy()
        img_copy.thumbnail((300, 300), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img_copy)
        self.preview_images[index] = photo  # Keep reference
        
        # Update preview label
        self.preview_labels[index].configure(image=photo, text="")
        
        # Update file path label
        basename = os.path.basename(filename)
        if len(basename) > 25:
            basename = basename[:22] + "..."
        self.image_labels[index].configure(text=basename)
        
        # Update button text
        self.select_buttons[index].configure(text=f"Change Image {index + 1}")
    
    def update_status(self):
        """Update the status message"""
        selected_count = sum(1 for img in self.images if img is not None)
        
        if selected_count == 0:
            self.status_label.configure(text="Select 4 images to begin")
        elif selected_count < 4:
            self.status_label.configure(text=f"{selected_count}/4 images selected")
        else:
            # Check dimensions
            if self.check_dimensions():
                dimensions = self.images[0].size
                chunk_size = self.chunk_size.get()
                sub_chunk = chunk_size // 2
                self.status_label.configure(
                    text=f"Ready! Output: {dimensions[0]}x{dimensions[1]} pixels, {chunk_size}x{chunk_size} chunks ({sub_chunk}x{sub_chunk} per quadrant)"
                )
            else:
                self.status_label.configure(
                    text="Error: All images must have the same dimensions"
                )
    
    def check_dimensions(self):
        """Check if all selected images have the same dimensions"""
        if not all(img is not None for img in self.images):
            return False
        
        first_size = self.images[0].size
        return all(img.size == first_size for img in self.images)
    
    def check_ready_to_process(self):
        """Enable/disable the process button based on readiness"""
        ready = (all(img is not None for img in self.images) and 
                self.check_dimensions())
        
        self.process_btn.configure(state="normal" if ready else "disabled")
    
    def process_images(self):
        """Combine the 4 images using configurable chunk interleaving"""
        if not all(img is not None for img in self.images):
            messagebox.showerror("Error", "Please select all 4 images")
            return
        
        if not self.check_dimensions():
            messagebox.showerror("Error", "All images must have the same dimensions")
            return
        
        try:
            self.status_label.configure(text="Processing images...")
            self.root.update()
            
            # Get chunk size
            chunk_size = self.chunk_size.get()
            if chunk_size < 2 or chunk_size % 2 != 0:
                messagebox.showerror("Error", "Chunk size must be an even number >= 2")
                return
            
            # Convert images to RGB mode
            rgb_images = []
            for img in self.images:
                if img.mode != 'RGB':
                    rgb_images.append(img.convert('RGB'))
                else:
                    rgb_images.append(img)
            
            width, height = rgb_images[0].size
            
            # Create output image
            output = Image.new('RGB', (width, height))
            
            # Calculate sub-chunk size (each quadrant size)
            sub_chunk = chunk_size // 2
            
            # Interleave chunks using PIL crop and paste
            for y in range(0, height, chunk_size):
                for x in range(0, width, chunk_size):
                    # Calculate actual chunk boundaries (handle edge cases)
                    y_end = min(y + chunk_size, height)
                    x_end = min(x + chunk_size, width)
                    
                    # Calculate sub-chunk boundaries
                    y_mid = min(y + sub_chunk, height)
                    x_mid = min(x + sub_chunk, width)
                    
                    # Top-left quadrant from image 1
                    if y < y_mid and x < x_mid:
                        region = rgb_images[0].crop((x, y, x_mid, y_mid))
                        output.paste(region, (x, y))
                    
                    # Top-right quadrant from image 2
                    if y < y_mid and x_mid < x_end:
                        region = rgb_images[1].crop((x_mid, y, x_end, y_mid))
                        output.paste(region, (x_mid, y))
                    
                    # Bottom-left quadrant from image 3
                    if y_mid < y_end and x < x_mid:
                        region = rgb_images[2].crop((x, y_mid, x_mid, y_end))
                        output.paste(region, (x, y_mid))
                    
                    # Bottom-right quadrant from image 4
                    if y_mid < y_end and x_mid < x_end:
                        region = rgb_images[3].crop((x_mid, y_mid, x_end, y_end))
                        output.paste(region, (x_mid, y_mid))
            
            # Ask user where to save
            save_path = filedialog.asksaveasfilename(
                title="Save Combined Image",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if save_path:
                output.save(save_path)
                self.status_label.configure(text=f"Image saved: {os.path.basename(save_path)}")
                messagebox.showinfo("Success", f"Combined image saved as:\n{save_path}")
            else:
                self.status_label.configure(text="Save cancelled")
                
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error combining images: {str(e)}")
            self.status_label.configure(text="Error occurred during processing")
    
    def clear_all(self):
        """Clear all selected images"""
        self.images = [None, None, None, None]
        self.image_paths = [None, None, None, None]
        self.preview_images = [None, None, None, None]
        
        for i in range(4):
            self.preview_labels[i].configure(image="", text="No image selected")
            self.image_labels[i].configure(text="")
            self.select_buttons[i].configure(text=f"Select Image {i+1}")
        
        self.process_btn.configure(state="disabled")
        self.status_label.configure(text="Select 4 images to begin")

def main():
    root = tk.Tk()
    app = ImageInterleaver(root)
    root.mainloop()

if __name__ == "__main__":
    main()
