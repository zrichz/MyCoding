#!/usr/bin/env python3
"""
Quick test to isolate the thumbnail display issue
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from pathlib import Path

class ThumbnailTest:
    def __init__(self, root):
        self.root = root
        self.root.title("Thumbnail Display Test")
        self.root.geometry("900x700")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Add button to select image
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(btn_frame, text="Load Test Image", command=self.load_test_image).pack(side="left")
        ttk.Button(btn_frame, text="Select Image File", command=self.select_image).pack(side="left", padx=(10, 0))
        
        # Create canvas
        self.canvas = tk.Canvas(main_frame, bg="white", width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        
        # Status label
        self.status = tk.StringVar(value="Ready to load image")
        ttk.Label(main_frame, textvariable=self.status).pack(pady=(10, 0))
        
    def load_test_image(self):
        """Load a test image from test_images directory"""
        test_image_path = Path("test_images/test_small.jpg")
        if test_image_path.exists():
            self.load_and_display_image(test_image_path)
        else:
            self.status.set("Test image not found")
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif")]
        )
        if file_path:
            self.load_and_display_image(Path(file_path))
    
    def load_and_display_image(self, image_path):
        """Load and display an image"""
        try:
            self.status.set(f"Loading: {image_path.name}")
            
            # Load original image
            image = Image.open(image_path)
            self.status.set(f"Loaded: {image.size[0]}x{image.size[1]}")
            
            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((800, 800), Image.Resampling.LANCZOS)
            self.status.set(f"Thumbnail: {thumbnail.size[0]}x{thumbnail.size[1]}")
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(thumbnail)
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Get canvas size
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Center image
            x = max(0, (canvas_width - thumbnail.width) // 2)
            y = max(0, (canvas_height - thumbnail.height) // 2)
            
            # Place image
            self.canvas.create_image(x, y, anchor="nw", image=self.photo)
            
            self.status.set(f"Displayed at ({x}, {y})")
            
        except Exception as e:
            self.status.set(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThumbnailTest(root)
    root.mainloop()
