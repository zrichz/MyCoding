#!/usr/bin/env python3
"""
Minimal GUI test for image display
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path

def test_minimal_gui():
    root = tk.Tk()
    root.title("Minimal Image Display Test")
    root.geometry("600x500")
    
    # Create canvas
    canvas = tk.Canvas(root, bg="lightgray", width=500, height=400)
    canvas.pack(pady=20)
    
    # Load test image
    test_dir = Path('/home/rich/MyCoding/image_processors/test_images')
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if image_files:
        try:
            # Load and display image
            image = Image.open(image_files[0])
            thumbnail = image.copy()
            thumbnail.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumbnail)
            
            # Place image on canvas
            canvas.create_image(50, 50, anchor="nw", image=photo)
            
            # Keep reference to prevent garbage collection
            canvas.image = photo
            
            # Add label
            label = ttk.Label(root, text=f"Test image: {image_files[0].name} ({image.size})")
            label.pack()
            
            print(f"✓ Loaded test image: {image_files[0].name}")
            print(f"✓ Original size: {image.size}")
            print(f"✓ Thumbnail size: {thumbnail.size}")
            print("✓ Image should be visible in the window")
            
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            error_label = ttk.Label(root, text=f"Error: {e}")
            error_label.pack()
    else:
        print("✗ No test images found")
        no_image_label = ttk.Label(root, text="No test images found - run create_test_images.py")
        no_image_label.pack()
    
    # Add instruction
    instruction = ttk.Label(root, text="Close this window to continue")
    instruction.pack(pady=10)
    
    print("\nClose the window when you're done testing...")
    root.mainloop()

if __name__ == "__main__":
    test_minimal_gui()
