#!/usr/bin/env python3
"""
Quick test to check PIL/ImageTk thumbnail display
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

def test_thumbnail():
    root = tk.Tk()
    root.title("Thumbnail Test")
    root.geometry("600x600")
    
    def load_and_show():
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
        )
        
        if filename:
            try:
                # Load image
                original = Image.open(filename)
                print(f"Original image size: {original.size}")
                
                # Convert to RGB if needed
                if original.mode != 'RGB':
                    original = original.convert('RGB')
                
                # Create thumbnail
                thumbnail = original.copy()
                thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
                print(f"Thumbnail size: {thumbnail.size}")
                
                # Display
                photo = ImageTk.PhotoImage(thumbnail)
                label.configure(image=photo, text="")
                label.image = photo  # Keep reference
                
                status.configure(text=f"Loaded: {os.path.basename(filename)} - Original: {original.size}, Thumbnail: {thumbnail.size}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
                print(f"Error: {e}")
    
    # UI
    btn = tk.Button(root, text="Load Image", command=load_and_show)
    btn.pack(pady=20)
    
    label = tk.Label(root, text="No image loaded", bg="lightgray")
    label.pack(padx=20, pady=20, expand=True, fill='both')
    
    status = tk.Label(root, text="Click Load Image", font=("Arial", 10))
    status.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    test_thumbnail()
