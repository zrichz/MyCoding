#!/usr/bin/env python3
"""
Test crop selection functionality
"""

import tkinter as tk
from PIL import Image, ImageTk
import sys
import os

def test_crop_selection():
    """Test the crop selection on a simple canvas"""
    
    root = tk.Tk()
    root.title("Crop Selection Test")
    root.geometry("600x500")
    
    # Test variables
    crop_start_x = None
    crop_start_y = None
    crop_end_x = None
    crop_end_y = None
    crop_rectangle = None
    is_selecting = False
    
    # Create canvas
    canvas = tk.Canvas(root, bg="white", width=500, height=400)
    canvas.pack(padx=20, pady=20)
    
    # Load a test image
    try:
        img = Image.open("test_images/test_small.jpg")
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        
        # Place image at center
        x_offset = (500 - img.width) // 2
        y_offset = (400 - img.height) // 2
        
        canvas.create_image(x_offset, y_offset, anchor="nw", image=photo, tags="image")
        
        def start_selection(event):
            nonlocal crop_start_x, crop_start_y, is_selecting, crop_rectangle
            
            x, y = event.x, event.y
            if (x_offset <= x <= x_offset + img.width and 
                y_offset <= y <= y_offset + img.height):
                
                is_selecting = True
                crop_start_x = x
                crop_start_y = y
                
                if crop_rectangle:
                    canvas.delete(crop_rectangle)
                    crop_rectangle = None
                
                print(f"Selection started at: ({x}, {y})")
        
        def update_selection(event):
            nonlocal crop_end_x, crop_end_y, crop_rectangle
            
            if not is_selecting or crop_start_x is None or crop_start_y is None:
                return
            
            x = max(x_offset, min(event.x, x_offset + img.width))
            y = max(y_offset, min(event.y, y_offset + img.height))
            
            crop_end_x = x
            crop_end_y = y
            
            if crop_rectangle:
                canvas.delete(crop_rectangle)
            
            crop_rectangle = canvas.create_rectangle(
                crop_start_x, crop_start_y,
                crop_end_x, crop_end_y,
                outline="red", width=2
            )
        
        def end_selection(event):
            nonlocal is_selecting
            
            if not is_selecting or crop_start_x is None or crop_start_y is None:
                return
            
            is_selecting = False
            
            x = max(x_offset, min(event.x, x_offset + img.width))
            y = max(y_offset, min(event.y, y_offset + img.height))
            
            width = abs(x - crop_start_x)
            height = abs(y - crop_start_y)
            
            print(f"Selection ended at: ({x}, {y})")
            print(f"Selection size: {width} x {height}")
            
            if width > 10 and height > 10:
                print("✅ Valid crop selection!")
            else:
                print("❌ Selection too small")
        
        # Bind events
        canvas.bind("<Button-1>", start_selection)
        canvas.bind("<B1-Motion>", update_selection)
        canvas.bind("<ButtonRelease-1>", end_selection)
        
        # Instructions
        instruction_label = tk.Label(root, text="Click and drag on the image to test crop selection", 
                                   font=("Arial", 12))
        instruction_label.pack(pady=10)
        
        print("✅ Crop selection test ready!")
        print("Click and drag on the image to test crop selection")
        print("Check console for selection coordinates")
        
        # Keep reference to photo to prevent garbage collection
        globals()['_photo_ref'] = photo
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        root.destroy()

if __name__ == "__main__":
    if not os.path.exists("test_images/test_small.jpg"):
        print("❌ Test image not found. Please ensure test_images/test_small.jpg exists.")
        sys.exit(1)
    
    test_crop_selection()
