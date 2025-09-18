import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Target dimensions (2:1 ratio)
W, H = 512, 256

def select_and_prepare_image():
    """
    Open a file dialog to select an image, then rescale/crop it to the target dimensions.
    """
    # Hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        # User cancelled, create default synthetic image
        print("No file selected. Using default synthetic image.")
        base = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, W//2, H], fill='lightblue')
        draw.rectangle([W//2, 0, W, H], fill='lightgreen')
        return base
    
    try:
        # Load the image
        img = Image.open(file_path)
        print(f"Loaded image: {os.path.basename(file_path)} ({img.size[0]}x{img.size[1]})")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate scaling to fit target dimensions while maintaining aspect ratio
        original_w, original_h = img.size
        target_ratio = W / H  # 2:1
        original_ratio = original_w / original_h
        
        if original_ratio > target_ratio:
            # Image is wider than target ratio - fit by height and crop width
            scale_factor = H / original_h
            new_w = int(original_w * scale_factor)
            new_h = H
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Crop to target width (center crop)
            crop_x = (new_w - W) // 2
            img_final = img_scaled.crop((crop_x, 0, crop_x + W, H))
            
        else:
            # Image is taller than target ratio - fit by width and crop height
            scale_factor = W / original_w
            new_w = W
            new_h = int(original_h * scale_factor)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Crop to target height (center crop)
            crop_y = (new_h - H) // 2
            img_final = img_scaled.crop((0, crop_y, W, crop_y + H))
        
        print(f"Processed image to {W}x{H} (2:1 ratio)")
        return img_final
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        print(f"Error loading image: {e}")
        # Fall back to synthetic image
        base = Image.new('RGB', (W, H), 'white')
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, W//2, H], fill='lightblue')
        draw.rectangle([W//2, 0, W, H], fill='lightgreen')
        return base
    
    finally:
        root.destroy()

# 1. Get base image from user selection or create default
base_img = select_and_prepare_image()

# 2. Create a larger canvas (50% larger) maintaining 2:1 ratio with mid-grey background and gridlines
CANVAS_W = int(W * 1.5)
CANVAS_H = int(H * 1.5)

# Ensure canvas maintains 2:1 ratio
if CANVAS_W / CANVAS_H != 2.0:
    CANVAS_H = CANVAS_W // 2  # Force 2:1 ratio

canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), color=(128, 128, 128))  # Mid-grey background

# Add faint gridlines to show rotation effects
draw = ImageDraw.Draw(canvas)
grid_color = (110, 110, 110)  # Slightly darker grey for grid
grid_spacing = 32  # Grid every 32 pixels

# Draw vertical gridlines
for x in range(0, CANVAS_W, grid_spacing):
    draw.line([(x, 0), (x, CANVAS_H)], fill=grid_color, width=1)

# Draw horizontal gridlines  
for y in range(0, CANVAS_H, grid_spacing):
    draw.line([(0, y), (CANVAS_W, y)], fill=grid_color, width=1)

# Center the original image on the larger canvas
offset_x = (CANVAS_W - W) // 2
offset_y = (CANVAS_H - H) // 2
canvas.paste(base_img, (offset_x, offset_y))

print(f"Original image ({W}x{H}) placed on larger canvas ({CANVAS_W}x{CANVAS_H}) with 2:1 ratio")
print(f"Canvas offset: ({offset_x}, {offset_y})")
print(f"Grid spacing: {grid_spacing} pixels")

def recursive_split(img, x, y, w, h, depth, angle):
    """
    Recursively split a 2:1 rectangle into two squares, rotate each around the shared hinge,
    then recurse into each square.
    """
    if depth == 0:
        return

    # Calculate square size (height == square width)
    sq = h

    # Define boxes for left/right squares
    left_box  = (x,      y, x+sq, y+sq)
    right_box = (x+sq,   y, x+2*sq, y+sq)

    # Crop and rotate around hinge points at the bottom corners
    # The hinge points are relative to the cropped image coordinates, not global canvas
    # Left square rotates CCW around its bottom right corner
    # Right square rotates CW around its bottom left corner
    
    # For cropped images, the hinge is relative to the crop, not the canvas
    left_hinge_local = (sq, sq)    # Bottom right of left square (relative to cropped image)
    right_hinge_local = (0, sq)    # Bottom left of right square (relative to cropped image)
    
    # Debug info for first level
    if depth == 3:
        print(f"Level {depth}: Left box: {left_box}, Right box: {right_box}")
        print(f"Level {depth}: Left hinge: {left_hinge_local}, Right hinge: {right_hinge_local}")
        print(f"Level {depth}: Angle: {angle}")
    
    left_img  = img.crop(left_box).rotate(angle, center=left_hinge_local, expand=False)   # CCW rotation
    right_img = img.crop(right_box).rotate(-angle, center=right_hinge_local, expand=False) # CW rotation

    # Paste back into place - the rotated images go back to their original positions
    # Note: The rotation might cause parts to extend beyond the original box bounds
    img.paste(left_img,  (x, y))
    img.paste(right_img, (x+sq, y))

    # Recurse, decaying the angle
    recursive_split(img, x,    y,    sq, sq, depth-1, angle*0.9)
    recursive_split(img, x+sq, y,    sq, sq, depth-1, -angle*0.9)

# 3. Apply recursion to the original image area on the larger canvas
# Any rotated parts that exceed the original bounds will show in the expanded canvas
out = canvas.copy()
recursive_split(out, offset_x, offset_y, W, H, depth=3, angle=15)

# 3.5. Restore background grid on top so we can see what's happening
draw_final = ImageDraw.Draw(out)
for x in range(0, CANVAS_W, grid_spacing):
    draw_final.line([(x, 0), (x, CANVAS_H)], fill=grid_color, width=1)
for y in range(0, CANVAS_H, grid_spacing):
    draw_final.line([(0, y), (CANVAS_W, y)], fill=grid_color, width=1)

# 4. Display the result
plt.figure(figsize=(16,12))
plt.imshow(out)
plt.axis('off')
plt.title(f'Recursive Hinged Rotation - Full Canvas ({CANVAS_W}x{CANVAS_H})')
plt.show()

# Also save the result
output_filename = 'iterative_art_result.png'
out.save(output_filename)
print(f"Result saved as: {output_filename}")

