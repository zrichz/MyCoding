import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Target dimensions for loaded image (max size, maintaining aspect ratio)
MAX_SIZE = 1024

# Canvas dimensions (fixed size for all outputs)
CANVAS_W, CANVAS_H = 1800, 900

def select_and_prepare_image():
    """
    Open a file dialog to select an image, then scale it to fit within MAX_SIZE x MAX_SIZE
    while maintaining aspect ratio.
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
        # User cancelled, create default synthetic 2:1 ratio image
        print("No file selected. Using default 2:1 synthetic image.")
        default_w = MAX_SIZE
        default_h = MAX_SIZE // 2
        base = Image.new('RGB', (default_w, default_h), 'white')
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, default_w//2, default_h], fill='lightblue')
        draw.rectangle([default_w//2, 0, default_w, default_h], fill='lightgreen')
        print(f"Default image dimensions: {default_w}x{default_h} (2:1 ratio)")
        return base
    
    try:
        # Load the image
        img = Image.open(file_path)
        print(f"Loaded image: {os.path.basename(file_path)} ({img.size[0]}x{img.size[1]})")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Process image to ensure 2:1 ratio (width:height) for L-System splitting
        original_w, original_h = img.size
        print(f"Original aspect ratio: {original_w/original_h:.2f}:1")
        
        # Target 2:1 ratio dimensions within MAX_SIZE constraints
        # If height is the limiting factor: h = MAX_SIZE, w = 2 * MAX_SIZE
        # If width is the limiting factor: w = MAX_SIZE, h = MAX_SIZE/2
        if original_w / original_h >= 2.0:
            # Image is already wider than 2:1, height becomes limiting factor
            target_h = min(MAX_SIZE, original_h)
            target_w = target_h * 2
        else:
            # Image is narrower than 2:1, width becomes limiting factor  
            target_w = min(MAX_SIZE, original_w)
            target_h = target_w // 2
        
        # Ensure we don't exceed MAX_SIZE in either dimension
        if target_w > MAX_SIZE:
            target_w = MAX_SIZE
            target_h = MAX_SIZE // 2
        if target_h > MAX_SIZE:
            target_h = MAX_SIZE
            target_w = MAX_SIZE * 2
            
        print(f"Target 2:1 dimensions: {target_w}x{target_h} (ratio: {target_w/target_h:.1f}:1)")
        
        # Scale the image to fit the target dimensions
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        scale_factor = min(scale_w, scale_h)
        
        scaled_w = int(original_w * scale_factor)
        scaled_h = int(original_h * scale_factor)
        img_scaled = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        
        # Create final 2:1 canvas and center the scaled image
        img_final = Image.new('RGB', (target_w, target_h), (128, 128, 128))
        
        # Center the scaled image on the 2:1 canvas
        paste_x = (target_w - scaled_w) // 2
        paste_y = (target_h - scaled_h) // 2
        img_final.paste(img_scaled, (paste_x, paste_y))
        
        print(f"Final image: {target_w}x{target_h} with 2:1 ratio for L-System processing")
        
        return img_final
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        print(f"Error loading image: {e}")
        # Fall back to synthetic 2:1 ratio image
        fallback_w = MAX_SIZE
        fallback_h = MAX_SIZE // 2
        base = Image.new('RGB', (fallback_w, fallback_h), 'white')
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, fallback_w//2, fallback_h], fill='lightblue')
        draw.rectangle([fallback_w//2, 0, fallback_w, fallback_h], fill='lightgreen')
        print(f"Using fallback 2:1 synthetic image: {fallback_w}x{fallback_h}")
        return base
    
    finally:
        root.destroy()

# 1. Get base image from user selection or create default
base_img = select_and_prepare_image()

# Get actual image dimensions
img_w, img_h = base_img.size

# 2. Create fixed-size canvas with mid-grey background and gridlines
canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), color=(128, 128, 128))  # Mid-grey background

# Add gridlines to show rotation effects
draw = ImageDraw.Draw(canvas)
grid_color = (10, 10, 110)  # Dark blue gridlines
grid_spacing = 32  # Grid every 32 pixels
for x in range(0, CANVAS_W, grid_spacing):
    draw.line([(x, 0), (x, CANVAS_H)], fill=grid_color, width=1)
for y in range(0, CANVAS_H, grid_spacing):
    draw.line([(0, y), (CANVAS_W, y)], fill=grid_color, width=1)

# Center the original image on the fixed canvas
offset_x = (CANVAS_W - img_w) // 2
offset_y = (CANVAS_H - img_h) // 2
canvas.paste(base_img, (offset_x, offset_y))

print(f"Original image ({img_w}x{img_h}) placed on fixed canvas ({CANVAS_W}x{CANVAS_H})")
print(f"Canvas offset: ({offset_x}, {offset_y})")
print(f"Grid spacing: {grid_spacing} pixels")

def recursive_split_advanced(img, x, y, w, h, depth, angle, cumulative_angle=0):
    """
    Advanced recursive split with complete spatial isolation.
    Each piece gets its own coordinate space to prevent overwrites.
    Uses separate buffer canvases for each branch to eliminate interference.
    """
    if depth == 0:
        return img

    import math
    import numpy as np
    
    # For L-System fern patterns, we need 2:1 rectangles split into squares
    # Calculate square size (height of the rectangle)
    sq = h

    # Define boxes for left/right squares from 2:1 rectangle
    left_box  = (x, y, x+sq, y+sq)        # Left square
    right_box = (x+sq, y, x+w, y+sq)      # Right square (using full width w)

    # The hinge point where pieces connect (shared edge center)
    hinge_x = x + sq
    hinge_y = y + sq // 2

    # Calculate cumulative angles for unfurling effect
    left_total_angle = cumulative_angle + angle      # Unfurl counterclockwise
    right_total_angle = cumulative_angle - angle     # Unfurl clockwise

    print(f"Level {depth}: Processing area ({x},{y}) {w}x{h}")
    print(f"Level {depth}: Hinge at ({hinge_x},{hinge_y})")
    print(f"Level {depth}: Left angle: {left_total_angle}°, Right angle: {right_total_angle}°")

    # ============================================================================
    # STEP 1: EXTRACT PIECES INTO SEPARATE BUFFERS
    # ============================================================================
    
    # Extract pieces from the current image state
    left_piece = img.crop(left_box)
    right_piece = img.crop(right_box)
    
    # Create completely separate working canvases for each piece
    # Make them large enough to handle any rotation without clipping
    buffer_size = max(CANVAS_W, CANVAS_H)
    left_buffer = Image.new('RGB', (buffer_size, buffer_size), (128, 128, 128))
    right_buffer = Image.new('RGB', (buffer_size, buffer_size), (128, 128, 128))
    
    # Calculate buffer centers
    buffer_center = buffer_size // 2
    
    # Place pieces in buffers - position them so their hinge edges align with buffer center
    # Left piece: its right edge (where it connects) should be at buffer center
    left_buffer.paste(left_piece, (buffer_center - sq, buffer_center - sq//2))
    
    # Right piece: its left edge (where it connects) should be at buffer center  
    right_buffer.paste(right_piece, (buffer_center, buffer_center - sq//2))

    # ============================================================================
    # STEP 2: ROTATE PIECES AROUND HINGE POINTS IN ISOLATION
    # ============================================================================
    
    # Left piece rotates around its right edge center (the hinge point)
    left_hinge_in_buffer = (buffer_center, buffer_center)
    left_rotated_buffer = left_buffer.rotate(left_total_angle, center=left_hinge_in_buffer, expand=False)
    
    # Right piece rotates around its left edge center (the hinge point)
    right_hinge_in_buffer = (buffer_center, buffer_center)
    right_rotated_buffer = right_buffer.rotate(right_total_angle, center=right_hinge_in_buffer, expand=False)
    
    # ============================================================================
    # STEP 3: PROCESS CHILDREN IN COMPLETE ISOLATION
    # ============================================================================
    
    if depth > 1:
        print(f"  -> Processing left branch in isolation at depth {depth-1}")
        # Process left piece completely in its own buffer
        # Each square gets subdivided into a 2:1 rectangle for true fern branching
        # Left square (sq x sq) becomes (sq x sq/2) 2:1 rectangle at top of square
        left_rect_h = sq // 2  # Half height for 2:1 ratio
        left_final_buffer = recursive_split_advanced(
            left_rotated_buffer, 
            buffer_center - sq, buffer_center - sq//2, sq, left_rect_h,
            depth-1, angle, left_total_angle
        )
        
        print(f"  -> Processing right branch in isolation at depth {depth-1}")
        # Process right piece completely in its own buffer  
        # Right square (sq x sq) becomes (sq x sq/2) 2:1 rectangle at top of square
        right_rect_h = sq // 2  # Half height for 2:1 ratio
        right_final_buffer = recursive_split_advanced(
            right_rotated_buffer,
            buffer_center, buffer_center - sq//2, sq, right_rect_h, 
            depth-1, angle, right_total_angle
        )
    else:
        left_final_buffer = left_rotated_buffer
        right_final_buffer = right_rotated_buffer

    # ============================================================================
    # STEP 4: COMPOSITE BACK TO MAIN IMAGE
    # ============================================================================
    
    # Create result image as copy of input
    result_img = img.copy()
    
    # Clear the original area with debug outline
    draw_result = ImageDraw.Draw(result_img)
    outline_color = (255, max(50, 255 - depth * 40), max(50, 255 - depth * 40))
    draw_result.rectangle([x, y, x+w, y+h], outline=outline_color, width=2)
    draw_result.rectangle([x, y, x+w, y+h], fill=(128, 128, 128))
    
    # Calculate where to place the processed buffers back onto the main canvas
    # The hinge point should align with buffer_center in each buffer
    
    # Position buffer content back to original hinge point
    left_paste_x = hinge_x - buffer_center
    left_paste_y = hinge_y - buffer_center
    
    right_paste_x = hinge_x - buffer_center
    right_paste_y = hinge_y - buffer_center
    
    # Create masks to identify non-background pixels for clean compositing
    def create_content_mask(buffer):
        """Create a mask of non-background (non-gray) pixels"""
        buffer_array = np.array(buffer)
        # Background is (128, 128, 128)
        mask = ~((buffer_array[:,:,0] == 128) & 
                (buffer_array[:,:,1] == 128) & 
                (buffer_array[:,:,2] == 128))
        return Image.fromarray((mask * 255).astype('uint8')).convert('L')
    
    left_mask = create_content_mask(left_final_buffer)
    right_mask = create_content_mask(right_final_buffer)
    
    # Paste using masks to avoid overwriting and ensure clean compositing
    try:
        # Only paste if within canvas bounds
        if (left_paste_x > -buffer_size and left_paste_y > -buffer_size and 
            left_paste_x < CANVAS_W and left_paste_y < CANVAS_H):
            result_img.paste(left_final_buffer, (left_paste_x, left_paste_y), left_mask)
        
        if (right_paste_x > -buffer_size and right_paste_y > -buffer_size and 
            right_paste_x < CANVAS_W and right_paste_y < CANVAS_H):
            result_img.paste(right_final_buffer, (right_paste_x, right_paste_y), right_mask)
            
    except Exception as e:
        print(f"Compositing error at depth {depth}: {e}")
    
    # ============================================================================
    # STEP 5: ADD DEBUG OUTLINES AROUND ROTATED SECTIONS
    # ============================================================================
    
    # Draw debug outlines around the rotated and composited sections
    draw_final = ImageDraw.Draw(result_img)
    
    # Create depth-based color gradient for easy identification
    # Red for deeper levels, yellow for shallower levels
    outline_r = 255
    outline_g = max(50, min(255, 50 + depth * 60))  # 50 to 230 range
    outline_b = max(50, min(255, 50 + depth * 40))  # 50 to 170 range
    section_outline_color = (outline_r, outline_g, outline_b)
    
    # Draw outlines around the actual pasted content areas
    try:
        # Left section outline - draw around the actual rotated content
        if (left_paste_x > -buffer_size and left_paste_y > -buffer_size and 
            left_paste_x < CANVAS_W and left_paste_y < CANVAS_H):
            
            # Find the bounding box of actual content in the left buffer
            left_bbox = left_final_buffer.getbbox()
            if left_bbox:
                left_content_x1 = left_paste_x + left_bbox[0]
                left_content_y1 = left_paste_y + left_bbox[1] 
                left_content_x2 = left_paste_x + left_bbox[2]
                left_content_y2 = left_paste_y + left_bbox[3]
                
                # Draw outline around left rotated section
                draw_final.rectangle([left_content_x1, left_content_y1, 
                                    left_content_x2, left_content_y2], 
                                   outline=section_outline_color, width=2)
                
                # Add depth label for debugging
                draw_final.text((left_content_x1 + 5, left_content_y1 + 5), 
                              f"L{depth}", fill=section_outline_color)
        
        # Right section outline - draw around the actual rotated content
        if (right_paste_x > -buffer_size and right_paste_y > -buffer_size and 
            right_paste_x < CANVAS_W and right_paste_y < CANVAS_H):
            
            # Find the bounding box of actual content in the right buffer
            right_bbox = right_final_buffer.getbbox()
            if right_bbox:
                right_content_x1 = right_paste_x + right_bbox[0]
                right_content_y1 = right_paste_y + right_bbox[1]
                right_content_x2 = right_paste_x + right_bbox[2] 
                right_content_y2 = right_paste_y + right_bbox[3]
                
                # Draw outline around right rotated section
                draw_final.rectangle([right_content_x1, right_content_y1,
                                    right_content_x2, right_content_y2],
                                   outline=section_outline_color, width=2)
                
                # Add depth label for debugging
                draw_final.text((right_content_x1 + 5, right_content_y1 + 5), 
                              f"R{depth}", fill=section_outline_color)
        
        # Draw hinge point marker for debugging
        hinge_marker_size = 8
        draw_final.ellipse([hinge_x - hinge_marker_size, hinge_y - hinge_marker_size,
                           hinge_x + hinge_marker_size, hinge_y + hinge_marker_size], 
                          fill=(255, 0, 255), outline=(255, 255, 255), width=2)
        
        # Draw rotation angle indicators as lines from hinge
        import math
        line_length = 30
        
        # Left rotation indicator
        rad_left = math.radians(left_total_angle)
        left_end_x = hinge_x + int(line_length * math.cos(rad_left))
        left_end_y = hinge_y + int(line_length * math.sin(rad_left))
        draw_final.line([(hinge_x, hinge_y), (left_end_x, left_end_y)], 
                       fill=(0, 255, 0), width=3)  # Green for left
        
        # Right rotation indicator  
        rad_right = math.radians(right_total_angle)
        right_end_x = hinge_x + int(line_length * math.cos(rad_right))
        right_end_y = hinge_y + int(line_length * math.sin(rad_right))
        draw_final.line([(hinge_x, hinge_y), (right_end_x, right_end_y)], 
                       fill=(255, 100, 0), width=3)  # Orange for right
        
        print(f"Level {depth}: Added debug outlines, hinge marker, and rotation indicators at ({hinge_x},{hinge_y})")
                          
    except Exception as e:
        print(f"Debug outline error at depth {depth}: {e}")
    
    return result_img

def recursive_split(img, x, y, w, h, depth, angle, cumulative_angle=0):
    """
    Recursively split a 2:1 rectangle into two squares and rotate each around
    their connection point to create an unfurling fern-like L-System pattern.
    Each branch accumulates rotation from its parent for natural spiraling.
    Uses isolated processing to prevent overwrites.
    """
    if depth == 0:
        return img

    import math
    
    # Calculate square size (height == square width)
    sq = h

    # Define boxes for left/right squares
    left_box  = (x,      y, x+sq, y+sq)
    right_box = (x+sq,   y, x+2*sq, y+sq)

    # The connection point where the two squares meet (shared edge center)
    # This is the "hinge" around which each piece unfurls
    hinge_x = x + sq
    hinge_y = y + sq // 2

    # Calculate cumulative angles for unfurling effect
    # Each branch builds upon its parent's rotation, creating natural spiraling
    left_total_angle = cumulative_angle + angle      # Unfurl counterclockwise
    right_total_angle = cumulative_angle - angle     # Unfurl clockwise (opposite direction)

    # Debug info for tracking the process
    print(f"Level {depth}: Processing area ({x},{y}) {w}x{h}")
    print(f"Level {depth}: Left angle: {left_total_angle}°, Right angle: {right_total_angle}°")

    # Crop the squares from the CURRENT state of the image
    left_crop = img.crop(left_box)
    right_crop = img.crop(right_box)

    # For proper fern-like unfurling, each piece rotates around its connection edge
    # Left square rotates around its right edge center (where it connects to right square)
    left_rotated = left_crop.rotate(left_total_angle, center=(sq, sq//2), expand=True)
    
    # Right square rotates around its left edge center (where it connects to left square)  
    right_rotated = right_crop.rotate(right_total_angle, center=(0, sq//2), expand=True)

    # Calculate new positions using trigonometry to maintain connection at hinge
    rad_left = math.radians(left_total_angle)
    rad_right = math.radians(right_total_angle)
    
    # For left piece: it rotates around its right edge, so we calculate where that edge goes
    left_offset_x = hinge_x - int(sq * math.cos(rad_left) - (sq//2) * math.sin(rad_left))
    left_offset_y = hinge_y - int(sq * math.sin(rad_left) + (sq//2) * math.cos(rad_left))
    
    # For right piece: it rotates around its left edge  
    right_offset_x = hinge_x - int(0 * math.cos(rad_right) - (sq//2) * math.sin(rad_right))
    right_offset_y = hinge_y - int(0 * math.sin(rad_right) + (sq//2) * math.cos(rad_right))

    # Create a COMPLETELY ISOLATED working copy for this level
    result_img = img.copy()
    
    # Draw red outline around original rectangle before splitting (for visualization)
    draw_result = ImageDraw.Draw(result_img)
    draw_result.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=2)
    
    # Clear the original areas to avoid overlapping
    draw_result.rectangle([x, y, x+w, y+h], fill=(128, 128, 128))

    # Paste the rotated pieces onto the result image
    try:
        result_img.paste(left_rotated, (left_offset_x, left_offset_y))
        result_img.paste(right_rotated, (right_offset_x, right_offset_y))
        
        # Draw red outlines around the rotated pieces for visualization
        draw_result = ImageDraw.Draw(result_img)
        
        # Left piece outline with unique color for this depth
        outline_color = (255, 255 - depth * 40, 255 - depth * 40)  # Red to yellow gradient by depth
        
        left_outline = [
            left_offset_x, left_offset_y,
            left_offset_x + left_rotated.size[0], left_offset_y + left_rotated.size[1]
        ]
        draw_result.rectangle(left_outline, outline=outline_color, width=1)
        
        # Right piece outline
        right_outline = [
            right_offset_x, right_offset_y,
            right_offset_x + right_rotated.size[0], right_offset_y + right_rotated.size[1]
        ]
        draw_result.rectangle(right_outline, outline=outline_color, width=1)
        
        # Store actual positions for recursion
        actual_left_x, actual_left_y = left_offset_x, left_offset_y
        actual_right_x, actual_right_y = right_offset_x, right_offset_y
        
    except Exception as e:
        # Fallback positioning if calculations go out of bounds
        print(f"Positioning fallback at depth {depth}: {e}")
        safe_left_x = max(0, min(left_offset_x, result_img.size[0] - left_rotated.size[0]))
        safe_left_y = max(0, min(left_offset_y, result_img.size[1] - left_rotated.size[1]))
        safe_right_x = max(0, min(right_offset_x, result_img.size[0] - right_rotated.size[0]))
        safe_right_y = max(0, min(right_offset_y, result_img.size[1] - right_rotated.size[1]))
        
        result_img.paste(left_rotated, (safe_left_x, safe_left_y))
        result_img.paste(right_rotated, (safe_right_x, safe_right_y))
        
        # Store actual positions for recursion
        actual_left_x, actual_left_y = safe_left_x, safe_left_y
        actual_right_x, actual_right_y = safe_right_x, safe_right_y

    # RECURSIVE PROCESSING: Each child gets its own isolated workspace
    # Process left branch in complete isolation
    if depth > 1:
        print(f"  -> Recursing left branch at depth {depth-1}")
        result_img = recursive_split(result_img, actual_left_x, actual_left_y, sq, sq, 
                                   depth-1, angle, left_total_angle)
        
        print(f"  -> Recursing right branch at depth {depth-1}")
        result_img = recursive_split(result_img, actual_right_x, actual_right_y, sq, sq, 
                                   depth-1, angle, right_total_angle)
    
    return result_img

# 3. Apply recursion to the original image area on the larger canvas
# Any rotated parts that exceed the original bounds will show in the expanded canvas
out = canvas.copy()
print("Starting recursive split process...")
# *************************************************************************************************
# Using ADVANCED BUFFERING to prevent overwrites during iterative processing
# *************************************************************************************************
out = recursive_split_advanced(out, offset_x, offset_y, img_w, img_h, depth=3, angle=15)
# *************************************************************************************************
print("Recursive split process completed!")

draw_final = ImageDraw.Draw(out)

# 4. Display the result
plt.figure(figsize=(16,12))
plt.imshow(out)
plt.axis('off')
plt.title(f'Recursive Hinged Rotation - Full Canvas ({CANVAS_W}x{CANVAS_H})')
plt.show()

print("=== COMPLETE ===")