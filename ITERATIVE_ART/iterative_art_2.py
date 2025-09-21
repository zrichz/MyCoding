#!/home/rich/MyCoding/image_processors/.venv/bin/python3
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math

# Canvas dimensions
CANVAS_W, CANVAS_H = 1200, 800

#=======================================================================================
RECT_W, RECT_H = 680, 340 # Initial rectangle dimensions (2:1 ratio
rot_angle = 15   # Rotation angle per depth level
depth = 3       # Set recursion depth here
#=======================================================================================

def rotate_point_around_center(px, py, cx, cy, angle_degrees):
    """Rotate a point (px, py) around center (cx, cy) by angle_degrees"""
    angle_rad = math.radians(angle_degrees)
    
    # Translate point to origin
    rel_x = px - cx
    rel_y = py - cy
    
    # Rotate around origin
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    new_x = rel_x * cos_a - rel_y * sin_a
    new_y = rel_x * sin_a + rel_y * cos_a
    
    # Translate back
    return (new_x + cx, new_y + cy)

def draw_rectangle_outline(draw, x, y, w, h, color=(255, 255, 255), width=2):
    """Draw a rectangle outline"""
    draw.rectangle([x, y, x+w, y+h], outline=color, width=width)

def recursive_split_rectangles(canvas, x, y, w, h, depth, rotation_angle, colors):
    """
    Simple recursive splitting:
    1. Split 2:1 rectangle into 2 equal halves
    2. Rotate ONE half away from the other by rot_angle degrees using adjoining corner as pivot
    3. Recurse on both resulting rectangles
    """
    if depth == 0:
        return
    
    # Get drawing context
    draw = ImageDraw.Draw(canvas)
    
    # Current level color
    color = colors[depth % len(colors)]
    
    # Draw the current rectangle
    draw_rectangle_outline(draw, x, y, w, h, color, width=2)
    
    print(f"Depth {depth}: Processing rectangle at ({x},{y}) size {w}x{h}, rotation: {rotation_angle}°")
    
    # Determine split direction based on aspect ratio
    if w > h:  # Horizontal rectangle - split vertically
        # Split into left and right halves
        left_w = w // 2
        right_w = w - left_w
        
        left_rect = (x, y, left_w, h)
        right_rect = (x + left_w, y, right_w, h)
        
        # Pivot point is the adjoining corner at the base (bottom edge)
        pivot_x = x + left_w
        pivot_y = y + h
        
        print(f"  Horizontal split: Left {left_rect}, Right {right_rect}, Pivot ({pivot_x},{pivot_y})")
        
        # Keep left half stationary, rotate right half away
        left_corners = calculate_rectangle_corners(left_rect, 0)  # No rotation
        right_corners = calculate_rotated_corners(right_rect, pivot_x, pivot_y, rot_angle)
        
        # Calculate centers for recursion positioning
        left_center = (x + left_w//2, y + h//2)
        right_center = calculate_rotated_center(right_rect, pivot_x, pivot_y, rot_angle)
        
        # Store values for later use
        is_horizontal = True
        stationary_corners = left_corners
        rotated_corners = right_corners
        stationary_center = left_center
        rotated_center = right_center
        stationary_w = left_w
        stationary_h = h
        rotated_w = right_w
        rotated_h = h
        
    else:  # Vertical rectangle - split horizontally  
        # Split into top and bottom halves
        top_h = h // 2
        bottom_h = h - top_h
        
        top_rect = (x, y, w, top_h)
        bottom_rect = (x, y + top_h, w, bottom_h)
        
        # Pivot point is the adjoining corner at the base (bottom edge of entire rectangle)
        pivot_x = x + w
        pivot_y = y + h
        
        print(f"  Vertical split: Top {top_rect}, Bottom {bottom_rect}, Pivot ({pivot_x},{pivot_y})")
        
        # Keep top half stationary, rotate bottom half away
        top_corners = calculate_rectangle_corners(top_rect, 0)  # No rotation
        bottom_corners = calculate_rotated_corners(bottom_rect, pivot_x, pivot_y, rot_angle)
        
        # Calculate centers for recursion positioning
        top_center = (x + w//2, y + top_h//2)
        bottom_center = calculate_rotated_center(bottom_rect, pivot_x, pivot_y, rot_angle)
        
        # Store values for later use
        is_horizontal = False
        stationary_corners = top_corners
        rotated_corners = bottom_corners
        stationary_center = top_center
        rotated_center = bottom_center
        stationary_w = w
        stationary_h = top_h
        rotated_w = w
        rotated_h = bottom_h
    
    # Draw pivot point
    pivot_size = 4
    draw.ellipse([pivot_x - pivot_size, pivot_y - pivot_size,
                  pivot_x + pivot_size, pivot_y + pivot_size], 
                 fill=(255, 0, 255))  # Magenta pivot
    
    # Draw both rectangles using unified variables
    draw_rotated_rectangle(draw, stationary_corners, color, width=2)
    draw_rotated_rectangle(draw, rotated_corners, (255, 100, 100), width=2)  # Different color for rotated half
    
    # Draw rotation indicator
    line_length = 20
    rad = math.radians(rot_angle)
    end_x = pivot_x + int(line_length * math.cos(rad))
    end_y = pivot_y + int(line_length * math.sin(rad))
    draw.line([(pivot_x, pivot_y), (end_x, end_y)], fill=(0, 255, 0), width=2)
    
    # Recurse on both halves
    if depth > 1:
        print(f"  Recursing on stationary half")
        recursive_split_rectangles(canvas, int(stationary_center[0] - stationary_w//2), int(stationary_center[1] - stationary_h//2),
                                 stationary_w, stationary_h, depth-1, rotation_angle, colors)
        
        print(f"  Recursing on rotated half")
        recursive_split_rectangles(canvas, int(rotated_center[0] - rotated_w//2), int(rotated_center[1] - rotated_h//2),
                                 rotated_w, rotated_h, depth-1, rotation_angle + rot_angle, colors)

def calculate_rotated_corners(square, hinge_x, hinge_y, angle_degrees):
    """Calculate the corners of a square after rotation around a hinge point"""
    x, y, w, h = square
    angle_rad = math.radians(angle_degrees)
    
    # Original corners relative to hinge
    corners = [
        (x - hinge_x, y - hinge_y),           # Top-left
        (x + w - hinge_x, y - hinge_y),       # Top-right  
        (x + w - hinge_x, y + h - hinge_y),   # Bottom-right
        (x - hinge_x, y + h - hinge_y)        # Bottom-left
    ]
    
    # Rotate each corner around origin (0,0) then translate back to hinge
    rotated_corners = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for cx, cy in corners:
        # Rotate around origin
        new_x = cx * cos_a - cy * sin_a
        new_y = cx * sin_a + cy * cos_a
        # Translate back to hinge position
        rotated_corners.append((new_x + hinge_x, new_y + hinge_y))
    
    return rotated_corners

def calculate_rotated_center(square, hinge_x, hinge_y, angle_degrees):
    """Calculate the center of a square after rotation around a hinge point"""
    x, y, w, h = square
    
    # Original center
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Center relative to hinge
    rel_x = center_x - hinge_x
    rel_y = center_y - hinge_y
    
    # Rotate around origin
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    new_x = rel_x * cos_a - rel_y * sin_a
    new_y = rel_x * sin_a + rel_y * cos_a
    
    # Translate back to hinge position
    return (new_x + hinge_x, new_y + hinge_y)

def draw_rotated_rectangle(draw, corners, color, width=2):
    """Draw a rectangle using its rotated corner coordinates"""
    # Draw lines between consecutive corners
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]
        draw.line([start, end], fill=color, width=width)

def calculate_rectangle_corners(rect, angle_degrees):
    """Calculate the corners of a rectangle (no rotation, just return corners)"""
    x, y, w, h = rect
    return [
        (x, y),           # Top-left
        (x + w, y),       # Top-right  
        (x + w, y + h),   # Bottom-right
        (x, y + h)        # Bottom-left
    ]

def main():
    print("=== ITERATIVE ART 2: SIMPLE RECTANGLE SPLITTING ===")
    
    # Create canvas with grid background
    canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), color=(40, 40, 40))  # Dark background
    
    # Add grid lines
    draw = ImageDraw.Draw(canvas)
    grid_color = (70, 70, 70)  # Dark grey grid
    grid_spacing = 50
    
    # Vertical grid lines
    for x in range(0, CANVAS_W, grid_spacing):
        draw.line([(x, 0), (x, CANVAS_H)], fill=grid_color, width=1)
    
    # Horizontal grid lines
    for y in range(0, CANVAS_H, grid_spacing):
        draw.line([(0, y), (CANVAS_W, y)], fill=grid_color, width=1)
    
    # Center the initial rectangle
    start_x = (CANVAS_W - RECT_W) // 2
    start_y = (CANVAS_H - RECT_H) // 2
    
    print(f"Starting rectangle: {RECT_W}x{RECT_H} at ({start_x}, {start_y})")
    
    # Color sequence for each depth level
    colors = [
        (255, 255, 255),  # White - depth 0
        (255, 200, 200),  # Light red - depth 1  
        (200, 255, 200),  # Light green - depth 2
        (200, 200, 255),  # Light blue - depth 3
        (255, 255, 200),  # Light yellow - depth 4
        (255, 200, 255),  # Light magenta - depth 5
        (200, 255, 255),  # Light cyan - depth 6
    ]
    
    # Define initial rectangle pivot point (center width, base)
    initial_pivot_x = start_x + RECT_W // 2  # Center width
    initial_pivot_y = start_y + RECT_H       # Base (bottom) of rectangle
    
    print(f"Initial rectangle pivot: center width + base = ({initial_pivot_x}, {initial_pivot_y})")
    
    # Draw the initial rectangle pivot point
    draw = ImageDraw.Draw(canvas)
    pivot_size = 6
    draw.ellipse([initial_pivot_x - pivot_size, initial_pivot_y - pivot_size,
                  initial_pivot_x + pivot_size, initial_pivot_y + pivot_size], 
                 fill=(255, 0, 255))  # Magenta pivot for initial rectangle
    
    # No initial rotation - rectangle stays in its original position
    print(f"No initial rectangle rotation applied")
    
    # Draw the initial rectangle outline in its original position
    draw.rectangle([start_x, start_y, start_x + RECT_W, start_y + RECT_H], outline=(255, 255, 0), width=3)  # Yellow outline for initial rectangle
    
    # Start recursion from the original rectangle position
    rotated_start_x = start_x
    rotated_start_y = start_y
    
    # Start recursion with no initial rotation
    initial_rotation = 0
    
    print(f"Recursion depth: {depth}")
    print(f"Starting recursion from position: ({rotated_start_x}, {rotated_start_y})")
    
    recursive_split_rectangles(canvas, rotated_start_x, rotated_start_y, RECT_W, RECT_H, 
                             depth, initial_rotation, colors)
    
    # Add title and info
    draw = ImageDraw.Draw(canvas)
    title_color = (255, 255, 255)
    draw.text((10, 10), f"Iterative Art 2: Rectangle Splitting (Depth {depth})", 
              fill=title_color)
    draw.text((10, 30), f"Canvas: {CANVAS_W}x{CANVAS_H}, Initial rect: {RECT_W}x{RECT_H}", 
              fill=title_color)
    draw.text((10, 50), "Magenta = hinge points, Green = left rotation, Orange = right rotation", 
              fill=title_color)
    
    # Display result
    plt.figure(figsize=(16, 10))
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('Iterative Rectangle Splitting with Rotation')
    plt.show()
    
    print("=== COMPLETE ===")

if __name__ == "__main__":
    main()
