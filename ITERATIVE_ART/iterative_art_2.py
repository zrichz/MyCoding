#!/home/rich/MyCoding/image_processors/.venv/bin/python3
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math

# Canvas dimensions
CANVAS_W, CANVAS_H = 1200, 800

#=======================================================================================
RECT_W, RECT_H = 680, 340 # Initial rectangle dimensions (2:1 ratio
rot_angle = 12   # Rotation angle per depth level
depth = 2       # Set recursion depth here
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

def recursive_split_rectangles(canvas, x, y, w, h, depth, angles, colors):
    """
    Correct iterative cycle:
    1. Split 2:1 rectangle into 2 squares
    2. Split each square into 2 rectangles (2:1 ratio each)  
    3. Rotate each pair around their respective pivot points
    4. Recurse on all four 2:1 rectangles
    """
    if depth == 0:
        return
    
    # Get drawing context
    draw = ImageDraw.Draw(canvas)
    
    # Current level color
    color = colors[depth % len(colors)]
    
    # Draw the current 2:1 rectangle before splitting
    draw_rectangle_outline(draw, x, y, w, h, color, width=2)
    
    print(f"Depth {depth}: Processing 2:1 rectangle at ({x},{y}) size {w}x{h}")
    
    
    
    # STEP 1: Split 2:1 rectangle into 2 squares
    sq = h  # Square size = height of rectangle
    left_square = (x, y, sq, sq)        # Left square
    right_square = (x + sq, y, sq, sq)  # Right square
    
    print(f"  Step 1: Split into squares - Left: {left_square}, Right: {right_square}")
    
    # STEP 2: Split each square into 2 rectangles (2:1 ratio each)
    rect_w = sq      # Rectangle width = square size
    rect_h = sq // 2 # Rectangle height = half square for 2:1 ratio
    
    # Left square splits into 2 rectangles (stacked vertically)
    left_rect1 = (x, y, rect_w, rect_h)                    # Top rectangle of left square
    left_rect2 = (x, y + rect_h, rect_w, rect_h)          # Bottom rectangle of left square
    
    # Right square splits into 2 rectangles (stacked vertically) 
    right_rect1 = (x + sq, y, rect_w, rect_h)             # Top rectangle of right square
    right_rect2 = (x + sq, y + rect_h, rect_w, rect_h)    # Bottom rectangle of right square
    
    print(f"  Step 2: Left square -> {left_rect1} and {left_rect2}")
    print(f"  Step 2: Right square -> {right_rect1} and {right_rect2}")
    
    # STEP 3: Define pivot points for rotation
    # For rotated rectangles, we need to calculate the actual rotated positions of the pivot points
    
    # Original pivot points (before any rotation)
    left_pivot_orig_x = x                 # Left edge of left square
    left_pivot_orig_y = y + sq // 2       # Halfway up the left square
    
    right_pivot_orig_x = x + w            # Right edge of right square (= x + 2*sq)
    right_pivot_orig_y = y + sq // 2      # Halfway up the right square
    
    # If this rectangle has been rotated (angles != 0), we need to apply that rotation to the pivot points
    current_rotation = angles['left']  # Assuming both left/right have same rotation at this level
    
    if current_rotation != 0:
        # Calculate the center of the current rectangle as reference point
        rect_center_x = x + w // 2
        rect_center_y = y + h // 2
        
        # Rotate the pivot points around the rectangle center
        left_pivot_x, left_pivot_y = rotate_point_around_center(
            left_pivot_orig_x, left_pivot_orig_y, rect_center_x, rect_center_y, current_rotation)
        right_pivot_x, right_pivot_y = rotate_point_around_center(
            right_pivot_orig_x, right_pivot_orig_y, rect_center_x, rect_center_y, current_rotation)
    else:
        # No rotation, use original positions
        left_pivot_x = left_pivot_orig_x
        left_pivot_y = left_pivot_orig_y
        right_pivot_x = right_pivot_orig_x
        right_pivot_y = right_pivot_orig_y
    
    print(f"  Step 3: Left pivot at ({int(round(left_pivot_x))},{int(round(left_pivot_y))}), Right pivot at ({int(round(right_pivot_x))},{int(round(right_pivot_y))})")
    
    # Draw pivot points
    pivot_size = 4
    draw.ellipse([left_pivot_x - pivot_size, left_pivot_y - pivot_size,
                  left_pivot_x + pivot_size, left_pivot_y + pivot_size], 
                 fill=(255, 0, 255))  # Magenta pivot
    draw.ellipse([right_pivot_x - pivot_size, right_pivot_y - pivot_size,
                  right_pivot_x + pivot_size, right_pivot_y + pivot_size], 
                 fill=(255, 0, 255))  # Magenta pivot
    
    # STEP 4: Calculate rotation angles for each rectangle
    # Create symmetric butterfly/book opening effect - mirror across vertical axis
    # REVERSED: Left side now rotates clockwise, right side rotates counter-clockwise
    
    # Left side rotates clockwise (negative degrees) - REVERSED
    left_rect1_angle = angles['left'] - rot_angle       # Top left rectangle rotates -12° CW
    left_rect2_angle = angles['left'] - rot_angle       # Bottom left rectangle also rotates -12° CW
    
    # Right side rotates counter-clockwise (positive degrees) - REVERSED  
    right_rect1_angle = angles['right'] + rot_angle     # Top right rectangle rotates +12° CCW
    right_rect2_angle = angles['right'] + rot_angle     # Bottom right rectangle also rotates +12° CCW
    
    print(f"  Step 4: Angles - LR1:{left_rect1_angle}° LR2:{left_rect2_angle}° RR1:{right_rect1_angle}° RR2:{right_rect2_angle}°")
    
    # STEP 5: Draw rotated rectangles
    # Left pair
    left_rect1_corners = calculate_rotated_corners(left_rect1, left_pivot_x, left_pivot_y, left_rect1_angle)
    left_rect2_corners = calculate_rotated_corners(left_rect2, left_pivot_x, left_pivot_y, left_rect2_angle)
    draw_rotated_rectangle(draw, left_rect1_corners, color, width=2)
    draw_rotated_rectangle(draw, left_rect2_corners, color, width=2)
    
    # Right pair
    right_rect1_corners = calculate_rotated_corners(right_rect1, right_pivot_x, right_pivot_y, right_rect1_angle)
    right_rect2_corners = calculate_rotated_corners(right_rect2, right_pivot_x, right_pivot_y, right_rect2_angle)
    draw_rotated_rectangle(draw, right_rect1_corners, color, width=2)
    draw_rotated_rectangle(draw, right_rect2_corners, color, width=2)
    
    # Draw rotation indicators
    line_length = 25
    
    # Left pivot indicators
    for angle, color_line in [(left_rect1_angle, (0, 255, 0)), (left_rect2_angle, (0, 200, 0))]:
        rad = math.radians(angle)
        end_x = left_pivot_x + int(line_length * math.cos(rad))
        end_y = left_pivot_y + int(line_length * math.sin(rad))
        draw.line([(left_pivot_x, left_pivot_y), (end_x, end_y)], fill=color_line, width=2)
    
    # Right pivot indicators  
    for angle, color_line in [(right_rect1_angle, (255, 100, 0)), (right_rect2_angle, (200, 80, 0))]:
        rad = math.radians(angle)
        end_x = right_pivot_x + int(line_length * math.cos(rad))
        end_y = right_pivot_y + int(line_length * math.sin(rad))
        draw.line([(right_pivot_x, right_pivot_y), (end_x, end_y)], fill=color_line, width=2)
    
    # STEP 6: Recurse on all four 2:1 rectangles
    if depth > 1:
        print(f"  Step 6: Recursing to depth {depth-1}")
        
        # Calculate centers of rotated rectangles for positioning
        left_rect1_center = calculate_rotated_center(left_rect1, left_pivot_x, left_pivot_y, left_rect1_angle)
        left_rect2_center = calculate_rotated_center(left_rect2, left_pivot_x, left_pivot_y, left_rect2_angle)
        right_rect1_center = calculate_rotated_center(right_rect1, right_pivot_x, right_pivot_y, right_rect1_angle)  
        right_rect2_center = calculate_rotated_center(right_rect2, right_pivot_x, right_pivot_y, right_rect2_angle)
        
        # New angles for next iteration - maintain REVERSED left/right symmetry
        # For rectangles from left side, they continue as left with negative angles (REVERSED)
        new_angles_lr1 = {'left': left_rect1_angle, 'right': -left_rect1_angle}
        new_angles_lr2 = {'left': left_rect2_angle, 'right': -left_rect2_angle}
        
        # For rectangles from right side, they continue as right with positive angles (REVERSED) 
        new_angles_rr1 = {'left': -right_rect1_angle, 'right': right_rect1_angle}
        new_angles_rr2 = {'left': -right_rect2_angle, 'right': right_rect2_angle}
        
        # Recurse into all four rectangles
        recursive_split_rectangles(canvas, int(left_rect1_center[0] - rect_w//2), int(left_rect1_center[1] - rect_h//2),
                                 rect_w, rect_h, depth-1, new_angles_lr1, colors)
        
        recursive_split_rectangles(canvas, int(left_rect2_center[0] - rect_w//2), int(left_rect2_center[1] - rect_h//2),
                                 rect_w, rect_h, depth-1, new_angles_lr2, colors)
        
        recursive_split_rectangles(canvas, int(right_rect1_center[0] - rect_w//2), int(right_rect1_center[1] - rect_h//2),
                                 rect_w, rect_h, depth-1, new_angles_rr1, colors)
        
        recursive_split_rectangles(canvas, int(right_rect2_center[0] - rect_w//2), int(right_rect2_center[1] - rect_h//2),
                                 rect_w, rect_h, depth-1, new_angles_rr2, colors)

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
    
    # Initial angles start at zero - REVERSED symmetric from start
    initial_angles = {'left': 0, 'right': 0}
    
    # Perform recursive splitting from the rotated position
    
    print(f"Recursion depth: {depth}")
    print(f"Starting recursion from rotated position: ({rotated_start_x}, {rotated_start_y})")
    
    recursive_split_rectangles(canvas, rotated_start_x, rotated_start_y, RECT_W, RECT_H, 
                             depth, initial_angles, colors)
    
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
