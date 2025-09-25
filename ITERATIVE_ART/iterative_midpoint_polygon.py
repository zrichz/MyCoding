#!/home/rich/MyCoding/image_processors/.venv/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Configuration - Modify these values to experiment
N_SIDES = 150        # Number of sides for the initial polygon
ITERATIONS = 20     # Number of iterations to perform
CANVAS_W, CANVAS_H = 1920, 1080  # Canvas dimensions optimized for your screen
INITIAL_RADIUS = 450  # Radius of the initial polygon

# Colors for each iteration (will cycle if more iterations than colors)
COLORS = [
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Blue
    '#96CEB4',  # Green
    '#FFEAA7',  # Yellow
    '#DDA0DD',  # Plum
    '#98D8C8',  # Mint
    '#F7DC6F',  # Light Yellow
    '#BB8FCE',  # Light Purple
    '#85C1E9',  # Light Blue
]

def generate_random_polygon(n_sides, max_radius, center_x, center_y):
    """
    Generate vertices forming a horizontal line across the screen
    with subtle variations in both height and horizontal position
    Returns list of (x, y) tuples
    """
    import random
    
    vertices = []
    
    # Create a horizontal line with variations
    # Spread points across most of the screen width
    screen_width = CANVAS_W * 0.8  # Use 80% of screen width
    start_x = center_x - screen_width / 2
    
    for i in range(n_sides):
        # Base horizontal position - evenly spaced across the line
        base_x = start_x + (screen_width * i) / (n_sides - 1)
        
        # Add subtle random horizontal variation (±5% of spacing)
        spacing = screen_width / (n_sides - 1) if n_sides > 1 else 0
        x_variation = (random.random() - 0.5) * spacing * 0.1
        x = base_x + x_variation
        
        # Base vertical position at center, with subtle random height variation
        # Vary height by ±50 pixels
        y_variation = (random.random() - 0.5) * 100
        y = center_y + y_variation
        
        vertices.append((x, y))
    
    # Sort vertices by x-coordinate to maintain left-to-right order
    vertices.sort(key=lambda vertex: vertex[0])
    
    return vertices

def calculate_blur_step(vertices):
    """
    Apply blur operation: p' = u/4 + p/2 + v/4
    Each point becomes a weighted average of itself and its neighbors
    For open line: endpoints use different weighting
    Returns list of blurred (x, y) tuples
    """
    blurred = []
    n = len(vertices)
    
    for i in range(n):
        current = vertices[i]
        
        if i == 0:
            # First vertex: only use current and next
            next_vertex = vertices[i + 1]
            new_x = current[0] * 0.75 + next_vertex[0] * 0.25
            new_y = current[1] * 0.75 + next_vertex[1] * 0.25
        elif i == n - 1:
            # Last vertex: only use previous and current
            prev_vertex = vertices[i - 1]
            new_x = prev_vertex[0] * 0.25 + current[0] * 0.75
            new_y = prev_vertex[1] * 0.25 + current[1] * 0.75
        else:
            # Middle vertices: use all three neighbors
            prev_vertex = vertices[i - 1]
            next_vertex = vertices[i + 1]
            new_x = prev_vertex[0] * 0.25 + current[0] * 0.5 + next_vertex[0] * 0.25
            new_y = prev_vertex[1] * 0.25 + current[1] * 0.5 + next_vertex[1] * 0.25
        
        blurred.append((new_x, new_y))
    
    return blurred

def calculate_sharpen_step(vertices):
    """
    Apply sharpen operation: p' = 2*p - u/2 - v/2
    Each point is enhanced relative to its neighbors
    For open line: endpoints use different weighting
    Returns list of sharpened (x, y) tuples
    """
    sharpened = []
    n = len(vertices)
    
    for i in range(n):
        current = vertices[i]
        
        if i == 0:
            # First vertex: only use current and next
            next_vertex = vertices[i + 1]
            new_x = 1.5 * current[0] - 0.5 * next_vertex[0]
            new_y = 1.5 * current[1] - 0.5 * next_vertex[1]
        elif i == n - 1:
            # Last vertex: only use previous and current
            prev_vertex = vertices[i - 1]
            new_x = 1.5 * current[0] - 0.5 * prev_vertex[0]
            new_y = 1.5 * current[1] - 0.5 * prev_vertex[1]
        else:
            # Middle vertices: use all three neighbors
            prev_vertex = vertices[i - 1]
            next_vertex = vertices[i + 1]
            new_x = 2 * current[0] - prev_vertex[0] * 0.5 - next_vertex[0] * 0.5
            new_y = 2 * current[1] - prev_vertex[1] * 0.5 - next_vertex[1] * 0.5
        
        sharpened.append((new_x, new_y))
    
    return sharpened

def draw_polygon(ax, vertices, color, linewidth=2.0, alpha=0.8, fill=False):
    """
    Draw an open line (not closed polygon) given its vertices
    """
    # Do not close the shape - just use vertices as they are
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    
    if fill:
        # Note: fill won't work properly for open lines, but keeping for compatibility
        ax.fill(x_coords, y_coords, color=color, alpha=alpha*0.3, edgecolor=color, linewidth=linewidth)
    else:
        ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, alpha=alpha)
    
    # No vertex dots - removed scatter plot

def main():
    print("=== ITERATIVE BLUR-SHARPEN POLYGON GENERATOR ===")
    print(f"Initial shape: Horizontal line with {N_SIDES} vertices")
    print(f"Iterations: {ITERATIONS}")
    print(f"Canvas: {CANVAS_W}x{CANVAS_H}")
    
    # Create figure optimized for 1920x1080 screen
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.set_facecolor('#001133')  # Dark blue background
    fig.patch.set_facecolor('#001133')
    
    # Set up the canvas
    center_x, center_y = CANVAS_W // 2, CANVAS_H // 2
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Generate initial polygon (completely random polygon)
    current_vertices = generate_random_polygon(N_SIDES, INITIAL_RADIUS, center_x, center_y)
    all_polygons = [current_vertices]  # Store all polygons for visualization
    
    print(f"Initial polygon vertices: {len(current_vertices)}")
    
    # Generate iterations - alternate between blur and sharpen
    for iteration in range(ITERATIONS):
        # Alternate between blur (even) and sharpen (odd) operations
        if iteration % 2 == 0:
            # Blur step
            new_vertices = calculate_blur_step(current_vertices)
            operation = "blur"
        else:
            # Sharpen step
            new_vertices = calculate_sharpen_step(current_vertices)
            operation = "sharpen"
        
        all_polygons.append(new_vertices)
        current_vertices = new_vertices
        
        print(f"Iteration {iteration + 1}: {operation} - Generated polygon with {len(new_vertices)} vertices")
    
    # Draw all polygons
    print("Drawing polygons...")
    
    for i, vertices in enumerate(all_polygons):
        # Special colors and thickness for first and last iterations
        if i == 0:
            color = '#0080FF'  # Bright blue for first iteration
            linewidth = 3  # Thicker for start
        elif i == len(all_polygons) - 1:
            color = '#FF0040'  # Bright red for last iteration
            linewidth = 3  # Thicker for end
        else:
            # All intermediate iterations in white
            color = '#FFFFFF'  # White for all intermediate steps
            linewidth = 1  # Normal thickness for intermediate steps
        
        # Make earlier polygons more transparent and later ones more opaque
        alpha = 0.4 + (i / len(all_polygons)) * 0.6
        
        # No fill for anything
        fill = False
        
        draw_polygon(ax, vertices, color, linewidth=linewidth, alpha=alpha, fill=fill)
    
    # Add title and information
    title_color = 'white'
    ax.text(50, CANVAS_H - 50, f'Iterative Blur-Sharpen Polygon', 
            fontsize=24, color=title_color, weight='bold')
    ax.text(50, CANVAS_H - 90, f'Horizontal line with {N_SIDES} vertices → {ITERATIONS} iterations', 
            fontsize=16, color=title_color)
    ax.text(50, CANVAS_H - 120, f'Alternating blur (smooth) and sharpen (enhance) operations', 
            fontsize=14, color=title_color, alpha=0.8)
    
    # Add iteration counter in corner
    ax.text(CANVAS_W - 200, 50, f'Total iterations: {len(all_polygons)}', 
            fontsize=12, color=title_color, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    print("=== COMPLETE ===")

if __name__ == "__main__":
    main()
