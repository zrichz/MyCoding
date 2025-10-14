#!/usr/bin/env python3
"""
Triangle Network Visualizer
Creates 100 random points and connects them based on distance threshold.
Colors triangles using red-blue gradient based on area (red=small, blue=large).
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import random
from itertools import combinations
import math

class TriangleNetworkVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Triangle Network Visualizer - Non-Overlapping")
        self.root.geometry("1500x1000")
        
        # Parameters
        self.num_points = 100
        self.canvas_width = 1400
        self.canvas_height = 800
        self.max_distance = 200  # Maximum possible distance for slider
        self.current_distance = 80  # Current distance threshold
        
        # Data
        self.points = []
        self.edges = []
        self.triangles = []
        self.triangle_areas = []
        
        # Generate initial random points
        self.generate_points()
        
        # Setup GUI
        self.setup_gui()
        
        # Initial update
        self.update_visualization()
    
    def setup_gui(self):
        """Setup the GUI with controls and matplotlib canvas"""
        
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Distance slider
        tk.Label(control_frame, text="Connection Distance Threshold:", 
                font=('Arial', 12, 'bold')).pack(anchor='w')
        
        self.distance_var = tk.DoubleVar(value=self.current_distance)
        self.distance_scale = tk.Scale(
            control_frame, 
            from_=10, 
            to=self.max_distance,
            orient='horizontal', 
            variable=self.distance_var,
            command=self.on_distance_change,
            length=400,
            resolution=5
        )
        self.distance_scale.pack(fill='x', pady=5)
        
        # Info labels
        info_frame = tk.Frame(control_frame)
        info_frame.pack(fill='x', pady=5)
        
        self.points_label = tk.Label(info_frame, text=f"Points: {self.num_points}", 
                                    font=('Arial', 10))
        self.points_label.pack(side='left', padx=(0, 20))
        
        self.edges_label = tk.Label(info_frame, text="Edges: 0", 
                                   font=('Arial', 10))
        self.edges_label.pack(side='left', padx=(0, 20))
        
        self.triangles_label = tk.Label(info_frame, text="Triangles: 0", 
                                       font=('Arial', 10))
        self.triangles_label.pack(side='left', padx=(0, 20))
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill='x', pady=5)
        
        tk.Button(button_frame, text="Generate New Points", 
                 command=self.generate_new_points,
                 font=('Arial', 10, 'bold'),
                 bg='lightblue').pack(side='left', padx=(0, 10))
        
        tk.Button(button_frame, text="Save Image", 
                 command=self.save_image,
                 font=('Arial', 10),
                 bg='lightgreen').pack(side='left')
        
        # Matplotlib figure (adjust aspect ratio for 1400x800)
        self.figure, self.ax = plt.subplots(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Configure plot
        self.ax.set_xlim(0, self.canvas_width)
        self.ax.set_ylim(0, self.canvas_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#808080')  # Mid-grey background
        self.ax.axis('off')  # Remove axes, labels, and ticks
    
    def generate_points(self):
        """Generate random points within the canvas"""
        self.points = []
        for _ in range(self.num_points):
            x = random.uniform(50, self.canvas_width - 50)  # Avoid edges
            y = random.uniform(50, self.canvas_height - 50)
            self.points.append((x, y))
    
    def generate_new_points(self):
        """Generate new random points and update visualization"""
        self.generate_points()
        self.update_visualization()
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def find_edges(self, distance_threshold):
        """Find all edges (connections) between points within distance threshold"""
        self.edges = []
        for i, j in combinations(range(len(self.points)), 2):
            dist = self.calculate_distance(self.points[i], self.points[j])
            if dist <= distance_threshold:
                self.edges.append((i, j))
    
    def triangles_overlap(self, tri1, tri2):
        """Check if two triangles overlap using comprehensive geometric tests with buffer zone"""
        
        # Check if triangles share vertices (they overlap if they share 2+ vertices)
        shared_vertices = set(tri1).intersection(set(tri2))
        if len(shared_vertices) >= 2:
            return True
        
        # If they share exactly 1 vertex, consider them overlapping to avoid visual crowding
        if len(shared_vertices) == 1:
            return True
        
        def point_in_triangle(px, py, x1, y1, x2, y2, x3, y3):
            """Check if point (px, py) is inside triangle using barycentric coordinates"""
            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(denom) < 1e-10:
                return False
            
            a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
            b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
            c = 1 - a - b
            
            return a >= -1e-10 and b >= -1e-10 and c >= -1e-10  # Small tolerance for edge cases
        
        def line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
            """Check if line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4) intersect"""
            def ccw(Ax, Ay, Bx, By, Cx, Cy):
                return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
            
            return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
                    ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))
        
        # Get triangle vertices
        p1_tri1, p2_tri1, p3_tri1 = [self.points[i] for i in tri1]
        p1_tri2, p2_tri2, p3_tri2 = [self.points[i] for i in tri2]
        
        # Add buffer zone by expanding triangles slightly
        def expand_triangle(p1, p2, p3, buffer=5):
            """Expand triangle outward by buffer pixels"""
            # Calculate centroid
            cx = (p1[0] + p2[0] + p3[0]) / 3
            cy = (p1[1] + p2[1] + p3[1]) / 3
            
            # Expand each vertex away from centroid
            expanded = []
            for px, py in [p1, p2, p3]:
                dx = px - cx
                dy = py - cy
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx = dx / length * buffer
                    dy = dy / length * buffer
                    expanded.append((px + dx, py + dy))
                else:
                    expanded.append((px, py))
            return expanded
        
        # Expand triangles with buffer
        exp_tri1 = expand_triangle(p1_tri1, p2_tri1, p3_tri1)
        exp_tri2 = expand_triangle(p1_tri2, p2_tri2, p3_tri2)
        
        # Check if any vertex of expanded tri1 is inside tri2
        for px, py in exp_tri1:
            if point_in_triangle(px, py, p1_tri2[0], p1_tri2[1], p2_tri2[0], p2_tri2[1], p3_tri2[0], p3_tri2[1]):
                return True
        
        # Check if any vertex of expanded tri2 is inside tri1
        for px, py in exp_tri2:
            if point_in_triangle(px, py, p1_tri1[0], p1_tri1[1], p2_tri1[0], p2_tri1[1], p3_tri1[0], p3_tri1[1]):
                return True
        
        # Check for edge intersections between the triangles
        tri1_edges = [(p1_tri1, p2_tri1), (p2_tri1, p3_tri1), (p3_tri1, p1_tri1)]
        tri2_edges = [(p1_tri2, p2_tri2), (p2_tri2, p3_tri2), (p3_tri2, p1_tri2)]
        
        for (p1, p2), (p3, p4) in [(e1, e2) for e1 in tri1_edges for e2 in tri2_edges]:
            if line_segments_intersect(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]):
                return True
        
        return False

    def find_triangles(self):
        """Find all triangles formed by the edges and select non-overlapping subset"""
        all_triangles = []
        all_triangle_areas = []
        
        # Create adjacency list from edges
        adjacency = {i: set() for i in range(len(self.points))}
        for i, j in self.edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Find all possible triangles: for each edge, check if there's a third point connected to both
        processed_triangles = set()
        
        for i, j in self.edges:
            # Find common neighbors of i and j
            common_neighbors = adjacency[i].intersection(adjacency[j])
            
            for k in common_neighbors:
                # Create triangle with vertices sorted to avoid duplicates
                triangle = tuple(sorted([i, j, k]))
                if triangle not in processed_triangles:
                    processed_triangles.add(triangle)
                    
                    # Calculate triangle area using cross product
                    p1, p2, p3 = self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]
                    area = abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)
                    
                    all_triangles.append(triangle)
                    all_triangle_areas.append(area)
        
        # Select non-overlapping triangles (greedy algorithm prioritizing larger areas)
        if not all_triangles:
            self.triangles = []
            self.triangle_areas = []
            return
        
        # Sort triangles by area (largest first)
        triangle_area_pairs = list(zip(all_triangles, all_triangle_areas))
        triangle_area_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Advanced non-overlapping selection with multiple strategies
        selected_triangles = []
        selected_areas = []
        
        # Strategy 1: Greedy by area (largest first)
        for triangle, area in triangle_area_pairs:
            # Check if this triangle overlaps with any already selected triangle
            overlaps = False
            for selected_triangle in selected_triangles:
                if self.triangles_overlap(triangle, selected_triangle):
                    overlaps = True
                    break
            
            if not overlaps:
                selected_triangles.append(triangle)
                selected_areas.append(area)
        
        # Strategy 2: If we have very few triangles, try a different approach
        if len(selected_triangles) < len(all_triangles) * 0.1:  # Less than 10% selected
            # Try selecting by spatial distribution to avoid clustering
            selected_triangles = []
            selected_areas = []
            
            # Sort by centroid distribution instead
            triangle_centroids = []
            for triangle, area in triangle_area_pairs:
                p1, p2, p3 = [self.points[i] for i in triangle]
                centroid_x = (p1[0] + p2[0] + p3[0]) / 3
                centroid_y = (p1[1] + p2[1] + p3[1]) / 3
                triangle_centroids.append((triangle, area, centroid_x, centroid_y))
            
            # Select triangles with minimum distance between centroids
            for triangle, area, cx, cy in triangle_centroids:
                overlaps = False
                for selected_triangle in selected_triangles:
                    if self.triangles_overlap(triangle, selected_triangle):
                        overlaps = True
                        break
                
                if not overlaps:
                    selected_triangles.append(triangle)
                    selected_areas.append(area)
        
        self.triangles = selected_triangles
        self.triangle_areas = selected_areas
        
        # Debug information
        total_possible = len(all_triangles)
        selected_count = len(selected_triangles)
        print(f"Triangle selection: {selected_count}/{total_possible} triangles selected (overlap prevention active)")
    
    def get_triangle_color(self, area, min_area, max_area):
        """Get color for triangle based on area (red=small, blue=large)"""
        if max_area == min_area:  # Avoid division by zero
            return (0.5, 0.0, 0.5)  # Purple for single area
        
        # Normalize area to 0-1 range
        normalized = (area - min_area) / (max_area - min_area)
        
        # Red to Blue gradient
        # Red: (1, 0, 0) -> Purple: (0.5, 0, 0.5) -> Blue: (0, 0, 1)
        red = 1.0 - normalized
        green = 0.0
        blue = normalized
        
        return (red, green, blue)
    
    def on_distance_change(self, value):
        """Handle distance slider change"""
        self.current_distance = float(value)
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization with current parameters"""
        # Clear the plot
        self.ax.clear()
        
        # Find edges and triangles
        self.find_edges(self.current_distance)
        self.find_triangles()
        
        # Update info labels
        self.edges_label.config(text=f"Edges: {len(self.edges)}")
        self.triangles_label.config(text=f"Triangles: {len(self.triangles)}")
        
        # Draw triangles with area-based coloring
        if self.triangles and self.triangle_areas:
            min_area = min(self.triangle_areas)
            max_area = max(self.triangle_areas)
            
            for triangle, area in zip(self.triangles, self.triangle_areas):
                # Get triangle vertices
                vertices = [self.points[i] for i in triangle]
                
                # Get color based on area
                color = self.get_triangle_color(area, min_area, max_area)
                
                # Create and add triangle patch
                triangle_patch = Polygon(vertices, facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                self.ax.add_patch(triangle_patch)
        
        # Draw edges with variable thickness and darkness based on length
        if self.edges:
            # Calculate all edge lengths for normalization
            edge_lengths = []
            for i, j in self.edges:
                p1, p2 = self.points[i], self.points[j]
                length = self.calculate_distance(p1, p2)
                edge_lengths.append(length)
            
            min_length = min(edge_lengths) if edge_lengths else 1
            max_length = max(edge_lengths) if edge_lengths else 1
            
            # Draw edges with varying thickness and opacity
            for (i, j), length in zip(self.edges, edge_lengths):
                p1, p2 = self.points[i], self.points[j]
                
                # Normalize length (0 = shortest, 1 = longest)
                if max_length > min_length:
                    normalized_length = (length - min_length) / (max_length - min_length)
                else:
                    normalized_length = 0
                
                # Shorter lines get thicker and darker (inverted relationship)
                # Thickness: 3.0 for shortest, 0.5 for longest
                thickness = 3.0 - (normalized_length * 2.5)
                
                # Alpha: 0.9 for shortest, 0.2 for longest
                alpha = 0.9 - (normalized_length * 0.7)
                
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', 
                           alpha=alpha, linewidth=thickness)
        
        # Draw points
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        self.ax.scatter(x_coords, y_coords, c='white', s=30, edgecolors='black', linewidths=1, zorder=5)
        
        # Configure plot appearance
        self.ax.set_xlim(0, self.canvas_width)
        self.ax.set_ylim(0, self.canvas_height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#808080')  # Mid-grey background
        self.ax.axis('off')  # Remove axes, labels, ticks, and title
        
        # Area range info removed for clean visualization
        
        # Refresh canvas
        self.canvas.draw()
    
    def save_image(self):
        """Save the current visualization as an image"""
        filename = f"triangle_network_d{self.current_distance:.0f}_t{len(self.triangles)}.png"
        self.figure.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization as {filename}")

def main():
    """Main function to run the application"""
    # Check dependencies
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install with: pip install matplotlib numpy")
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = TriangleNetworkVisualizer(root)
    
    print("Non-Overlapping Triangle Network Visualizer Started")
    print("- Canvas size: 1400x800 pixels")
    print("- Triangles are selected to prevent overlapping (larger triangles prioritized)")
    print("- Adjust the distance slider to change point connections")
    print("- Red triangles = small area, Blue triangles = large area")
    print("- Click 'Generate New Points' for new random layout")
    print("- Click 'Save Image' to export current visualization")
    
    root.mainloop()

if __name__ == "__main__":
    main()
