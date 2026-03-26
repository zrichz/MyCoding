"""
Overlapping Circles with Quantized Edge Detection
Draws colored circles with striped circumferences where only overlapping areas are colored.
Uses NumPy vectorization for high performance with quantized edge detection.
Only pixels at distances divisible by 2 from the circumference edge are colored.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import math
import time

class CircleOverlapVisualizer:
    def __init__(self, width=1600, height=900, num_circles=400):
        # Resolution
        self.width = width
        self.height = height
        
        # Circle parameters
        self.num_circles = num_circles  # Number of circles to draw
        self.min_radius = 60
        self.max_radius = 200
        
        # Edge thickness for circumferences
        self.edge_thickness = 30
        
        # Generate random circles
        self.circles = self.generate_circles()
        
        # Create the visualization
        self.create_image()
    
    def generate_circles(self):
        """Generate random circles with positions, sizes, and colors"""
        circles = []
        
        # Random bright colors
        bright_colors = [
            (1.0, 0.0, 0.0),  # Bright red
            (0.0, 1.0, 0.0),  # Bright green
            (0.0, 0.0, 1.0),  # Bright blue
            (1.0, 1.0, 0.0),  # Bright yellow
            (1.0, 0.0, 1.0),  # Bright magenta
            (0.0, 1.0, 1.0),  # Bright cyan
            (1.0, 0.5, 0.0),  # Bright orange
            (0.5, 0.0, 1.0),  # Bright purple
            (1.0, 0.8, 0.0),  # Bright gold
            (0.0, 1.0, 0.5),  # Bright lime green
            (1.0, 0.0, 0.5),  # Bright pink
            (0.5, 1.0, 0.0),  # Bright chartreuse
        ]
        
        for i in range(self.num_circles):
            # Random position (allow circles to be partially off-canvas)
            # Extend range beyond canvas boundaries by max radius
            extended_margin = self.max_radius
            x = random.uniform(-extended_margin, self.width + extended_margin)
            y = random.uniform(-extended_margin, self.height + extended_margin)
            
            # Random radius
            radius = random.uniform(self.min_radius, self.max_radius)
            
            # Random bright color
            color = random.choice(bright_colors)
            
            circles.append({
                'center': (x, y),
                'radius': radius,
                'color': color
            })
        
        return circles
    
    def distance_to_circle_center(self, px, py, circle):
        """Calculate distance from pixel to circle center"""
        cx, cy = circle['center']
        return math.sqrt((px - cx)**2 + (py - cy)**2)

    def create_image(self):
        """Create the full image with overlapping circles using vectorized operations"""
        print("Generating overlapping circles visualization...")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Number of circles: {len(self.circles)}")
        
        start_time = time.time()
        
        # Create coordinate grids for all pixels at once
        x_coords, y_coords = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        # Initialize image with dark grey background
        image = np.full((self.height, self.width, 3), 0.1, dtype=np.float32)
        
        # Initialize circumference count array
        circumference_count = np.zeros((self.height, self.width), dtype=int)
        
        # For each circle, create a mask of pixels on its circumference
        for i, circle in enumerate(self.circles):
            cx, cy = circle['center']
            radius = circle['radius']
            color = np.array(circle['color'])
            
            # Calculate distance from circle center for all pixels at once
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            
            # Create mask for pixels on the circumference
            on_edge = (distances >= radius - self.edge_thickness) & (distances <= radius)
            
            # Add quantization: only color pixels if distance from edge is divisible by 2
            # Distance from outer edge of circumference
            distance_from_edge = radius - distances
            # Quantize to integers and check if divisible by 3
            quantized_edge_distance = np.floor(distance_from_edge).astype(int)
            divisible_by_2 = (quantized_edge_distance % 3) == 0
            
            # Combine edge mask with divisibility requirement
            on_edge_quantized = on_edge & divisible_by_2
            
            # Count overlaps using quantized edge mask
            new_overlaps = on_edge_quantized & (circumference_count > 0)
            circumference_count += on_edge_quantized.astype(int)
            
            # Update colors where we have new overlaps (2+ circumferences)
            if np.any(new_overlaps):
                # Use the color of the current circle for new overlaps
                image[new_overlaps] = color
        
        # For areas with only one circumference, don't show any color (keep dark grey)
        single_circumference = circumference_count == 1
        image[single_circumference] = [0.1, 0.1, 0.1]  # Dark grey
        
        # Convert to 0-255 range for display
        self.image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        end_time = time.time()
        print(f"Image generation complete! Time taken: {end_time - start_time:.2f} seconds")
    
    def display_and_save(self):
        """Display and save the image"""
        # Create figure with exact pixel dimensions
        dpi = 100
        fig_width = self.width / dpi
        fig_height = self.height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display image
        ax.imshow(self.image, origin='upper')
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)  # Flip Y axis for correct orientation
        ax.axis('off')  # Remove axes
        
        # Remove margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save image
        filename = f"overlapping_circles_{self.num_circles}circles_{self.width}x{self.height}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
        
        print(f"Image saved as: {filename}")
        print("\nCircle details:")
        for i, circle in enumerate(self.circles):
            cx, cy = circle['center']
            r = circle['radius']
            color = circle['color']
            print(f"Circle {i+1}: Center({cx:.0f}, {cy:.0f}), Radius={r:.0f}, Color=RGB{color}")
        
        # Show image
        plt.show()

def main():
    """Main function to create and display the visualization"""
    print("Overlapping Circles with Distance-Based Color Interpolation")
    print("=" * 60)
    
    # Create visualizer
    visualizer = CircleOverlapVisualizer()
    
    # Display and save
    visualizer.display_and_save()

if __name__ == "__main__":
    main()
