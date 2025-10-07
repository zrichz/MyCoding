#!/usr/bin/env python3
"""
Test color palette generation for biomorph generator
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

class TestColorPalettes:
    def __init__(self):
        self.palettes = ["rainbow", "fire", "ocean", "plasma", "sunset", "forest"]
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        import math
        
        if s == 0:
            return v, v, v
            
        h *= 6.0
        i = int(h)
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q
    
    def generate_color_palette(self, palette_name, num_colors):
        """Generate a smooth color palette - copied from biomorph_generator.py"""
        colors = []
        
        if palette_name == "rainbow":
            # Rainbow: Red -> Orange -> Yellow -> Green -> Blue -> Purple
            for i in range(num_colors):
                hue = i / num_colors * 300  # 0 to 300 degrees (avoiding red repeat)
                sat = 1.0
                val = 1.0
                r, g, b = self.hsv_to_rgb(hue/360, sat, val)
                colors.append((int(r*255), int(g*255), int(b*255)))
                
        elif palette_name == "fire":
            # Fire: Black -> Red -> Orange -> Yellow -> White
            for i in range(num_colors):
                t = i / (num_colors - 1)
                if t < 0.25:
                    # Black to Red
                    r = int(255 * (t * 4))
                    g = 0
                    b = 0
                elif t < 0.5:
                    # Red to Orange
                    r = 255
                    g = int(255 * ((t - 0.25) * 4))
                    b = 0
                elif t < 0.75:
                    # Orange to Yellow
                    r = 255
                    g = 255
                    b = int(255 * ((t - 0.5) * 4))
                else:
                    # Yellow to White
                    r = 255
                    g = 255
                    b = 255
                colors.append((r, g, b))
                
        elif palette_name == "ocean":
            # Ocean: Dark Blue -> Light Blue -> Cyan -> White
            for i in range(num_colors):
                t = i / (num_colors - 1)
                r = int(255 * t * 0.3)
                g = int(255 * (0.3 + t * 0.7))
                b = int(255 * (0.5 + t * 0.5))
                colors.append((r, g, b))
                
        return colors
    
    def test_palette_generation(self):
        """Test generating color palettes and save sample strips"""
        print("Testing color palette generation...")
        
        for palette_name in ["rainbow", "fire", "ocean"]:
            print(f"Testing {palette_name} palette...")
            
            # Generate a palette with 256 colors
            palette = self.generate_color_palette(palette_name, 256)
            
            if len(palette) == 256:
                print(f"✅ {palette_name}: Generated {len(palette)} colors")
                
                # Create a color strip image for visual verification
                strip_height = 50
                strip_width = 256
                strip_data = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)
                
                for x in range(strip_width):
                    color = palette[x]
                    strip_data[:, x] = color
                
                # Save the strip
                strip_image = Image.fromarray(strip_data)
                strip_image.save(f"test_{palette_name}_palette.png")
                print(f"   Saved test_{palette_name}_palette.png")
            else:
                print(f"❌ {palette_name}: Expected 256 colors, got {len(palette)}")
        
        print("Color palette test complete!")

if __name__ == "__main__":
    tester = TestColorPalettes()
    tester.test_palette_generation()
