#!/usr/bin/env python3
"""
Quick test to verify biomorph generator creates 1200x800 images
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from biomorph_generator import BiomorphGenerator
import numpy as np

def test_dimensions():
    """Test that the biomorph generator creates 1200x800 images"""
    print("Testing biomorph generator dimensions...")
    
    # Create generator instance
    generator = BiomorphGenerator(None)  # No root window for headless testing
    
    # Test the core generation function
    width = generator.image_width.get()
    height = generator.image_height.get()
    
    print(f"Configured dimensions: {width} x {height}")
    
    # Generate a small test fractal
    fractal_data = generator.generate_biomorph(
        width, height, 
        const_real=0.5, const_imag=0.0,
        zoom=2.5, center_x=0.0, center_y=0.0,
        max_iter=50, escape_radius=10.0
    )
    
    if fractal_data is not None:
        actual_height, actual_width = fractal_data.shape
        print(f"Generated image dimensions: {actual_width} x {actual_height}")
        
        if actual_width == 1200 and actual_height == 800:
            print("✅ SUCCESS: Image dimensions are correct (1200x800)")
            return True
        else:
            print(f"❌ FAILED: Expected 1200x800, got {actual_width}x{actual_height}")
            return False
    else:
        print("❌ FAILED: No fractal data generated")
        return False

if __name__ == "__main__":
    test_dimensions()
