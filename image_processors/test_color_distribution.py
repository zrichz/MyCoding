#!/usr/bin/env python3
"""
Test the new ultra-aggressive color mapping system
"""
import numpy as np
from PIL import Image

def test_color_distribution():
    """Test how the new color mapping handles low iteration values"""
    print("Testing ultra-aggressive color mapping...")
    
    # Simulate typical biomorph iteration data (mostly 1-5, some up to 20)
    test_iterations = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 7, 12, 20])
    palette_size = 256
    
    print(f"Input iterations: {test_iterations}")
    print(f"Min: {np.min(test_iterations)}, Max: {np.max(test_iterations)}, Median: {np.median(test_iterations)}")
    
    # Apply the ultra-aggressive mapping
    escaped_iterations = test_iterations
    
    # Strategy 1: Extreme stretching for low values (1-10 iterations)
    stretched_values = np.clip(escaped_iterations, 1, 10)
    stretched_indices = ((stretched_values - 1) * palette_size * 0.8 / 9).astype(int)
    
    # Strategy 2: Multi-prime cycling
    cycle1 = ((escaped_iterations * 47) % palette_size)
    cycle2 = ((escaped_iterations * 73) % palette_size)
    cycle3 = ((escaped_iterations * 31) % palette_size)
    multi_cycle = (cycle1 + cycle2 + cycle3) // 3
    
    # Strategy 3: Position-based variation (improved)
    positions = np.arange(len(escaped_iterations))
    spatial_wave1 = ((positions * 3.7) % palette_size).astype(int)
    spatial_wave2 = ((positions * 2.3) % palette_size).astype(int)
    spatial_wave3 = ((positions * 5.1) % palette_size).astype(int)
    spatial_blend = (spatial_wave1 + spatial_wave2 + spatial_wave3) // 3
    
    # Strategy 4: Fractal indices
    fractal_indices = ((escaped_iterations * escaped_iterations * 13) % palette_size)
    
    # Strategy 5: Smooth interpolation
    smooth_offset = (positions % 256) / 256.0 * 20
    smooth_indices = ((escaped_iterations - 1 + smooth_offset) * palette_size / 10).astype(int) % palette_size
    
    # Combine all strategies
    final_indices = (
        stretched_indices * 0.4 +
        multi_cycle * 0.25 +
        spatial_blend * 0.15 +
        fractal_indices * 0.1 +
        smooth_indices * 0.1
    ).astype(int) % palette_size
    
    print(f"\nColor mapping results:")
    print(f"Stretched indices:  {stretched_indices}")
    print(f"Multi-cycle:        {multi_cycle}")
    print(f"Spatial blend:      {spatial_blend}")
    print(f"Fractal indices:    {fractal_indices}")
    print(f"Smooth indices:     {smooth_indices}")
    print(f"Final indices:      {final_indices}")
    
    # Check color distribution
    unique_colors = len(np.unique(final_indices))
    color_range = np.max(final_indices) - np.min(final_indices)
    
    print(f"\nColor distribution analysis:")
    print(f"Unique colors used: {unique_colors} out of {len(test_iterations)} pixels")
    print(f"Color range: {np.min(final_indices)} - {np.max(final_indices)} (span: {color_range})")
    print(f"Average color index: {np.mean(final_indices):.1f}")
    
    if unique_colors >= len(test_iterations) * 0.8:
        print("✅ SUCCESS: High color variety achieved!")
    else:
        print("❌ NEEDS IMPROVEMENT: Low color variety")
    
    if color_range >= palette_size * 0.3:
        print("✅ SUCCESS: Good color range coverage!")
    else:
        print("❌ NEEDS IMPROVEMENT: Limited color range")

if __name__ == "__main__":
    test_color_distribution()
