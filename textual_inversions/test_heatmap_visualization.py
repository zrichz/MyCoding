#!/usr/bin/env python3
"""
Test script to verify the quantized heatmap visualization works correctly
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Test configuration
HEATMAP_COLORS = ['#FF0000', "#9E560D", '#FFFF00', "#0044FF", '#00FF00', 
                  "#FF74EF", "#FF7B00", '#0080FF', "#ADADB3", "#5800AF"]

def test_quantized_colormap():
    """Test the quantized colormap with BoundaryNorm"""
    
    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(36, 24) * 0.5  # Random data similar to TI vectors
    
    # Create quantized colormap
    n_colors = len(HEATMAP_COLORS)
    quantized_cmap = mcolors.ListedColormap(HEATMAP_COLORS)
    
    # Create quantized normalization
    data_min, data_max = np.min(test_data), np.max(test_data)
    boundaries = np.linspace(data_min, data_max, n_colors + 1)
    norm = mcolors.BoundaryNorm(boundaries, quantized_cmap.N)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Regular colormap
    im1 = ax1.imshow(test_data, cmap='viridis', aspect='auto')
    ax1.set_title('Regular Colormap (viridis)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Quantized colormap
    im2 = ax2.imshow(test_data, cmap=quantized_cmap, norm=norm, aspect='auto')
    ax2.set_title(f'Quantized Colormap ({n_colors} colors)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.suptitle('Heatmap Visualization Test: Regular vs Quantized', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Test completed!")
    print(f"   Data range: [{data_min:.3f}, {data_max:.3f}]")
    print(f"   Number of colors: {n_colors}")
    print(f"   Color boundaries: {boundaries}")

if __name__ == "__main__":
    test_quantized_colormap()
