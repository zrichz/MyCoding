#!/usr/bin/env python3
"""
Test script to verify the diagonal banding fix in biomorph color mapping
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def test_spatial_variations():
    """Test the old vs new spatial variation methods"""
    width, height = 400, 300
    palette_size = 256
    
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    
    # OLD METHOD (causes banding)
    print("Testing OLD method (linear combinations)...")
    old_wave1 = ((x_coords * 0.02 + y_coords * 0.03) * palette_size).astype(int) % palette_size
    old_wave2 = ((x_coords * 0.017 + y_coords * 0.023) * palette_size).astype(int) % palette_size
    old_spatial = (old_wave1 + old_wave2) // 2
    old_result = old_spatial.reshape(height, width)
    
    # NEW METHOD (sine/cosine waves)
    print("Testing NEW method (trigonometric functions)...")
    norm_x = x_coords / width
    norm_y = y_coords / height
    
    wave1 = np.sin(norm_x * 2 * math.pi * 3.7 + norm_y * 2 * math.pi * 2.3) * 0.5 + 0.5
    wave2 = np.cos(norm_x * 2 * math.pi * 1.9 + norm_y * 2 * math.pi * 4.1) * 0.5 + 0.5
    wave3 = np.sin(norm_x * 2 * math.pi * 5.1 - norm_y * 2 * math.pi * 1.7) * 0.5 + 0.5
    
    new_spatial = ((wave1 + wave2 + wave3) / 3 * palette_size).astype(int) % palette_size
    new_result = new_spatial.reshape(height, width)
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Old method
    im1 = ax1.imshow(old_result, cmap='viridis', aspect='equal')
    ax1.set_title('OLD: Linear Combinations\n(Shows diagonal banding)')
    ax1.axis('off')
    
    # New method
    im2 = ax2.imshow(new_result, cmap='viridis', aspect='equal')
    ax2.set_title('NEW: Trigonometric Waves\n(Smooth, no banding)')
    ax2.axis('off')
    
    # Difference
    diff = np.abs(old_result.astype(float) - new_result.astype(float))
    im3 = ax3.imshow(diff, cmap='hot', aspect='equal')
    ax3.set_title('Difference\n(Red = Major changes)')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/rich/MyCoding/image_processors/banding_fix_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison saved as: banding_fix_comparison.png")
    
    # Analyze patterns
    print(f"\nOLD method stats:")
    print(f"  Min value: {old_result.min()}, Max value: {old_result.max()}")
    print(f"  Std deviation: {old_result.std():.2f}")
    print(f"  Unique values: {len(np.unique(old_result))}")
    
    print(f"\nNEW method stats:")
    print(f"  Min value: {new_result.min()}, Max value: {new_result.max()}")
    print(f"  Std deviation: {new_result.std():.2f}")
    print(f"  Unique values: {len(np.unique(new_result))}")
    
    # Check for diagonal patterns (simple test)
    old_diagonal_variance = np.var(np.diag(old_result))
    new_diagonal_variance = np.var(np.diag(new_result))
    
    print(f"\nDiagonal pattern analysis:")
    print(f"  OLD diagonal variance: {old_diagonal_variance:.2f}")
    print(f"  NEW diagonal variance: {new_diagonal_variance:.2f}")
    
    if new_diagonal_variance < old_diagonal_variance * 0.5:
        print("  ✅ NEW method significantly reduces diagonal patterns!")
    else:
        print("  ❌ NEW method may still have some diagonal patterns")
        
    return old_result, new_result

if __name__ == "__main__":
    print("🔍 Testing diagonal banding fix...")
    old, new = test_spatial_variations()
    print("\n✨ Test complete!")
