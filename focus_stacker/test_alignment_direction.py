#!/usr/bin/env python3
"""Test alignment direction to ensure offsets are corrected, not exacerbated."""

import cv2
import numpy as np
from image_alignment import ImageAligner

def create_test_with_known_offset():
    """Create test images with a known, controllable offset."""
    print("Creating test images with known offset...")
    
    # Create a base image with distinctive features
    base = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add some distinctive features
    cv2.rectangle(base, (100, 100), (200, 200), (255, 255, 255), -1)  # White square
    cv2.circle(base, (400, 200), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(base, (300, 50), (500, 100), (0, 0, 255), -1)  # Red rectangle
    
    # Create displaced version with KNOWN offset
    offset_x = 30  # Move 30 pixels right
    offset_y = 20  # Move 20 pixels down
    
    print(f"Applied offset: dx={offset_x}, dy={offset_y}")
    
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    displaced = cv2.warpAffine(base, M, (600, 400))
    
    # Save for visual inspection
    cv2.imwrite('test_reference.png', base)
    cv2.imwrite('test_displaced.png', displaced)
    
    return [base, displaced], (offset_x, offset_y)

def test_alignment_correction():
    """Test that alignment actually corrects the offset."""
    print("=" * 60)
    print("TESTING ALIGNMENT DIRECTION")
    print("=" * 60)
    
    # Create test images with known offset
    images, (true_offset_x, true_offset_y) = create_test_with_known_offset()
    
    print(f"\nTrue offset that needs correction: dx={true_offset_x}, dy={true_offset_y}")
    
    # Test different alignment methods
    methods = [
        ("ECC (full-res)", lambda imgs: ImageAligner.align_images_ecc(imgs, use_proxy=False)),
        ("ECC (proxy)", lambda imgs: ImageAligner.align_images_ecc(imgs, use_proxy=True, proxy_scale=0.5)),
    ]
    
    for method_name, align_func in methods:
        print(f"\n--- Testing {method_name} ---")
        
        try:
            aligned_images, align_time = align_func(images)
            
            # Save aligned result
            filename = f'test_aligned_{method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
            cv2.imwrite(filename, aligned_images[1])
            print(f"✓ Alignment completed in {align_time:.2f}s")
            print(f"✓ Saved result: {filename}")
            
            # Calculate difference between reference and aligned image
            diff = cv2.absdiff(images[0], aligned_images[1])
            diff_score = np.mean(diff)
            
            # Calculate difference between reference and original displaced image
            orig_diff = cv2.absdiff(images[0], images[1])
            orig_diff_score = np.mean(orig_diff)
            
            print(f"  Original difference score: {orig_diff_score:.2f}")
            print(f"  Aligned difference score:  {diff_score:.2f}")
            
            if diff_score < orig_diff_score:
                improvement = ((orig_diff_score - diff_score) / orig_diff_score) * 100
                print(f"  ✓ GOOD: Alignment improved by {improvement:.1f}%")
            else:
                worsening = ((diff_score - orig_diff_score) / orig_diff_score) * 100
                print(f"  ✗ BAD: Alignment made it worse by {worsening:.1f}%")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print(f"\n{'='*60}")
    print("Visual inspection files created:")
    print("- test_reference.png     (reference image)")
    print("- test_displaced.png     (displaced image - what we're trying to fix)")
    print("- test_aligned_*.png     (alignment results)")
    print("Compare these visually to verify alignment direction!")

if __name__ == "__main__":
    test_alignment_correction()
