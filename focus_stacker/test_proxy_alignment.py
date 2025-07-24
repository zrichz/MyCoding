#!/usr/bin/env python3
"""Test proxy-based alignment performance."""

import cv2
import numpy as np
import time
from image_alignment import ImageAligner

def create_test_images(width=2000, height=1500, count=3):
    """Create test images with some artificial displacement."""
    print(f"Creating {count} test images ({width}x{height})...")
    
    base_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structured content
    cv2.rectangle(base_image, (width//4, height//4), (3*width//4, 3*height//4), (255, 255, 255), 20)
    cv2.circle(base_image, (width//2, height//2), 200, (0, 255, 0), 30)
    
    images = [base_image]
    
    # Create displaced versions
    for i in range(1, count):
        # Small random displacement
        dx = np.random.randint(-50, 50)
        dy = np.random.randint(-50, 50)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        displaced = cv2.warpAffine(base_image, M, (width, height))
        images.append(displaced)
    
    return images

def test_alignment_speed():
    """Test alignment speed with and without proxy."""
    print("Testing ECC alignment speed...\n")
    
    # Create test images
    images = create_test_images(2000, 1500, 3)
    
    print("=" * 60)
    print("Test 1: Full-resolution ECC alignment")
    print("=" * 60)
    
    start_time = time.time()
    try:
        aligned_full, align_time = ImageAligner.align_images_ecc(
            images, use_proxy=False, progress_callback=lambda p, m: print(f"  {m}")
        )
        full_time = time.time() - start_time
        print(f"✓ Full-resolution alignment completed in {full_time:.2f} seconds\n")
    except Exception as e:
        print(f"✗ Full-resolution alignment failed: {e}\n")
        full_time = 999
    
    print("=" * 60)
    print("Test 2: Proxy-based ECC alignment (25% scale)")
    print("=" * 60)
    
    start_time = time.time()
    try:
        aligned_proxy, align_time = ImageAligner.align_images_ecc(
            images, use_proxy=True, proxy_scale=0.25, progress_callback=lambda p, m: print(f"  {m}")
        )
        proxy_time = time.time() - start_time
        print(f"✓ Proxy-based alignment completed in {proxy_time:.2f} seconds\n")
        
        # Calculate speedup
        if full_time < 999:
            speedup = full_time / proxy_time
            print(f"🚀 Speedup: {speedup:.1f}x faster with proxy alignment!")
        
    except Exception as e:
        print(f"✗ Proxy-based alignment failed: {e}\n")
    
    print("\n" + "=" * 60)
    print("Saving test results...")
    
    # Save results for visual comparison
    if 'aligned_full' in locals():
        cv2.imwrite('test_full_aligned.png', aligned_full[1])
        print("✓ Saved full-resolution result")
    
    if 'aligned_proxy' in locals():
        cv2.imwrite('test_proxy_aligned.png', aligned_proxy[1])
        print("✓ Saved proxy-based result")
    
    print("✓ Test completed!")

if __name__ == "__main__":
    test_alignment_speed()
