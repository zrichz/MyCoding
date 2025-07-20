"""
Test and Demo Script for Focus Stacker
======================================
This script demonstrates the capabilities of the World's Best Focus Stacker.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focus_stacking_algorithms import FocusStackingAlgorithms
from image_alignment import ImageAligner, QualityAssessment
from utils import ImageUtils, ValidationUtils, PerformanceUtils


def create_test_images():
    """Create synthetic test images with different focus points."""
    print("Creating synthetic test images...")
    
    # Create base image
    width, height = 800, 600
    center_x, center_y = width // 2, height // 2
    
    # Create pattern
    y, x = np.ogrid[:height, :width]
    
    images = []
    focus_points = [
        (center_x - 150, center_y - 100),  # Top-left focus
        (center_x, center_y),              # Center focus
        (center_x + 150, center_y + 100),  # Bottom-right focus
    ]
    
    for i, (fx, fy) in enumerate(focus_points):
        # Create distance map from focus point
        dist_from_focus = np.sqrt((x - fx)**2 + (y - fy)**2)
        
        # Create sharp pattern at focus point
        pattern = np.sin(x * 0.02) * np.cos(y * 0.02) * 255
        pattern = (pattern + 255) / 2  # Normalize to 0-255
        
        # Apply distance-based blur
        blur_strength = dist_from_focus / 100.0
        blur_strength = np.clip(blur_strength, 0, 10)
        
        # Create image with varying sharpness
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add pattern to all channels
        for c in range(3):
            channel = pattern.copy()
            
            # Apply position-dependent blur
            for y_pos in range(0, height, 50):
                for x_pos in range(0, width, 50):
                    y_end = min(y_pos + 50, height)
                    x_end = min(x_pos + 50, width)
                    
                    local_blur = blur_strength[y_pos:y_end, x_pos:x_end].mean()
                    if local_blur > 0.5:
                        kernel_size = int(local_blur * 2) | 1  # Ensure odd
                        kernel_size = min(kernel_size, 15)
                        
                        roi = channel[y_pos:y_end, x_pos:x_end]
                        roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                        channel[y_pos:y_end, x_pos:x_end] = roi
            
            image[:, :, c] = channel.astype(np.uint8)
        
        # Add some noise and color variation
        noise = np.random.randint(-10, 10, image.shape)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add color tint to distinguish images
        if i == 0:
            image[:, :, 0] = np.clip(image[:, :, 0] * 1.1, 0, 255)  # More red
        elif i == 2:
            image[:, :, 2] = np.clip(image[:, :, 2] * 1.1, 0, 255)  # More blue
        
        images.append(image)
        
        # Save test image
        cv2.imwrite(f"test_image_{i+1}.png", image)
        print(f"  Created test_image_{i+1}.png (focus at {fx}, {fy})")
    
    return images


def test_focus_stacking_algorithms():
    """Test all focus stacking algorithms."""
    print("\n" + "="*60)
    print("Testing Focus Stacking Algorithms")
    print("="*60)
    
    # Create or load test images
    images = create_test_images()
    
    # Validate images
    is_valid, message = ValidationUtils.validate_images(images)
    print(f"Image validation: {message}")
    
    if not is_valid:
        return
    
    # Test each algorithm
    algorithms = [
        ("Laplacian Pyramid", lambda imgs: FocusStackingAlgorithms.laplacian_pyramid_stack(imgs, levels=5, sigma=1.0)),
        ("Gradient-based", lambda imgs: FocusStackingAlgorithms.gradient_based_stack(imgs)),
        ("Variance-based", lambda imgs: FocusStackingAlgorithms.variance_based_stack(imgs)),
    ]
    
    results = {}
    
    for name, algorithm in algorithms:
        print(f"\nTesting {name}...")
        try:
            # Measure performance
            import time
            start_time = time.time()
            
            result = algorithm(images)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Save result
            output_file = f"result_{name.lower().replace('-', '_').replace(' ', '_')}.png"
            cv2.imwrite(output_file, result)
            
            # Calculate quality metrics
            quality = QualityAssessment.assess_stack_quality(images, result)
            
            results[name] = {
                'time': processing_time,
                'quality': quality,
                'output_file': output_file
            }
            
            print(f"  ✓ Completed in {processing_time:.2f} seconds")
            print(f"  ✓ Focus measure: {quality['stacked_focus_measure']:.2f}")
            print(f"  ✓ Improvement ratio: {quality['improvement_ratio']:.2f}x")
            print(f"  ✓ Saved to {output_file}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[name] = {'error': str(e)}
    
    return results


def test_image_alignment():
    """Test image alignment algorithms."""
    print("\n" + "="*60)
    print("Testing Image Alignment")
    print("="*60)
    
    # Create test images with slight shifts
    base_image = create_test_images()[1]  # Use center-focused image as base
    
    shifted_images = [base_image]  # Original image
    
    # Create shifted versions
    shifts = [(5, 3), (-3, 7), (2, -4)]
    for dx, dy in shifts:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(base_image, M, (base_image.shape[1], base_image.shape[0]))
        shifted_images.append(shifted)
    
    print(f"Created {len(shifted_images)} images with known shifts")
    
    # Test alignment methods
    alignment_methods = [
        ("ECC", lambda imgs: ImageAligner.align_images_ecc(imgs)),
        ("Phase Correlation", lambda imgs: ImageAligner.align_images_phase_correlation(imgs)),
        ("Feature-based", lambda imgs: ImageAligner.align_images_feature_based(imgs)),
        ("Auto", lambda imgs: ImageAligner.auto_align(imgs)),
    ]
    
    for name, method in alignment_methods:
        print(f"\nTesting {name} alignment...")
        try:
            import time
            start_time = time.time()
            
            aligned_images = method(shifted_images)
            
            end_time = time.time()
            print(f"  ✓ Completed in {end_time - start_time:.2f} seconds")
            
            # Save first aligned image for visual inspection
            if aligned_images:
                cv2.imwrite(f"aligned_{name.lower().replace(' ', '_')}.png", aligned_images[1])
                print(f"  ✓ Saved aligned image to aligned_{name.lower().replace(' ', '_')}.png")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_performance_analysis():
    """Test performance analysis utilities."""
    print("\n" + "="*60)
    print("Performance Analysis")
    print("="*60)
    
    # Create test images of different sizes
    test_sizes = [(400, 300), (800, 600), (1600, 1200)]
    
    for width, height in test_sizes:
        # Create dummy images
        images = [np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(5)]
        
        # Estimate memory usage
        memory_usage = PerformanceUtils.estimate_memory_usage(images)
        should_tile = PerformanceUtils.should_tile_processing(images)
        
        print(f"\nImage size: {width}x{height}")
        print(f"  Estimated memory usage: {memory_usage / (1024*1024):.1f} MB")
        print(f"  Should tile processing: {should_tile}")
        
        if should_tile:
            tile_h, tile_w = PerformanceUtils.calculate_optimal_tile_size((height, width))
            print(f"  Optimal tile size: {tile_w}x{tile_h}")


def test_quality_assessment():
    """Test quality assessment functions."""
    print("\n" + "="*60)
    print("Quality Assessment")
    print("="*60)
    
    # Create images with different focus qualities
    sharp_image = create_test_images()[1]  # Center-focused
    blurry_image = cv2.GaussianBlur(sharp_image, (15, 15), 0)
    
    images = [sharp_image, blurry_image]
    image_names = ["Sharp", "Blurry"]
    
    for name, image in zip(image_names, images):
        focus_laplacian = QualityAssessment.calculate_focus_measure(image, 'laplacian')
        focus_gradient = QualityAssessment.calculate_focus_measure(image, 'gradient')
        focus_variance = QualityAssessment.calculate_focus_measure(image, 'variance')
        
        print(f"\n{name} image quality:")
        print(f"  Laplacian focus measure: {focus_laplacian:.2f}")
        print(f"  Gradient focus measure: {focus_gradient:.2f}")
        print(f"  Variance focus measure: {focus_variance:.2f}")


def run_comprehensive_demo():
    """Run a comprehensive demonstration of all features."""
    print("🔥 World's Best Focus Stacker - Comprehensive Demo 🔥")
    print("="*60)
    
    try:
        # Test 1: Create and validate test data
        print("\n1. Creating synthetic test data...")
        images = create_test_images()
        print(f"   ✓ Created {len(images)} test images")
        
        # Test 2: Validate images
        print("\n2. Validating images...")
        is_valid, message = ValidationUtils.validate_images(images)
        print(f"   ✓ {message}")
        
        # Test 3: Test focus stacking
        print("\n3. Testing focus stacking algorithms...")
        stacking_results = test_focus_stacking_algorithms()
        
        # Test 4: Test alignment
        print("\n4. Testing image alignment...")
        test_image_alignment()
        
        # Test 5: Performance analysis
        print("\n5. Performance analysis...")
        test_performance_analysis()
        
        # Test 6: Quality assessment
        print("\n6. Quality assessment...")
        test_quality_assessment()
        
        # Summary
        print("\n" + "="*60)
        print("Demo Summary")
        print("="*60)
        
        print("\nStacking Algorithm Results:")
        for algorithm, result in stacking_results.items():
            if 'error' in result:
                print(f"  ❌ {algorithm}: {result['error']}")
            else:
                print(f"  ✅ {algorithm}: {result['time']:.2f}s, "
                      f"Quality: {result['quality']['improvement_ratio']:.2f}x")
        
        print("\nOutput Files Created:")
        output_files = []
        for i in range(3):
            output_files.append(f"test_image_{i+1}.png")
        
        for algorithm, result in stacking_results.items():
            if 'output_file' in result:
                output_files.append(result['output_file'])
        
        for file in output_files:
            if os.path.exists(file):
                print(f"  📁 {file}")
        
        print(f"\n🎉 Demo completed successfully!")
        print("   You can now examine the output images to see the focus stacking results.")
        print("   Try running 'python main.py --gui' to use the graphical interface!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("Please run this script from the focus_stacker directory")
        sys.exit(1)
    
    # Run the demo
    run_comprehensive_demo()
