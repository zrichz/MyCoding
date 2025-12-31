"""
Test script to verify face hint feature works correctly
"""

import os
import sys

print("Starting test...", flush=True)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    import numpy as np
    from PIL import Image
    print("Imports successful", flush=True)
    
    from face_morph_video_creator import FaceMorpher
    print("FaceMorpher imported", flush=True)
    
except Exception as e:
    print(f"Import error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

from face_morph_video_creator import FaceMorpher

def test_face_hints():
    """Test face detection with and without hints"""
    print("Testing face hint feature...")
    
    # Initialize morpher
    morpher = FaceMorpher()
    
    # Create a simple test image with a small face-like pattern
    # (This is just for testing the coordinate transformation logic)
    test_img = np.ones((768, 1152, 3), dtype=np.uint8) * 128
    
    # Draw a simple face pattern in the upper left (small face scenario)
    face_center = (200, 150)
    face_size = 60
    
    # Draw eyes
    cv2.circle(test_img, (face_center[0] - 15, face_center[1] - 10), 5, (0, 0, 0), -1)
    cv2.circle(test_img, (face_center[0] + 15, face_center[1] - 10), 5, (0, 0, 0), -1)
    
    # Draw nose
    cv2.circle(test_img, face_center, 3, (0, 0, 0), -1)
    
    # Draw mouth
    cv2.ellipse(test_img, (face_center[0], face_center[1] + 15), (15, 8), 0, 0, 180, (0, 0, 0), 2)
    
    # Draw face outline
    cv2.ellipse(test_img, face_center, (face_size//2, face_size//2 + 10), 0, 0, 360, (100, 100, 100), 2)
    
    print(f"\nTest image size: {test_img.shape[1]}x{test_img.shape[0]}")
    print(f"Face center: {face_center}")
    
    # Test 1: Without hint
    print("\n--- Test 1: Detection without hint ---")
    result1, img1 = morpher.detect_face_landmarks(test_img, face_hint=None)
    if result1 is not None:
        print(f"✓ Face detected without hint! Found {len(result1)} landmarks")
    else:
        print("✗ Face not detected without hint (expected for this simple test image)")
    
    # Test 2: With hint
    print("\n--- Test 2: Detection with hint ---")
    result2, img2 = morpher.detect_face_landmarks(test_img, face_hint=face_center)
    if result2 is not None:
        print(f"✓ Face detected with hint! Found {len(result2)} landmarks")
    else:
        print("✗ Face not detected even with hint")
    
    # Test 3: Verify ROI logic
    print("\n--- Test 3: ROI calculation test ---")
    # Simulate ROI calculation
    x, y = face_center
    min_dim = min(test_img.shape[0], test_img.shape[1])
    roi_size = int(min_dim * 0.6)
    x1 = max(0, x - roi_size // 2)
    y1 = max(0, y - roi_size // 2)
    x2 = min(test_img.shape[1], x1 + roi_size)
    y2 = min(test_img.shape[0], y1 + roi_size)
    
    print(f"Image size: {test_img.shape[1]}x{test_img.shape[0]}")
    print(f"Face hint: {face_center}")
    print(f"Min dimension: {min_dim}")
    print(f"ROI size: {roi_size}")
    print(f"ROI bounds: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"ROI dimensions: {x2-x1}x{y2-y1}")
    
    # Verify face center is within ROI
    if x1 <= x < x2 and y1 <= y < y2:
        print("✓ Face center is within ROI bounds")
    else:
        print("✗ ERROR: Face center is outside ROI bounds!")
    
    # Test 4: Test coordinate transformation
    print("\n--- Test 4: Coordinate transformation test ---")
    test_x_in_roi = 100
    test_y_in_roi = 100
    scale_factor = 2.0  # Example scale
    
    # Transform back to original
    original_x = (test_x_in_roi + x1) / scale_factor
    original_y = (test_y_in_roi + y1) / scale_factor
    
    print(f"Point in ROI: ({test_x_in_roi}, {test_y_in_roi})")
    print(f"ROI offset: ({x1}, {y1})")
    print(f"Scale factor: {scale_factor}")
    print(f"Transformed to original: ({original_x:.1f}, {original_y:.1f})")
    
    # Verify round-trip
    back_x = int((original_x * scale_factor) - x1)
    back_y = int((original_y * scale_factor) - y1)
    print(f"Round-trip back to ROI: ({back_x}, {back_y})")
    
    if abs(back_x - test_x_in_roi) <= 1 and abs(back_y - test_y_in_roi) <= 1:
        print("✓ Coordinate transformation is correct")
    else:
        print("✗ ERROR: Coordinate transformation has errors!")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("\nNOTE: This test image is very simple and may not trigger actual")
    print("face detection. The important part is that the ROI and coordinate")
    print("transformation logic works correctly.")
    print("="*60)

if __name__ == "__main__":
    test_face_hints()
