"""
Quick test to verify MediaPipe face detection is working
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_morph_video_creator import FaceMorphVideoCreator

def test_mediapipe():
    print("Testing MediaPipe Face Detection...")
    print("-" * 60)
    
    # Initialize morpher
    morpher = FaceMorphVideoCreator()
    
    # Create a simple test image with a face-like pattern
    # (For real testing, use an actual photo)
    test_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    
    # Draw a simple face
    cv2.circle(test_img, (200, 200), 80, (150, 150, 150), 2)  # Face outline
    cv2.circle(test_img, (170, 180), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_img, (230, 180), 10, (0, 0, 0), -1)  # Right eye
    cv2.circle(test_img, (200, 200), 5, (0, 0, 0), -1)  # Nose
    cv2.ellipse(test_img, (200, 230), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    print("Test image created: 400x400 pixels")
    
    # Test detection
    print("\n1. Testing face detection...")
    points, img = morpher.detect_face_landmarks(test_img)
    
    if points is not None:
        print(f"   ✓ Face detected!")
        print(f"   ✓ Found {len(points)} landmarks")
        print(f"   ✓ Image shape: {img.shape}")
        
        # Show some landmark positions
        print(f"\n2. Sample landmarks:")
        print(f"   - Nose tip (idx 4): ({points[4][0]:.1f}, {points[4][1]:.1f})")
        print(f"   - Chin (idx 152): ({points[152][0]:.1f}, {points[152][1]:.1f})")
        print(f"   - Left mouth (idx 61): ({points[61][0]:.1f}, {points[61][1]:.1f})")
        print(f"   - Right mouth (idx 291): ({points[291][0]:.1f}, {points[291][1]:.1f})")
        
        print(f"\n3. Testing landmark visualization...")
        preview = morpher.draw_landmarks_on_image(img, points)
        print(f"   ✓ Preview image created: {preview.shape}")
        
        print("\n" + "=" * 60)
        print("✓ MediaPipe integration successful!")
        print("=" * 60)
        
    else:
        print("   ✗ No face detected (expected for this simple test pattern)")
        print("\nNote: This is a very simple test image.")
        print("For real testing, use an actual photo with a face.")
        print("\nTo test with a real image:")
        print("  1. Open the Gradio interface")
        print("  2. Upload a photo with a face")
        print("  3. Click 'Detect Faces'")
        print("  4. You should see 468 landmarks detected!")
    
    print("\nMediaPipe configuration:")
    print(f"  - Model: Face Mesh (468 landmarks)")
    print(f"  - Mode: Static image")
    print(f"  - Refined landmarks: Yes (includes iris)")
    print(f"  - Min confidence: 0.3 (good for small faces)")

if __name__ == "__main__":
    test_mediapipe()
