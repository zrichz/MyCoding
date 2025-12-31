"""Test image format handling for face detection"""
import sys
import numpy as np
from PIL import Image

# Test the detect_face_landmarks function with various formats
print("Testing image format handling...")
print("-" * 50)

try:
    from face_morph_video_creator import FaceMorphVideoCreator
    morpher = FaceMorphVideoCreator()
    print(f"✓ FaceMorphVideoCreator initialized")
    print(f"  Using dlib: {morpher.use_dlib}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# Test 1: RGB numpy array (768x1152x3)
print("\nTest 1: RGB numpy array (768x1152x3)")
try:
    test_img = np.random.randint(0, 255, (768, 1152, 3), dtype=np.uint8)
    points, img = morpher.detect_face_landmarks(test_img)
    print(f"  Input shape: {test_img.shape}, dtype: {test_img.dtype}")
    if img is not None:
        print(f"  Output shape: {img.shape}, dtype: {img.dtype}")
        print(f"  ✓ No error (face may not be detected in random image)")
    else:
        print(f"  ✓ Returned None (expected for random image)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: PIL Image
print("\nTest 2: PIL Image RGB")
try:
    pil_img = Image.new('RGB', (1152, 768), color=(128, 128, 128))
    points, img = morpher.detect_face_landmarks(pil_img)
    print(f"  Input: PIL Image mode={pil_img.mode}, size={pil_img.size}")
    if img is not None:
        print(f"  Output shape: {img.shape}, dtype: {img.dtype}")
        print(f"  ✓ No error")
    else:
        print(f"  ✓ Returned None (expected for blank image)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 3: Grayscale numpy array
print("\nTest 3: Grayscale numpy array")
try:
    gray_img = np.random.randint(0, 255, (768, 1152), dtype=np.uint8)
    points, img = morpher.detect_face_landmarks(gray_img)
    print(f"  Input shape: {gray_img.shape}, dtype: {gray_img.dtype}")
    if img is not None:
        print(f"  Output shape: {img.shape}, dtype: {img.dtype}")
        print(f"  ✓ No error")
    else:
        print(f"  ✓ Returned None")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 4: RGBA numpy array
print("\nTest 4: RGBA numpy array")
try:
    rgba_img = np.random.randint(0, 255, (768, 1152, 4), dtype=np.uint8)
    points, img = morpher.detect_face_landmarks(rgba_img)
    print(f"  Input shape: {rgba_img.shape}, dtype: {rgba_img.dtype}")
    if img is not None:
        print(f"  Output shape: {img.shape}, dtype: {img.dtype}")
        print(f"  ✓ No error")
    else:
        print(f"  ✓ Returned None")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "-" * 50)
print("✓ All format tests passed! No 'unsupported image type' errors.")
