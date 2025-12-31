"""Quick test to verify dlib is set up correctly"""
import sys
import os

print("Testing dlib setup...")
print("-" * 50)

# Test 1: Import dlib
try:
    import dlib
    print("✓ dlib imported successfully")
    print(f"  Version: {dlib.__version__}")
except ImportError as e:
    print(f"✗ Failed to import dlib: {e}")
    sys.exit(1)

# Test 2: Check model file exists
model_path = "shape_predictor_68_face_landmarks.dat"
if os.path.exists(model_path):
    print(f"✓ Model file found: {model_path}")
    print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
else:
    print(f"✗ Model file not found: {model_path}")
    sys.exit(1)

# Test 3: Load detector and predictor
try:
    detector = dlib.get_frontal_face_detector()
    print("✓ Face detector loaded")
    
    predictor = dlib.shape_predictor(model_path)
    print("✓ Landmark predictor loaded")
except Exception as e:
    print(f"✗ Failed to load detector/predictor: {e}")
    sys.exit(1)

# Test 4: Import face_morph_video_creator
try:
    from face_morph_video_creator import FaceMorphVideoCreator, DLIB_AVAILABLE
    print("✓ face_morph_video_creator imported")
    print(f"  DLIB_AVAILABLE: {DLIB_AVAILABLE}")
except Exception as e:
    print(f"✗ Failed to import face_morph_video_creator: {e}")
    sys.exit(1)

# Test 5: Initialize FaceMorphVideoCreator
try:
    creator = FaceMorphVideoCreator()
    print("✓ FaceMorphVideoCreator initialized")
    print(f"  Using dlib: {creator.use_dlib}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

print("-" * 50)
print("✓ All tests passed! dlib is ready to use.")
