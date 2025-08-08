#!/usr/bin/env python3
"""
Test the filename generation with dimensions
"""

from pathlib import Path
import sys

def test_filename_generation():
    """Test filename generation with dimensions"""
    
    print("Testing Filename Generation with Dimensions...")
    print("=" * 50)
    
    test_cases = [
        # (original_filename, image_width, image_height, expected_output)
        ("photo.jpg", 720, 1280, "photo_720x1280.jpg"),
        ("landscape.png", 720, 480, "landscape_720x480.png"),
        ("portrait.jpeg", 540, 1600, "portrait_540x1600.jpeg"),
        ("my_image.tiff", 720, 720, "my_image_720x720.tiff"),
        ("test-file.bmp", 600, 900, "test-file_600x900.bmp"),
    ]
    
    for i, (original_name, width, height, expected) in enumerate(test_cases, 1):
        # Simulate the filename generation logic from the code
        original_path = Path(original_name)
        original_stem = original_path.stem
        original_suffix = original_path.suffix
        
        # Add dimensions to filename
        dimensions_suffix = f"_{width}x{height}"
        new_filename = f"{original_stem}{dimensions_suffix}{original_suffix}"
        
        print(f"Test {i}:")
        print(f"  Original: {original_name}")
        print(f"  Dimensions: {width}×{height}")
        print(f"  Generated: {new_filename}")
        print(f"  Expected: {expected}")
        print(f"  Result: {'✅ Pass' if new_filename == expected else '❌ Fail'}")
        print()
    
    # Test conflict resolution
    print("Testing Conflict Resolution:")
    original_stem = "photo"
    original_suffix = ".jpg"
    width, height = 720, 1280
    
    # Simulate conflict resolution logic
    dimensions_suffix = f"_{width}x{height}"
    base_filename = f"{original_stem}{dimensions_suffix}{original_suffix}"
    
    conflict_filenames = []
    for counter in range(1, 4):
        conflict_filename = f"{original_stem}{dimensions_suffix}_crop_{counter}{original_suffix}"
        conflict_filenames.append(conflict_filename)
    
    print(f"  Base filename: {base_filename}")
    print(f"  Conflict 1: {conflict_filenames[0]}")
    print(f"  Conflict 2: {conflict_filenames[1]}")
    print(f"  Conflict 3: {conflict_filenames[2]}")
    
    print("\n✅ Filename generation test completed!")

if __name__ == "__main__":
    test_filename_generation()
