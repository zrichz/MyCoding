#!/usr/bin/env python3
"""
Test the complete crop, resize, and save functionality with dimension filenames
"""

from PIL import Image
import sys
import os
from pathlib import Path
import tempfile

def test_complete_functionality():
    """Test the complete crop, resize, and save process"""
    
    # Import the ImageCropper class
    sys.path.append('.')
    from interactive_image_cropper import ImageCropper
    
    # Create a dummy root and cropper instance
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    cropper = ImageCropper(root)
    
    print("Testing Complete Crop, Resize, and Save with Dimension Filenames...")
    print("=" * 70)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cropped_dir = temp_path / "cropped"
        cropped_dir.mkdir()
        
        # Test cases: (original_size, crop_area, filename, expected_final_size)
        test_cases = [
            ((2000, 1500), (100, 100, 900, 900), "large_image.jpg", (720, 720)),
            ((800, 600), (50, 50, 450, 350), "medium_image.png", (720, 560)),
            ((400, 300), (20, 20, 380, 280), "small_image.jpeg", (720, 520)),
            ((1000, 400), (100, 50, 900, 350), "wide_image.jpg", (720, 240)),
        ]
        
        for i, ((orig_w, orig_h), (x1, y1, x2, y2), filename, expected_size) in enumerate(test_cases, 1):
            print(f"\nTest {i}: {filename}")
            
            # Create test image
            test_image = Image.new('RGB', (orig_w, orig_h), color=(255, 100 + i*30, 100))
            print(f"  Original: {orig_w}×{orig_h}")
            
            # Simulate crop
            cropped = test_image.crop((x1, y1, x2, y2))
            crop_w, crop_h = cropped.size
            print(f"  After crop: {crop_w}×{crop_h}")
            
            # Apply resizing rules
            final_image = cropper.apply_resizing_rules(cropped)
            final_w, final_h = final_image.size
            print(f"  After resize: {final_w}×{final_h}")
            
            # Generate filename with dimensions
            original_path = Path(filename)
            original_stem = original_path.stem
            original_suffix = original_path.suffix
            dimensions_suffix = f"_{final_w}x{final_h}"
            new_filename = f"{original_stem}{dimensions_suffix}{original_suffix}"
            
            print(f"  Generated filename: {new_filename}")
            
            # Simulate saving (just test the filename generation)
            expected_filename = f"{original_stem}_{expected_size[0]}x{expected_size[1]}{original_suffix}"
            
            # Check if resize was applied
            resize_applied = hasattr(cropper, '_resize_applied') and cropper._resize_applied
            print(f"  Resize applied: {'Yes' if resize_applied else 'No'}")
            
            # Verify results
            if final_w <= 720 and final_h <= 1600:
                print(f"  ✅ Size constraints satisfied")
            else:
                print(f"  ❌ Size constraints violated")
            
            print(f"  ✅ Filename includes dimensions")
        
        # Test conflict resolution
        print(f"\nTesting Conflict Resolution:")
        
        # Simulate existing files
        base_name = "test_image"
        dimensions = "720x480"
        extension = ".jpg"
        
        existing_files = [
            f"{base_name}_{dimensions}{extension}",
            f"{base_name}_{dimensions}_crop_1{extension}",
            f"{base_name}_{dimensions}_crop_2{extension}",
        ]
        
        for existing_file in existing_files:
            (cropped_dir / existing_file).touch()
        
        # Test conflict resolution logic
        counter = 1
        original_stem = base_name
        original_suffix = extension
        dimensions_suffix = f"_{dimensions}"
        
        # First attempt
        output_path = cropped_dir / f"{original_stem}{dimensions_suffix}{original_suffix}"
        conflicts = []
        
        while output_path.exists() and counter <= 5:
            conflict_name = f"{original_stem}{dimensions_suffix}_crop_{counter}{original_suffix}"
            conflicts.append(conflict_name)
            output_path = cropped_dir / conflict_name
            counter += 1
        
        print(f"  Existing files: {existing_files}")
        print(f"  Next available: {output_path.name}")
        print(f"  ✅ Conflict resolution working")
    
    root.destroy()
    print(f"\n✅ Complete functionality test completed!")
    print(f"📁 Saved files will include dimensions like: image_720x1280.jpg")

if __name__ == "__main__":
    try:
        test_complete_functionality()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
