#!/usr/bin/env python3
"""
Test the image resizing rules implementation
"""

from PIL import Image
import sys
import os

def test_resizing_rules():
    """Test the resizing rules with various image sizes"""
    
    # Import the ImageCropper class to access the resizing method
    sys.path.append('.')
    from interactive_image_cropper import ImageCropper
    
    # Create a dummy root and cropper instance to access the method
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    cropper = ImageCropper(root)
    
    print("Testing Image Resizing Rules...")
    print("=" * 50)
    
    # Test cases: (width, height, expected_behavior)
    test_cases = [
        # Rule 1: Width > 720px should reduce to 720px
        (1000, 800, "Reduce width to 720px"),
        (1500, 600, "Reduce width to 720px"),
        
        # Rule 2: Height > 1600px should reduce to 1600px  
        (600, 2000, "Reduce height to 1600px"),
        (400, 1800, "Reduce height to 1600px"),
        
        # Rule 3: Width < 720px and height < (20/9) * width should increase width to 720px
        (500, 1000, "Increase width to 720px (aspect check)"),
        (600, 1200, "Increase width to 720px (aspect check)"),
        
        # Rule 4: Height < 1600px and width < (9/20) * height should increase width to 720px
        (200, 1000, "Increase width to 720px (height ratio check)"),
        (300, 1400, "Increase width to 720px (height ratio check)"),
        
        # No resizing needed
        (720, 1280, "No resizing needed"),
        (600, 1067, "No resizing needed"),
        (400, 600, "No resizing needed"),
    ]
    
    for i, (width, height, expected) in enumerate(test_cases, 1):
        # Create a test image
        test_image = Image.new('RGB', (width, height), color='red')
        original_size = test_image.size
        
        # Apply resizing rules
        resized_image = cropper.apply_resizing_rules(test_image)
        final_size = resized_image.size
        
        # Check if resize was applied
        resize_applied = hasattr(cropper, '_resize_applied') and cropper._resize_applied
        
        print(f"Test {i}: {width}×{height}")
        print(f"  Expected: {expected}")
        print(f"  Result: {original_size} → {final_size}")
        print(f"  Resized: {'Yes' if resize_applied else 'No'}")
        
        # Verify specific rules
        final_width, final_height = final_size
        
        # Check if rules are satisfied
        rule_violations = []
        
        if final_width > 720:
            rule_violations.append("Width > 720px")
        if final_height > 1600:
            rule_violations.append("Height > 1600px")
        
        # Check aspect ratio rules
        if final_width < 720 and final_height < (20/9) * final_width:
            rule_violations.append("Width < 720px and height < (20/9) * width")
        if final_height < 1600 and final_width < (9/20) * final_height:
            rule_violations.append("Height < 1600px and width < (9/20) * height")
        
        if rule_violations:
            print(f"  ❌ Rule violations: {', '.join(rule_violations)}")
        else:
            print(f"  ✅ All rules satisfied")
        
        print()
    
    root.destroy()
    print("✅ Resizing rules test completed!")

if __name__ == "__main__":
    try:
        test_resizing_rules()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
