#!/usr/bin/env python3
"""
Test the 9:20 minimum aspect ratio constraint in the interactive crop selection.
This test simulates the crop selection logic to verify the aspect ratio enforcement.
"""

def test_aspect_ratio_constraint():
    """Test the 9:20 minimum aspect ratio constraint logic"""
    print("Testing 9:20 minimum aspect ratio constraint...")
    
    # Simulate the constraint logic from update_crop_selection method
    min_aspect_ratio = 9.0 / 20.0  # 0.45
    
    test_cases = [
        # (raw_width, raw_height, expected_min_width)
        (100, 200, 90),   # height=200, min_width=200*0.45=90, raw_width=100 > 90 ✓
        (50, 200, 90),    # height=200, min_width=200*0.45=90, raw_width=50 < 90, should be adjusted to 90
        (45, 100, 45),    # height=100, min_width=100*0.45=45, raw_width=45 = 45 ✓
        (30, 100, 45),    # height=100, min_width=100*0.45=45, raw_width=30 < 45, should be adjusted to 45
        (200, 400, 180),  # height=400, min_width=400*0.45=180, raw_width=200 > 180 ✓
        (100, 400, 180),  # height=400, min_width=400*0.45=180, raw_width=100 < 180, should be adjusted to 180
    ]
    
    for i, (raw_width, raw_height, expected_min_width) in enumerate(test_cases):
        print(f"\nTest case {i+1}: width={raw_width}, height={raw_height}")
        
        adjusted_width = raw_width  # Initialize with original width
        
        if raw_height > 0:
            min_width_for_height = raw_height * min_aspect_ratio
            adjusted_width = max(raw_width, min_width_for_height)
            
            print(f"  Required minimum width: {min_width_for_height:.1f}")
            print(f"  Adjusted width: {adjusted_width:.1f}")
            print(f"  Expected: {expected_min_width}")
            
            # Check if the logic is working correctly
            if abs(adjusted_width - expected_min_width) < 0.1:
                print(f"  ✓ PASS")
            else:
                print(f"  ✗ FAIL - Expected {expected_min_width}, got {adjusted_width:.1f}")
        
        # Verify aspect ratio
        if raw_height > 0:
            final_ratio = adjusted_width / raw_height
            min_ratio = min_aspect_ratio
            if final_ratio >= min_ratio - 0.001:  # Small tolerance for floating point
                print(f"  ✓ Aspect ratio {final_ratio:.3f} >= {min_ratio:.3f}")
            else:
                print(f"  ✗ Aspect ratio {final_ratio:.3f} < {min_ratio:.3f}")

def test_drag_simulation():
    """Simulate actual drag operations to test the constraint"""
    print("\n" + "="*50)
    print("Simulating drag operations...")
    
    # Simulate crop selection parameters
    image_width, image_height = 800, 600
    image_x_offset, image_y_offset = 50, 50
    
    # Test drag scenarios
    drag_scenarios = [
        # (start_x, start_y, end_x, end_y, description)
        (100, 100, 200, 300, "Tall narrow rectangle"),
        (100, 100, 150, 200, "Very narrow rectangle"), 
        (100, 100, 300, 150, "Wide short rectangle"),
        (100, 100, 120, 250, "Extremely narrow rectangle"),
    ]
    
    min_aspect_ratio = 9.0 / 20.0
    
    for i, (start_x, start_y, end_x, end_y, description) in enumerate(drag_scenarios):
        print(f"\nDrag scenario {i+1}: {description}")
        print(f"  Original: start({start_x}, {start_y}) → end({end_x}, {end_y})")
        
        # Calculate raw dimensions
        raw_width = abs(end_x - start_x)
        raw_height = abs(end_y - start_y)
        print(f"  Raw dimensions: {raw_width}×{raw_height}")
        
        # Apply constraint
        if raw_height > 0:
            min_width_for_height = raw_height * min_aspect_ratio
            if raw_width < min_width_for_height:
                raw_width = min_width_for_height
                print(f"  Width adjusted to: {raw_width:.1f}")
        
        # Calculate final coordinates
        if end_x >= start_x:
            final_x = start_x + raw_width
        else:
            final_x = start_x - raw_width
            
        final_y = end_y
        
        # Calculate final aspect ratio
        final_width = abs(final_x - start_x)
        final_height = abs(final_y - start_y)
        
        if final_height > 0:
            aspect_ratio = final_width / final_height
            print(f"  Final dimensions: {final_width:.1f}×{final_height:.1f}")
            print(f"  Final aspect ratio: {aspect_ratio:.3f}")
            
            if aspect_ratio >= min_aspect_ratio - 0.001:
                print(f"  ✓ Constraint satisfied ({aspect_ratio:.3f} >= {min_aspect_ratio:.3f})")
            else:
                print(f"  ✗ Constraint violated ({aspect_ratio:.3f} < {min_aspect_ratio:.3f})")

if __name__ == "__main__":
    print("9:20 Aspect Ratio Constraint Test")
    print("=" * 50)
    
    test_aspect_ratio_constraint()
    test_drag_simulation()
    
    print("\n" + "="*50)
    print("Test completed!")
