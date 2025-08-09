#!/usr/bin/env python3
"""
Test the end crop selection aspect ratio constraint
"""

def test_end_crop_aspect_ratio():
    """Test the aspect ratio constraint logic used in end_crop_selection"""
    print("Testing end crop selection aspect ratio constraint...")
    
    min_aspect_ratio = 9.0 / 20.0  # 0.45
    
    # Test scenarios: (start_x, start_y, end_x, end_y, description)
    test_scenarios = [
        (100, 100, 120, 200, "Very narrow selection - should be widened"),
        (100, 100, 150, 250, "Slightly narrow - should be widened"),
        (100, 100, 200, 300, "Good aspect ratio - should remain unchanged"),
        (100, 100, 300, 200, "Wide selection - should remain unchanged"),
        (200, 200, 150, 300, "Dragging left and down - narrow"),
        (200, 200, 250, 150, "Dragging right and up - wide"),
    ]
    
    for i, (start_x, start_y, end_x, end_y, description) in enumerate(test_scenarios):
        print(f"\nTest {i+1}: {description}")
        print(f"  Input: start({start_x}, {start_y}) → end({end_x}, {end_y})")
        
        # Apply the same logic as in end_crop_selection
        raw_width = abs(end_x - start_x)
        raw_height = abs(end_y - start_y)
        
        print(f"  Raw dimensions: {raw_width}×{raw_height}")
        
        # Apply aspect ratio constraint
        if raw_height > 0:
            min_width_for_height = raw_height * min_aspect_ratio
            if raw_width < min_width_for_height:
                print(f"  Adjusting width from {raw_width} to {min_width_for_height:.1f}")
                raw_width = min_width_for_height
        
        # Calculate final coordinates
        if end_x >= start_x:  # Dragging right
            final_x = start_x + raw_width
        else:  # Dragging left
            final_x = start_x - raw_width
            
        if end_y >= start_y:  # Dragging down
            final_y = start_y + raw_height
        else:  # Dragging up
            final_y = start_y - raw_height
        
        # Calculate final dimensions and aspect ratio
        final_width = abs(final_x - start_x)
        final_height = abs(final_y - start_y)
        
        if final_height > 0:
            final_aspect_ratio = final_width / final_height
            print(f"  Final dimensions: {final_width:.1f}×{final_height:.1f}")
            print(f"  Final aspect ratio: {final_aspect_ratio:.3f}")
            
            if final_aspect_ratio >= min_aspect_ratio - 0.001:
                print(f"  ✓ Constraint satisfied ({final_aspect_ratio:.3f} >= {min_aspect_ratio:.3f})")
            else:
                print(f"  ✗ Constraint violated ({final_aspect_ratio:.3f} < {min_aspect_ratio:.3f})")
        else:
            print(f"  Warning: Zero height")

if __name__ == "__main__":
    test_end_crop_aspect_ratio()
