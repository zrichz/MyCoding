#!/usr/bin/env python3
"""
Test script to verify the max/min averaging algorithm works correctly
"""

import numpy as np

def test_max_min_averaging():
    """Test the max/min averaging logic with a simple example"""
    
    # Create test data: 3 vectors with 5 dimensions each
    test_vectors = np.array([
        [1.0, -2.0, 3.0, -1.0, 0.5],   # Vector 1
        [2.0, -1.0, 1.0, -3.0, -0.5],  # Vector 2 
        [0.5, -3.0, 2.0, -0.5, 1.0]    # Vector 3
    ])
    
    print("Test Data:")
    print("Vector 1:", test_vectors[0])
    print("Vector 2:", test_vectors[1])
    print("Vector 3:", test_vectors[2])
    
    num_dimensions = test_vectors.shape[1]
    result_vector = np.zeros(num_dimensions)
    
    print(f"\nProcessing {num_dimensions} dimensions:")
    
    for dim_idx in range(num_dimensions):
        # Extract all values at this dimension across all vectors
        values_at_dimension = test_vectors[:, dim_idx]
        
        # Calculate average for this dimension
        avg_value = np.mean(values_at_dimension)
        
        print(f"\nDimension {dim_idx}:")
        print(f"  Values: {values_at_dimension}")
        print(f"  Average: {avg_value:.3f}")
        
        if avg_value > 0:
            # Use maximum value if average is positive
            result_vector[dim_idx] = np.max(values_at_dimension)
            print(f"  Average > 0, using MAX: {result_vector[dim_idx]}")
        elif avg_value < 0:
            # Use minimum value if average is negative
            result_vector[dim_idx] = np.min(values_at_dimension)
            print(f"  Average < 0, using MIN: {result_vector[dim_idx]}")
        else:
            # Handle exact zero case
            result_vector[dim_idx] = 0.0
            print(f"  Average = 0, using ZERO: {result_vector[dim_idx]}")
    
    print(f"\nFinal Result Vector: {result_vector}")
    
    # Verify expected results manually:
    # Dim 0: [1.0, 2.0, 0.5] → avg = 1.167 (positive) → max = 2.0 ✓
    # Dim 1: [-2.0, -1.0, -3.0] → avg = -2.0 (negative) → min = -3.0 ✓ 
    # Dim 2: [3.0, 1.0, 2.0] → avg = 2.0 (positive) → max = 3.0 ✓
    # Dim 3: [-1.0, -3.0, -0.5] → avg = -1.5 (negative) → min = -3.0 ✓
    # Dim 4: [0.5, -0.5, 1.0] → avg = 0.333 (positive) → max = 1.0 ✓
    
    expected = np.array([2.0, -3.0, 3.0, -3.0, 1.0])
    print(f"Expected Result:   {expected}")
    
    if np.allclose(result_vector, expected):
        print("✅ Test PASSED - Algorithm works correctly!")
    else:
        print("❌ Test FAILED - Algorithm has issues!")
        print(f"Difference: {result_vector - expected}")
    
    return result_vector

if __name__ == "__main__":
    test_max_min_averaging()
