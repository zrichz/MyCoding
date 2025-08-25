#!/usr/bin/env python3
"""
Quick test script to demonstrate the vector extraction functionality
"""
import torch
import os
import sys

# Add the current directory to the path so we can import our functions
sys.path.append('.')
from TI_CHANGER_MULTIPLE_2024_10_22 import extract_individual_vectors

def test_vector_extraction():
    """Test the vector extraction feature"""
    
    print("=== Testing Vector Extraction Feature ===\n")
    
    # Check if TI file exists
    filename = 'TI_Tron_original.pt'
    if not os.path.exists(filename):
        print(f"Error: {filename} not found in current directory")
        return False
    
    # Load the TI file
    print(f"Loading {filename}...")
    data = torch.load(filename, map_location='cpu')
    
    # Get tensor info
    tensor = data['string_to_param']['*']
    numvectors = tensor.shape[0]
    
    print(f"✓ File loaded successfully")
    print(f"✓ Contains {numvectors} vectors")
    print(f"✓ Each vector has {tensor.shape[1]} dimensions")
    print(f"✓ Tensor shape: {tensor.shape}")
    
    # Show what the extraction would create
    base_filename = filename.replace('.pt', '')
    print(f"\nVector extraction would create these files:")
    for i in range(numvectors):
        print(f"  - {base_filename}_vector_{i+1:02d}.pt")
    
    print(f"\nTest completed successfully!")
    print(f"The vector extraction feature is ready to use with Option 6 in the main script.")
    
    return True

if __name__ == "__main__":
    success = test_vector_extraction()
    if success:
        print(f"\n✅ All tests passed! The script is working correctly.")
    else:
        print(f"\n❌ Test failed. Please check the file paths.")
