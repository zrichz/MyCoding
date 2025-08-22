#!/usr/bin/env python3
"""
Test script to demonstrate the vector extraction functionality
"""
import torch
import numpy as np
import os

def create_test_ti_file():
    """Create a test TI file with multiple vectors for demonstration"""
    
    # Create a sample TI structure with 4 vectors of 768 dimensions each
    num_vectors = 4
    vector_size = 768
    
    # Create random test data that resembles actual TI embeddings
    np.random.seed(42)  # For reproducible results
    test_vectors = np.random.randn(num_vectors, vector_size) * 0.1  # Small random values
    
    # Convert to PyTorch tensor
    test_tensor = torch.tensor(test_vectors, dtype=torch.float32, requires_grad=True)
    
    # Create the TI data structure
    test_data = {
        'string_to_param': {'*': test_tensor},
        'string_to_token': {'*': 49408},  # Example token ID
        'name': 'test_embedding',
        'step': 1000
    }
    
    # Save the test file
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, "TEST_4vectors.pt")
    torch.save(test_data, filepath)
    
    print(f"Created test TI file: {filepath}")
    print(f"Contains {num_vectors} vectors of {vector_size} dimensions each")
    
    return filepath

if __name__ == "__main__":
    test_file = create_test_ti_file()
    print(f"\nTest file created successfully!")
    print(f"You can now test the vector extraction feature using this file.")
    print(f"In the main script, change the filename variable to 'TEST_4vectors.pt' to test.")
