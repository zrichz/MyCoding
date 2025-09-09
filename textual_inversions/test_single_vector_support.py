#!/usr/bin/env python3
"""
Test script for single-vector TI loading functionality
"""

import torch
import os
import sys

print("Script starting...")
print(f"Python path: {sys.path}")

# Add the current directory to Python path for imports
sys.path.append('/home/rich/MyCoding/textual_inversions')

print("Attempting imports...")

try:
    # Import the functions we want to test
    from TI_CHANGER_MULTIPLE_2024_10_22 import analyze_pt_file, load_ti_file_flexible
    print("✅ Imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Other error during import: {e}")
    sys.exit(1)

def test_single_vector_loading():
    """Test the enhanced loading functions with single-vector files"""
    
    # Test files
    test_files = [
        "/home/rich/MyCoding/textual_inversions/textual_inversions/1vLiquidLight.pt",
        "/home/rich/MyCoding/textual_inversions/textual_inversions/wlop_style_learned_embeds.bin"
    ]
    
    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
            
        print(f"\n{'='*80}")
        print(f"TESTING: {os.path.basename(filepath)}")
        print(f"{'='*80}")
        
        # Test analysis function
        print("\n1. Testing analyze_pt_file():")
        print("-" * 40)
        analyze_result = analyze_pt_file(filepath)
        print(f"Analysis result: {'✅ Success' if analyze_result else '❌ Failed'}")
        
        # Test flexible loading function
        print("\n2. Testing load_ti_file_flexible():")
        print("-" * 40)
        load_result = load_ti_file_flexible(filepath)
        
        if load_result:
            data, tensor, tensor_path, is_single_vector = load_result
            print(f"✅ Loading successful!")
            print(f"   Data type: {type(data)}")
            print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            print(f"   Tensor shape: {tensor.shape}")
            print(f"   Tensor path: {tensor_path}")
            print(f"   Is single vector: {is_single_vector}")
            print(f"   Tensor dtype: {tensor.dtype}")
            print(f"   Tensor device: {tensor.device}")
            
            # Verify the structure
            if 'string_to_param' in data and '*' in data['string_to_param']:
                print(f"   ✅ Standard structure created successfully")
                stored_tensor = data['string_to_param']['*']
                print(f"   Stored tensor shape: {stored_tensor.shape}")
                print(f"   Tensor match: {torch.equal(tensor, stored_tensor)}")
            else:
                print(f"   ❌ Standard structure missing")
                
        else:
            print(f"❌ Loading failed!")

def test_operations_compatibility():
    """Test that operations work with single-vector files"""
    
    print(f"\n{'='*80}")
    print("TESTING OPERATIONS COMPATIBILITY")
    print(f"{'='*80}")
    
    # Load a single-vector file
    filepath = "/home/rich/MyCoding/textual_inversions/textual_inversions/1vLiquidLight.pt"
    if not os.path.exists(filepath):
        print(f"❌ Test file not found: {filepath}")
        return
        
    load_result = load_ti_file_flexible(filepath)
    if not load_result:
        print(f"❌ Failed to load test file")
        return
        
    data, tensor, tensor_path, is_single_vector = load_result
    np_array = tensor.cpu().detach().numpy()
    
    print(f"Test file loaded: {os.path.basename(filepath)}")
    print(f"Single vector: {is_single_vector}")
    print(f"Array shape: {np_array.shape}")
    
    # Test basic operations that should work
    print("\n🧪 Testing basic operations:")
    
    # Test element-wise operations (should work)
    try:
        # Threshold decimation test
        decimated = np_array.copy()
        decimated[abs(decimated) < 0.1] = 0
        print(f"   ✅ Decimation: {decimated.shape}")
        
        # Scalar division test
        divided = np_array / 2.0
        print(f"   ✅ Scalar division: {divided.shape}")
        
        # L2 normalization test
        import numpy as np
        norm_factor = np.linalg.norm(np_array, axis=1, keepdims=True)
        normalized = np_array / (norm_factor + 1e-8)
        print(f"   ✅ L2 normalization: {normalized.shape}")
        
        # Rolling/shifting test
        rolled = np.roll(np_array, 10, axis=1)
        print(f"   ✅ Rolling/shifting: {rolled.shape}")
        
        print(f"\n✅ All single-vector compatible operations work!")
        
    except Exception as e:
        print(f"❌ Error in operations: {e}")

if __name__ == "__main__":
    test_single_vector_loading()
    test_operations_compatibility()
