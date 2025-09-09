#!/usr/bin/env python3
"""
Simple test to verify single-vector support is working
"""

import torch
import os
import numpy as np

def simple_test():
    """Test the single-vector file loading directly"""
    
    # Test files
    test_files = [
        "/home/rich/MyCoding/textual_inversions/textual_inversions/1vLiquidLight.pt",
        "/home/rich/MyCoding/textual_inversions/textual_inversions/wlop_style_learned_embeds.bin"
    ]
    
    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
            
        print(f"\n{'='*60}")
        print(f"TESTING: {os.path.basename(filepath)}")
        print(f"{'='*60}")
        
        try:
            # Load the file directly
            data = torch.load(filepath, map_location='cpu', weights_only=False)
            print(f"✅ File loaded successfully")
            print(f"   Type: {type(data)}")
            print(f"   Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            
            # Find the tensor
            if isinstance(data, dict):
                for key, value in data.items():
                    if torch.is_tensor(value):
                        print(f"   Found tensor at key '{key}': shape {value.shape}")
                        
                        # Test conversion to 2D
                        if len(value.shape) == 1:
                            converted = value.unsqueeze(0)
                            print(f"   ✅ Converted 1D to 2D: {value.shape} → {converted.shape}")
                            
                            # Test basic operations
                            np_array = converted.cpu().detach().numpy()
                            print(f"   ✅ NumPy conversion: {np_array.shape}")
                            print(f"   ✅ Value range: [{np_array.min():.6f}, {np_array.max():.6f}]")
                            
                            # Test a simple operation
                            decimated = np_array.copy()
                            decimated[abs(decimated) < 0.1] = 0
                            print(f"   ✅ Decimation test: {decimated.shape}, zeros: {np.sum(decimated == 0)}")
                            
                        else:
                            print(f"   ✅ Already 2D tensor: {value.shape}")
                            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_test()
