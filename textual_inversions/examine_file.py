#!/usr/bin/env python3
"""
Direct test of specific TI file structure
"""

import torch
import os

def examine_file(filepath):
    """Examine the structure of a specific TI file"""
    try:
        print(f"\n=== Examining {filepath} ===")
        data = torch.load(filepath, map_location='cpu')
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            
            # Examine each key
            for key, value in data.items():
                print(f"\nKey: '{key}'")
                print(f"  Type: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"  Sub-keys: {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        print(f"    '{sub_key}': {type(sub_value)}")
                        if torch.is_tensor(sub_value):
                            print(f"      Tensor shape: {sub_value.shape}")
                            print(f"      Tensor dtype: {sub_value.dtype}")
                elif torch.is_tensor(value):
                    print(f"  Tensor shape: {value.shape}")
                    print(f"  Tensor dtype: {value.dtype}")
                else:
                    print(f"  Value: {value}")
                    
        elif torch.is_tensor(data):
            print(f"Raw tensor shape: {data.shape}")
            print(f"Raw tensor dtype: {data.dtype}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Test one of the user's files
    test_file = "textual_inversions/8vSw-0600.pt"
    if os.path.exists(test_file):
        examine_file(test_file)
    else:
        print(f"File not found: {test_file}")
        print("Available files:")
        if os.path.exists("textual_inversions"):
            for f in os.listdir("textual_inversions"):
                if f.endswith(".pt"):
                    print(f"  {f}")
