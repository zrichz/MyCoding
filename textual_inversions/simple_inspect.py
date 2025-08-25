#!/usr/bin/env python3
"""
Simple inspection of flamefractal file
"""
import torch

try:
    print("Loading file...")
    data = torch.load(r"textual_inversions/-z-flamefractal.pt", map_location='cpu')
    print(f"SUCCESS: Loaded file")
    print(f"Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Dictionary keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, dict):
                print(f"    Sub-keys: {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {type(sub_value)}")
                    if torch.is_tensor(sub_value):
                        print(f"        Tensor shape: {sub_value.shape}")
            elif torch.is_tensor(value):
                print(f"    Tensor shape: {value.shape}")
    elif torch.is_tensor(data):
        print(f"Raw tensor shape: {data.shape}")
    else:
        print(f"Other type: {str(data)[:100]}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
