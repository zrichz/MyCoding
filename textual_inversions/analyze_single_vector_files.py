#!/usr/bin/env python3
"""
Quick analysis of single-vector .pt files to understand their structure
"""

import torch
import os

def analyze_pt_file_detailed(filepath):
    """Analyze a .pt file structure in detail"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        data = torch.load(filepath, map_location='cpu')
        
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print("\nDictionary structure:")
            for key, value in data.items():
                print(f"  '{key}': {type(value)}")
                if torch.is_tensor(value):
                    print(f"    Shape: {value.shape}")
                    print(f"    Dtype: {value.dtype}")
                    print(f"    Device: {value.device}")
                    if len(value.shape) <= 2 and value.numel() <= 10:
                        print(f"    Values: {value}")
                elif isinstance(value, dict):
                    print(f"    Sub-dictionary with {len(value)} keys:")
                    for subkey, subvalue in value.items():
                        print(f"      '{subkey}': {type(subvalue)}")
                        if torch.is_tensor(subvalue):
                            print(f"        Shape: {subvalue.shape}")
                            print(f"        Dtype: {subvalue.dtype}")
                            if len(subvalue.shape) <= 2 and subvalue.numel() <= 10:
                                print(f"        Values: {subvalue}")
                        
        elif torch.is_tensor(data):
            print(f"\nDirect tensor:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Device: {data.device}")
            print(f"  Min value: {data.min()}")
            print(f"  Max value: {data.max()}")
            print(f"  Mean: {data.mean()}")
            print(f"  Std: {data.std()}")
            
            if len(data.shape) == 1:
                print(f"  This is a 1D tensor - single vector!")
                print(f"  Vector length: {data.shape[0]}")
            elif len(data.shape) == 2:
                print(f"  This is a 2D tensor")
                print(f"  Number of vectors: {data.shape[0]}")
                print(f"  Vector dimensions: {data.shape[1]}")
                
        else:
            print(f"\nUnexpected data type: {type(data)}")
            print(f"Data: {data}")
            
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    # Analyze the known single-vector files
    files_to_analyze = [
        "/home/rich/MyCoding/textual_inversions/textual_inversions/1vLiquidLight.pt",
        "/home/rich/MyCoding/textual_inversions/textual_inversions/wlop_style_learned_embeds.bin"
    ]
    
    for filepath in files_to_analyze:
        if os.path.exists(filepath):
            analyze_pt_file_detailed(filepath)
        else:
            print(f"\nFILE NOT FOUND: {filepath}")

if __name__ == "__main__":
    main()
