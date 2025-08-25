#!/usr/bin/env python3
"""
Quick test of the flexible TI loading functionality
"""

import torch
import os

def analyze_pt_file(filepath):
    """
    Analyze a .pt file to determine if it contains textual inversion data
    """
    try:
        data = torch.load(filepath, map_location='cpu')
        print(f"\n=== Analyzing {os.path.basename(filepath)} ===")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            
            # Check for common TI patterns
            patterns = [
                ['string_to_param', '*'],
                ['emb_params', '*'],  
                ['embeddings', '*'],
                ['string_to_param'],
                ['embeddings'],
                ['*']
            ]
            
            for pattern in patterns:
                current = data
                path_found = True
                for key in pattern:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                        print(f"  Found key: {key}")
                    else:
                        path_found = False
                        break
                
                if path_found and torch.is_tensor(current):
                    print(f"  ✅ Found tensor at path: {' -> '.join(pattern)}")
                    print(f"  📊 Tensor shape: {current.shape}")
                    print(f"  🔢 Tensor dtype: {current.dtype}")
                    return True
                    
        elif torch.is_tensor(data):
            print(f"  📊 Raw tensor shape: {data.shape}")
            print(f"  🔢 Tensor dtype: {data.dtype}")
            if len(data.shape) == 2:
                print("  ✅ Looks like a raw embedding tensor")
                return True
        
        print("  ❌ No embedding tensor patterns found")
        return False
        
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return False

def test_files():
    """Test all .pt files in the textual_inversions directory"""
    pt_files = [
        "textual_inversions/8vSw-0600.pt",
        "textual_inversions/8vSw-1401.pt", 
        "textual_inversions/TI_Tron_original.pt",
        "textual_inversions/-z-flamefractal.pt",
        "textual_inversions/-z-liquidlight.pt"
    ]
    
    for filepath in pt_files:
        if os.path.exists(filepath):
            analyze_pt_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    test_files()
