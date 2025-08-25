#!/usr/bin/env python3
"""
Test the flexible loading functions on the problematic file
"""
import torch
import os

def find_embedding_tensor(data):
    """
    Flexibly search for embedding tensors in various TI file formats
    Returns: (path_description, tensor) if found, None if not found
    """
    if not isinstance(data, dict):
        return None
    
    # Common patterns for textual inversion files
    search_patterns = [
        # Standard Automatic1111 format
        (['string_to_param', '*'], "string_to_param['*']"),
        (['string_to_param', 'embedding'], "string_to_param['embedding']"),
        
        # Alternative key names
        (['emb_params', '*'], "emb_params['*']"),
        (['embeddings', '*'], "embeddings['*']"),
        (['embedding', '*'], "embedding['*']"),
        
        # Direct embedding keys
        (['*'], "root level '*'"),
        (['embedding'], "root level 'embedding'"),
        (['tensor'], "root level 'tensor'"),
        (['vectors'], "root level 'vectors'"),
        (['weights'], "root level 'weights'"),
        
        # Token-based patterns
        (['string_to_param'], "string_to_param (checking all keys)"),
        (['embeddings'], "embeddings (checking all keys)"),
        (['emb_params'], "emb_params (checking all keys)"),
    ]
    
    # Try each pattern
    for key_path, description in search_patterns:
        try:
            current = data
            
            # Navigate through the key path
            for key in key_path[:-1]:
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    break
            else:
                # We successfully navigated to the parent, now check the final key
                final_key = key_path[-1]
                
                if final_key in current:
                    tensor_candidate = current[final_key]
                    if torch.is_tensor(tensor_candidate) and len(tensor_candidate.shape) == 2:
                        return (description, tensor_candidate)
                elif len(key_path) == 1 and final_key in ['string_to_param', 'embeddings', 'emb_params']:
                    # For containers, check all their contents
                    container = current[final_key]
                    if isinstance(container, dict):
                        for sub_key, sub_value in container.items():
                            if torch.is_tensor(sub_value) and len(sub_value.shape) == 2:
                                return (f"{description}['{sub_key}']", sub_value)
        except:
            continue
    
    return None

def test_flamefractal():
    filepath = r"textual_inversions/-z-flamefractal.pt"
    
    try:
        print(f"Testing file: {filepath}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        
        # Load the file
        data = torch.load(filepath, map_location='cpu')
        print(f"Loaded successfully. Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            
            # Try our flexible finder
            result = find_embedding_tensor(data)
            if result:
                path, tensor = result
                print(f"✅ Found embedding at: {path}")
                print(f"   Shape: {tensor.shape}")
                print(f"   Data type: {tensor.dtype}")
                print(f"   Min/Max: {tensor.min():.6f} / {tensor.max():.6f}")
            else:
                print("❌ No embedding tensor found with flexible search")
                
                # Manual examination
                print("\nManual examination:")
                for key, value in data.items():
                    print(f"Key '{key}': {type(value)}")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            print(f"  Sub-key '{sub_key}': {type(sub_value)}")
                            if torch.is_tensor(sub_value):
                                print(f"    Tensor shape: {sub_value.shape}")
                    elif torch.is_tensor(value):
                        print(f"  Direct tensor shape: {value.shape}")
                    else:
                        print(f"  Value: {str(value)[:50]}")
        
        return data
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_flamefractal()
