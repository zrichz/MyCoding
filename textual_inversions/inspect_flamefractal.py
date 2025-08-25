#!/usr/bin/env python3
"""
Detailed inspection of the problematic -z-flamefractal.pt file
"""

import torch
import os

def inspect_file_detailed(filepath):
    """Perform a comprehensive inspection of the .pt file structure"""
    try:
        print(f"=== DETAILED INSPECTION: {os.path.basename(filepath)} ===")
        print(f"File path: {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")
        
        if not os.path.exists(filepath):
            print("❌ File not found!")
            return
        
        # Get file size
        file_size = os.path.getsize(filepath)
        print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Load the file
        print("\n--- LOADING FILE ---")
        data = torch.load(filepath, map_location='cpu')
        print(f"✅ File loaded successfully")
        print(f"Root data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"\n--- DICTIONARY STRUCTURE ---")
            print(f"Number of top-level keys: {len(data)}")
            print(f"Top-level keys: {list(data.keys())}")
            
            # Examine each key in detail
            for key, value in data.items():
                print(f"\n🔍 KEY: '{key}'")
                print(f"   Type: {type(value)}")
                
                if isinstance(value, dict):
                    print(f"   Sub-dictionary with {len(value)} keys: {list(value.keys())}")
                    
                    # Examine sub-keys
                    for sub_key, sub_value in value.items():
                        print(f"   └── '{sub_key}': {type(sub_value)}")
                        
                        if torch.is_tensor(sub_value):
                            print(f"       📊 Tensor shape: {sub_value.shape}")
                            print(f"       📊 Tensor dtype: {sub_value.dtype}")
                            print(f"       📊 Tensor device: {sub_value.device}")
                            if len(sub_value.shape) <= 2:
                                print(f"       📊 Min/Max: {sub_value.min().item():.6f} / {sub_value.max().item():.6f}")
                                print(f"       📊 Mean/Std: {sub_value.mean().item():.6f} / {sub_value.std().item():.6f}")
                        elif isinstance(sub_value, (int, float)):
                            print(f"       📊 Value: {sub_value}")
                        elif isinstance(sub_value, str):
                            print(f"       📊 String: '{sub_value}'")
                        elif isinstance(sub_value, dict):
                            print(f"       📊 Nested dict with keys: {list(sub_value.keys())}")
                        else:
                            print(f"       📊 Other type: {type(sub_value)}")
                            
                elif torch.is_tensor(value):
                    print(f"   📊 Direct tensor shape: {value.shape}")
                    print(f"   📊 Direct tensor dtype: {value.dtype}")
                    print(f"   📊 Direct tensor device: {value.device}")
                    if len(value.shape) <= 2:
                        print(f"   📊 Min/Max: {value.min().item():.6f} / {value.max().item():.6f}")
                        print(f"   📊 Mean/Std: {value.mean().item():.6f} / {value.std().item():.6f}")
                elif isinstance(value, (int, float)):
                    print(f"   📊 Value: {value}")
                elif isinstance(value, str):
                    print(f"   📊 String: '{value}'")
                else:
                    print(f"   📊 Other type: {type(value)}")
                    print(f"   📊 String representation: {str(value)[:100]}...")
        
        elif torch.is_tensor(data):
            print(f"\n--- RAW TENSOR ---")
            print(f"📊 Tensor shape: {data.shape}")
            print(f"📊 Tensor dtype: {data.dtype}")
            print(f"📊 Tensor device: {data.device}")
            if len(data.shape) <= 2:
                print(f"📊 Min/Max: {data.min().item():.6f} / {data.max().item():.6f}")
                print(f"📊 Mean/Std: {data.mean().item():.6f} / {data.std().item():.6f}")
        
        else:
            print(f"\n--- UNKNOWN TYPE ---")
            print(f"Type: {type(data)}")
            print(f"String representation: {str(data)[:200]}...")
        
        # Try to detect TI patterns
        print(f"\n--- TI PATTERN DETECTION ---")
        patterns_found = []
        
        if isinstance(data, dict):
            # Check standard patterns
            if 'string_to_param' in data:
                patterns_found.append("✅ Found 'string_to_param' key")
                if isinstance(data['string_to_param'], dict):
                    if '*' in data['string_to_param']:
                        tensor = data['string_to_param']['*']
                        if torch.is_tensor(tensor):
                            patterns_found.append(f"✅ Found tensor at 'string_to_param']['*'] with shape {tensor.shape}")
                        else:
                            patterns_found.append(f"❌ 'string_to_param']['*'] is not a tensor: {type(tensor)}")
                    else:
                        patterns_found.append(f"❌ No '*' key in 'string_to_param', keys: {list(data['string_to_param'].keys())}")
                else:
                    patterns_found.append(f"❌ 'string_to_param' is not a dict: {type(data['string_to_param'])}")
            else:
                patterns_found.append("❌ No 'string_to_param' key found")
            
            if 'string_to_token' in data:
                patterns_found.append("✅ Found 'string_to_token' key")
            else:
                patterns_found.append("❌ No 'string_to_token' key found")
                
            # Check alternative patterns
            for alt_key in ['emb_params', 'embeddings', 'embedding']:
                if alt_key in data:
                    patterns_found.append(f"✅ Found alternative key '{alt_key}'")
                    
            # Look for any 2D tensors
            tensors_found = []
            def find_tensors(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        find_tensors(value, current_path)
                elif torch.is_tensor(obj) and len(obj.shape) == 2:
                    tensors_found.append((path, obj.shape))
            
            find_tensors(data)
            
            if tensors_found:
                patterns_found.append(f"✅ Found {len(tensors_found)} 2D tensors:")
                for path, shape in tensors_found:
                    patterns_found.append(f"    - {path}: {shape}")
            else:
                patterns_found.append("❌ No 2D tensors found")
        
        for pattern in patterns_found:
            print(pattern)
        
        # Final assessment
        print(f"\n--- COMPATIBILITY ASSESSMENT ---")
        is_compatible = False
        issues = []
        
        if isinstance(data, dict):
            if 'string_to_param' in data and isinstance(data['string_to_param'], dict):
                if '*' in data['string_to_param'] and torch.is_tensor(data['string_to_param']['*']):
                    tensor = data['string_to_param']['*']
                    if len(tensor.shape) == 2:
                        is_compatible = True
                        print("✅ COMPATIBLE: Standard TI format detected")
                    else:
                        issues.append(f"Tensor shape is {tensor.shape}, expected 2D")
                else:
                    issues.append("Missing or invalid '*' tensor in 'string_to_param'")
            else:
                issues.append("Missing or invalid 'string_to_param' structure")
                
                # Check if flexible loading could handle it
                if tensors_found:
                    print("⚠️  POTENTIALLY COMPATIBLE: Could be handled by flexible loading")
                    is_compatible = True
        elif torch.is_tensor(data) and len(data.shape) == 2:
            is_compatible = True
            print("✅ COMPATIBLE: Raw tensor format detected")
        
        if not is_compatible:
            print("❌ INCOMPATIBLE: This file cannot be processed as a textual inversion")
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  - {issue}")
        
        return data
        
    except Exception as e:
        print(f"❌ ERROR during inspection: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    filepath = r"C:\MyPythonCoding\MyCoding\textual_inversions\textual_inversions\-z-flamefractal.pt"
    inspect_file_detailed(filepath)
