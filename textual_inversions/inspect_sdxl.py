import safetensors.torch as st
import torch

# Load both SDXL TI files
file1 = r'c:\MyCoding\textual_inversions\SDXL_TI_90sPhoto.safetensors'
file2 = r'c:\MyCoding\textual_inversions\SDXL_TI_GoPro.safetensors'

for filepath in [file1, file2]:
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print('='*60)
    
    data = st.load_file(filepath)
    
    print(f"Keys: {list(data.keys())}")
    print(f"\nShapes and dtypes:")
    for k, v in data.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Show sample values
    for k, v in data.items():
        print(f"\n{k} sample values:")
        print(f"  Min: {v.min().item():.6f}")
        print(f"  Max: {v.max().item():.6f}")
        print(f"  Mean: {v.mean().item():.6f}")
        if len(v.shape) == 2:
            print(f"  First row sample (first 10 values): {v[0, :10]}")
