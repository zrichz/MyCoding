#!/usr/bin/env python3
"""
Analysis of flamefractal file based on hex inspection
"""
import torch
import pickle
import zipfile
import os

def analyze_flamefractal_safe():
    filepath = r"textual_inversions/-z-flamefractal.pt"
    
    print(f"=== ANALYSIS OF {os.path.basename(filepath)} ===")
    print(f"File size: {os.path.getsize(filepath)} bytes")
    
    # First, try to examine as a ZIP (since PyTorch files are ZIP archives)
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            print("\n✅ File is a valid ZIP archive (PyTorch format)")
            print("Contents:")
            for info in zf.filelist:
                print(f"  - {info.filename} ({info.file_size} bytes)")
                
            # Try to read the data.pkl file
            if 'archive/data.pkl' in zf.namelist():
                print("\n📄 Reading archive/data.pkl...")
                with zf.open('archive/data.pkl') as pkl_file:
                    # Read first few bytes to inspect
                    first_bytes = pkl_file.read(100)
                    print(f"First 50 characters: {first_bytes[:50]}")
                    
    except zipfile.BadZipFile:
        print("❌ Not a valid ZIP file")
    except Exception as e:
        print(f"❌ Error reading ZIP: {e}")
    
    # Now try torch.load with error handling
    print(f"\n🔧 Attempting torch.load...")
    try:
        # Try with a timeout approach using signal (if available)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("torch.load took too long")
        
        # Set up timeout for Windows (different approach needed)
        try:
            # This won't work on Windows, but let's try anyway
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 second timeout
        except:
            print("Signal timeout not available on Windows, proceeding without timeout...")
        
        data = torch.load(filepath, map_location='cpu')
        
        # Cancel the alarm
        try:
            signal.alarm(0)
        except:
            pass
            
        print("✅ Successfully loaded with torch.load!")
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            
            # Look for any strings that might give us clues
            for key, value in data.items():
                print(f"\nKey '{key}':")
                print(f"  Type: {type(value)}")
                
                if isinstance(value, str):
                    print(f"  String value: {value}")
                elif isinstance(value, dict):
                    print(f"  Dict keys: {list(value.keys())}")
                    for sub_key, sub_value in list(value.items())[:3]:  # First 3 items
                        print(f"    {sub_key}: {type(sub_value)}")
                        if torch.is_tensor(sub_value):
                            print(f"      Tensor shape: {sub_value.shape}")
                elif torch.is_tensor(value):
                    print(f"  Tensor shape: {value.shape}")
                    print(f"  Tensor dtype: {value.dtype}")
                else:
                    str_rep = str(value)[:100]
                    print(f"  Value preview: {str_rep}")
        
        return data
        
    except TimeoutError:
        print("❌ torch.load timed out - file might be problematic")
    except Exception as e:
        print(f"❌ torch.load failed: {e}")
        print(f"Error type: {type(e).__name__}")
    
    return None

if __name__ == "__main__":
    result = analyze_flamefractal_safe()
    if result is None:
        print("\n🔍 CONCLUSION: This file appears to be corrupted or not a standard textual inversion file.")
        print("The filename suggests it's fractal-flame related, which might be a different type of data.")
        print("This would explain why it doesn't have the expected TI structure.")
    else:
        print(f"\n✅ CONCLUSION: File loaded successfully. Analysis complete.")
