#!/usr/bin/env python3
"""
Quick test script to demonstrate the enhanced file validation
"""
import torch
import os
import sys

# Add the current directory to the path so we can import our functions
sys.path.append('.')
from TI_CHANGER_MULTIPLE_2024_10_22 import analyze_pt_file

def test_file_validation():
    """Test the file validation on various .pt files"""
    
    print("=" * 60)
    print("TI Changer - File Validation Test")
    print("=" * 60)
    
    # Look for .pt files in current directory
    pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    
    if not pt_files:
        print("No .pt files found in current directory")
        return
    
    print(f"Found {len(pt_files)} .pt files to analyze:")
    for i, filename in enumerate(pt_files, 1):
        print(f"{i}. {filename}")
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS:")
    print("="*60)
    
    valid_ti_files = []
    
    for filename in pt_files:
        print(f"\n{'='*20} {filename} {'='*20}")
        try:
            if analyze_pt_file(filename):
                valid_ti_files.append(filename)
                print("✅ COMPATIBLE: This file can be used with TI Changer")
            else:
                print("❌ INCOMPATIBLE: This file cannot be used with TI Changer")
        except Exception as e:
            print(f"❌ ERROR: Failed to analyze - {e}")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Total files analyzed: {len(pt_files)}")
    print(f"Compatible TI files: {len(valid_ti_files)}")
    print(f"Incompatible files: {len(pt_files) - len(valid_ti_files)}")
    
    if valid_ti_files:
        print(f"\n✅ COMPATIBLE FILES (can be used with TI Changer):")
        for filename in valid_ti_files:
            print(f"   - {filename}")
    
    incompatible = [f for f in pt_files if f not in valid_ti_files]
    if incompatible:
        print(f"\n❌ INCOMPATIBLE FILES (cannot be used with TI Changer):")
        for filename in incompatible:
            print(f"   - {filename}")
    
    print(f"\n{'='*60}")
    print("The enhanced TI Changer will now properly detect and reject")
    print("incompatible files, providing clear error messages.")
    print("="*60)

if __name__ == "__main__":
    test_file_validation()
