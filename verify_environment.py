#!/usr/bin/env python3
"""
Verification script to check virtual environment setup
"""

import sys
import os
import subprocess

def check_python_environment():
    """Check current Python environment details"""
    print("=" * 60)
    print("PYTHON ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'None')}")
    
    # Check if we're in the correct virtual environment
    expected_path = "C:\\MyPythonCoding\\MyCoding\\venv_mycoding\\Scripts\\python.exe"
    if sys.executable.lower() == expected_path.lower():
        print("✅ Using correct virtual environment (venv_mycoding)")
    else:
        print("❌ NOT using venv_mycoding!")
        print(f"Expected: {expected_path}")
        print(f"Actual: {sys.executable}")
    
    print("\n" + "=" * 60)
    print("INSTALLED PACKAGES")
    print("=" * 60)
    
    # Check key packages
    key_packages = ['torch', 'matplotlib', 'numpy', 'scikit-learn']
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - NOT AVAILABLE")
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    venv_vars = ['VIRTUAL_ENV', 'VIRTUAL_ENV_PROMPT']
    for var in venv_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    print("\n" + "=" * 60)
    print("PATH CHECK")
    print("=" * 60)
    
    path_entries = os.environ.get('PATH', '').split(';')
    mycoding_paths = [p for p in path_entries if 'MyPythonCoding' in p or 'venv' in p]
    
    print("Python/Virtual environment paths in PATH:")
    for path in mycoding_paths:
        if path.strip():
            if 'venv_mycoding' in path:
                print(f"✅ {path}")
            elif 'venv_myDL1' in path or 'myDLvenv1' in path:
                print(f"❌ {path} (SPURIOUS - should be removed)")
            else:
                print(f"ℹ️  {path}")

if __name__ == "__main__":
    check_python_environment()
