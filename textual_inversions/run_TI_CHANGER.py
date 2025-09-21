#!/usr/bin/env python3
"""
Cross-platform launcher for TI Changer Multiple
Works on Windows, Linux, and macOS
"""

import os
import sys
import subprocess
import platform

def find_python_executable():
    """Find the appropriate Python executable for the current platform"""
    system = platform.system().lower()
    
    if system == "windows":
        # Try to find the specific virtual environment first
        venv_paths = [
            r"C:\MyPythonCoding\MyCoding\image_processors\.venv\Scripts\python.exe",
            r".venv\Scripts\python.exe",
            r"venv\Scripts\python.exe"
        ]
        
        for path in venv_paths:
            if os.path.exists(path):
                return path
        
        # Fall back to system Python
        for cmd in ["python", "python3", "py"]:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
                
    else:  # Linux, macOS, etc.
        # Try to find virtual environment
        venv_paths = [
            "../image_processors/.venv/bin/python",
            ".venv/bin/python",
            "venv/bin/python"
        ]
        
        for path in venv_paths:
            if os.path.exists(path):
                return path
        
        # Fall back to system Python
        for cmd in ["python3", "python"]:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    
    return None

def check_dependencies(python_cmd):
    """Check if required dependencies are installed"""
    try:
        result = subprocess.run([
            python_cmd, "-c", 
            "import torch, matplotlib, numpy, tkinter; print('All dependencies available')"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print("Missing dependencies. Error:", result.stderr)
            return False
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False

def install_dependencies(python_cmd):
    """Install required dependencies"""
    packages = ["torch", "matplotlib", "numpy"]
    
    print("Installing required packages...")
    try:
        subprocess.run([python_cmd, "-m", "pip", "install"] + packages, check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def main():
    print("=" * 50)
    print("TI Changer Multiple - Cross-Platform Launcher")
    print("=" * 50)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print("")
    
    # Find Python executable
    python_cmd = find_python_executable()
    if not python_cmd:
        print("Error: Could not find Python installation!")
        print("Please install Python 3.7 or later.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print(f"Found Python: {python_cmd}")
    
    # Get Python version
    try:
        result = subprocess.run([python_cmd, "--version"], capture_output=True, text=True)
        print(f"Version: {result.stdout.strip()}")
    except:
        print("Could not determine Python version")
    
    print("")
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies(python_cmd):
        print("Required packages missing.")
        response = input("Would you like to install them automatically? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not install_dependencies(python_cmd):
                print("Failed to install dependencies. Please install manually:")
                print("pip install torch matplotlib numpy")
                input("Press Enter to exit...")
                sys.exit(1)
        else:
            print("Please install required packages manually:")
            print("pip install torch matplotlib numpy")
            input("Press Enter to exit...")
            sys.exit(1)
    
    print("All dependencies satisfied!")
    print("")
    
    # Check if the main script exists
    script_name = "TI_CHANGER_MULTIPLE_2024_10_22.py"
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print(f"Found script: {script_name}")
    print("")
    print("Starting TI Changer Multiple...")
    print("-" * 40)
    print("")
    
    # Run the main script
    try:
        subprocess.run([python_cmd, script_name], check=True)
        print("")
        print("TI Changer completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running TI Changer: {e}")
        print("Please check the error messages above.")
    except KeyboardInterrupt:
        print("")
        print("TI Changer interrupted by user.")
    
    print("")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
