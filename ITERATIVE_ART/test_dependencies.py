"""
Test script to check dependencies and run a simplified version
"""
import sys
print("Python version:", sys.version)

try:
    import tkinter as tk
    print("✓ tkinter available")
except ImportError as e:
    print("✗ tkinter error:", e)

try:
    from PIL import Image, ImageDraw, ImageTk
    print("✓ PIL/Pillow available")
except ImportError as e:
    print("✗ PIL/Pillow error:", e)

try:
    import numpy as np
    print("✓ numpy available")
except ImportError as e:
    print("✗ numpy error:", e)

print("\nStarting basic test...")

# Basic tkinter test
if 'tkinter' in sys.modules:
    root = tk.Tk()
    root.title("L-System Test")
    root.geometry("400x300")
    
    label = tk.Label(root, text="L-System Fern Generator Test\nDependencies OK!", 
                    font=("Arial", 14))
    label.pack(expand=True)
    
    close_btn = tk.Button(root, text="Close", command=root.quit)
    close_btn.pack(pady=10)
    
    print("✓ Test window created - close it to continue")
    root.mainloop()
else:
    print("✗ Cannot create test window - tkinter not available")

print("Test completed.")