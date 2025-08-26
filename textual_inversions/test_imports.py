#!/usr/bin/env python3
"""Quick test script to verify required imports."""

print("Testing required imports...")
print("-" * 40)

try:
    import torch
    print("✓ PyTorch: OK")
except ImportError as e:
    print(f"✗ PyTorch: Missing ({e})")

try:
    import numpy
    print("✓ NumPy: OK")
except ImportError as e:
    print(f"✗ NumPy: Missing ({e})")

try:
    import matplotlib.pyplot
    print("✓ Matplotlib: OK")
except ImportError as e:
    print(f"✗ Matplotlib: Missing ({e})")

try:
    import tkinter
    print("✓ Tkinter: OK")
except ImportError as e:
    print(f"✗ Tkinter: Missing ({e})")

print("-" * 40)
print("Import test complete.")
