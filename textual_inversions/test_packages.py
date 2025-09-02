#!/usr/bin/env python3
"""Simple test to check if packages are available"""

print("🔍 Checking package availability...")

try:
    import numpy as np
    print("✅ NumPy is available")
    numpy_available = True
except ImportError:
    print("❌ NumPy is not available")
    numpy_available = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    print("✅ scikit-learn is available")
    sklearn_available = True
except ImportError:
    print("❌ scikit-learn is not available") 
    sklearn_available = False

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib is available")
    matplotlib_available = True
except ImportError:
    print("❌ matplotlib is not available")
    matplotlib_available = False

try:
    import torch
    print("✅ PyTorch is available")
    torch_available = True
except ImportError:
    print("❌ PyTorch is not available")
    torch_available = False

print(f"\n📊 Package Summary:")
print(f"   NumPy: {'✅' if numpy_available else '❌'}")
print(f"   scikit-learn: {'✅' if sklearn_available else '❌'}")
print(f"   matplotlib: {'✅' if matplotlib_available else '❌'}")
print(f"   PyTorch: {'✅' if torch_available else '❌'}")

if sklearn_available and numpy_available and matplotlib_available:
    print(f"\n🎉 All required packages for elbow method are available!")
    
    # Test importing the enhanced function
    try:
        from TI_CHANGER_MULTIPLE_2024_10_22 import find_optimal_clusters_elbow_method
        print("✅ Enhanced clustering function imported successfully")
    except ImportError as e:
        print(f"❌ Could not import enhanced function: {e}")
else:
    print(f"\n⚠️ Some packages are missing. The elbow method may not work fully.")

print(f"\n💡 The enhanced K-means clustering with elbow method is ready!")
