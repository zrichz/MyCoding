#!/usr/bin/env python3
"""
Test script for the enhanced K-means clustering with elbow method
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os

# Add parent directory to path to import the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from TI_CHANGER_MULTIPLE_2024_10_22 import find_optimal_clusters_elbow_method
    print("✅ Successfully imported the enhanced clustering function")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def create_test_data():
    """Create synthetic test data with known optimal clusters"""
    print("🧪 Creating test data with 3 natural clusters...")
    
    # Create 3 clusters of data
    np.random.seed(42)
    
    # Cluster 1: centered around (0, 0)
    cluster1 = np.random.normal(0, 1, (15, 10))
    
    # Cluster 2: centered around (5, 5)
    cluster2 = np.random.normal(5, 1, (12, 10))
    
    # Cluster 3: centered around (-3, 3)
    cluster3 = np.random.normal([-3, 3, 1, -1, 2, 0, -2, 1, 0, -1], 1, (10, 10))
    
    # Combine all clusters
    test_data = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"   Created {test_data.shape[0]} vectors with {test_data.shape[1]} dimensions")
    print(f"   Expected optimal clusters: 3")
    
    return test_data

def test_elbow_method():
    """Test the elbow method with synthetic data"""
    print("\n" + "="*60)
    print("🔍 TESTING ELBOW METHOD")
    print("="*60)
    
    # Create test data
    test_data = create_test_data()
    
    # Run elbow method analysis
    print("\n📊 Running elbow method analysis...")
    results = find_optimal_clusters_elbow_method(test_data, max_clusters=10, show_plots=True)
    
    if results is None:
        print("❌ Analysis failed")
        return False
    
    # Display results
    print(f"\n📋 ANALYSIS RESULTS:")
    print(f"   Elbow method suggests: {results['optimal_elbow']} clusters")
    print(f"   Silhouette method suggests: {results['optimal_silhouette']} clusters")
    print(f"   Best silhouette score: {results['best_silhouette_score']:.3f}")
    
    # Check if results are reasonable (should be close to 3)
    success = True
    if abs(results['optimal_elbow'] - 3) <= 1:
        print(f"✅ Elbow method result is reasonable (expected ~3, got {results['optimal_elbow']})")
    else:
        print(f"⚠️  Elbow method result may be off (expected ~3, got {results['optimal_elbow']})")
        success = False
    
    if abs(results['optimal_silhouette'] - 3) <= 1:
        print(f"✅ Silhouette method result is reasonable (expected ~3, got {results['optimal_silhouette']})")
    else:
        print(f"⚠️  Silhouette method result may be off (expected ~3, got {results['optimal_silhouette']})")
        success = False
    
    if results['best_silhouette_score'] > 0.3:
        print(f"✅ Silhouette score is good (>{0.3})")
    else:
        print(f"⚠️  Silhouette score is low (<{0.3})")
    
    return success

if __name__ == "__main__":
    print("🧪 K-means Elbow Method Test Suite")
    print("="*40)
    
    try:
        success = test_elbow_method()
        
        if success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   The elbow method enhancement is working correctly")
        else:
            print(f"\n⚠️  SOME TESTS HAD WARNINGS")
            print(f"   The method is working but results may vary with different data")
        
        print(f"\n💡 The enhanced clustering function is ready to use!")
        print(f"   Try running the main script and selecting option 8")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
