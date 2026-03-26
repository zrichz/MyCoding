"""
TI Changer SDXL - Convert and manipulate SDXL textual inversion files
Adapted for Stable Diffusion XL which uses dual embeddings (CLIP-L and CLIP-G)
"""

# Load a .safetensors textual inversion file and show / manipulate it
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import sys
import time
import math
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.ndimage import zoom
import safetensors.torch as st

# Configure matplotlib for better display (standalone Python version)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

# ========================================
# VISUALIZATION CONFIGURATION
# ========================================
# Using standard matplotlib divergent colormap for heatmap visualizations
# Available options: 'RdBu', 'coolwarm', 'seismic', 'RdYlBu', 'bwr'

# Heatmap dimensions (adjust based on your vector size preferences)
HEATMAP_HEIGHT_CLIP_L = 24
HEATMAP_WIDTH_CLIP_L = 32  # 768 / 24 = 32

HEATMAP_HEIGHT_CLIP_G = 32
HEATMAP_WIDTH_CLIP_G = 40  # 1280 / 32 = 40


def analyze_safetensors_file(filepath):
    """
    Analyze a .safetensors file to determine its type and structure
    Specifically for SDXL textual inversions which have clip_l and clip_g embeddings
    """
    try:
        data = st.load_file(filepath)
        print(f"\nFile Analysis: {os.path.basename(filepath)}")
        print("=" * 50)
        
        # Check basic structure
        if isinstance(data, dict):
            print("✓ File is a safetensors dictionary file")
            print(f"Top-level keys: {list(data.keys())}")
            
            # Check for SDXL TI structure (clip_l and clip_g)
            has_clip_l = 'clip_l' in data or 'emb_l' in data
            has_clip_g = 'clip_g' in data or 'emb_g' in data
            
            if has_clip_l and has_clip_g:
                clip_l_key = 'clip_l' if 'clip_l' in data else 'emb_l'
                clip_g_key = 'clip_g' if 'clip_g' in data else 'emb_g'
                
                clip_l_tensor = data[clip_l_key]
                clip_g_tensor = data[clip_g_key]
                
                print(f"✓ Valid SDXL Textual Inversion detected!")
                print(f"✓ CLIP-L tensor ({clip_l_key}): {clip_l_tensor.shape} (expected: [N, 768])")
                print(f"✓ CLIP-G tensor ({clip_g_key}): {clip_g_tensor.shape} (expected: [N, 1280])")
                
                if len(clip_l_tensor.shape) == 2 and len(clip_g_tensor.shape) == 2:
                    print(f"✓ Number of vectors (CLIP-L): {clip_l_tensor.shape[0]}")
                    print(f"✓ Number of vectors (CLIP-G): {clip_g_tensor.shape[0]}")
                    print(f"✓ CLIP-L dimensions: {clip_l_tensor.shape[1]}")
                    print(f"✓ CLIP-G dimensions: {clip_g_tensor.shape[1]}")
                    return True
                else:
                    print("✗ Unexpected tensor shapes for SDXL TI")
                    return False
            else:
                print("✗ Not a standard SDXL TI file (missing clip_l or clip_g)")
                print("Available keys:", list(data.keys()))
                return False
        else:
            print(f"✗ File contains: {type(data)}")
            print("✗ Expected a dictionary structure for SDXL textual inversions")
            
        return False
        
    except Exception as e:
        print(f"✗ Error analyzing file: {e}")
        return False


def load_ti_file_sdxl(filepath):
    """
    Load an SDXL TI file with flexible structure detection
    Returns: (data_dict, clip_l_tensor, clip_g_tensor) or None if not a valid SDXL TI file
    """
    try:
        data = st.load_file(filepath)
        
        # Handle dictionary files
        if isinstance(data, dict):
            # Check for CLIP-L and CLIP-G embeddings
            clip_l_key = None
            clip_g_key = None
            
            # Try different key variations
            for key in data.keys():
                if 'clip_l' in key.lower() or 'emb_l' in key.lower():
                    clip_l_key = key
                elif 'clip_g' in key.lower() or 'emb_g' in key.lower():
                    clip_g_key = key
            
            if clip_l_key and clip_g_key:
                clip_l_tensor = data[clip_l_key]
                clip_g_tensor = data[clip_g_key]
                
                # Validate tensor shapes
                if len(clip_l_tensor.shape) == 2 and len(clip_g_tensor.shape) == 2:
                    print(f"✅ Loaded SDXL TI successfully")
                    print(f"   CLIP-L: {clip_l_tensor.shape}")
                    print(f"   CLIP-G: {clip_g_tensor.shape}")
                    
                    return (data, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
        
        return None
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def get_user_choice():
    """
    User input function for operation selection
    """
    import sys
    
    # Clear any potential input buffer issues
    if hasattr(sys.stdin, 'flush'):
        sys.stdin.flush()
    
    print("\n" + "="*60)
    print("TI CHANGER SDXL - OPERATIONS")
    print("="*60)
    print("🟢 SDXL DUAL-EMBEDDING FILE: Operations affect both CLIP-L and CLIP-G")
    print("="*60)
    
    print(f"1. Apply smoothing to all vectors")
    print(f"2. Create single mean vector (condensed)")
    print(f"3. Divide all vectors by scalar")
    print(f"4. Extract individual vectors to separate files")
    print(f"5. Save top N vectors by absolute magnitude")
    print(f"6. K-means Clustering Reduction")
    print(f"7. Principal Component Analysis (PCA)")
    print(f"8. Quantile Transform (Uniform/Gaussian)")
    print(f"9. Nonlinear Squashing (tanh)")
    print(f"10. L² Normalization")
    print(f"11. Average specified vectors and combine with remaining")
    print("12. Compare two TI files (vector-by-vector analysis)")
    print("13. Correlation Matrix Visualization")
    print("="*60)
    
    valid_ops = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    
    # Improved input handling with retry logic
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Add a small delay to ensure console is ready
            import time
            time.sleep(0.1)
            
            print(f"\nAttempt {attempt + 1}: ", end="", flush=True)
            user_input = input("Choose operation (1-13): ").strip()
            
            if user_input in valid_ops:
                return user_input
            else:
                print(f"Invalid input: '{user_input}'. Please enter a number from 1 to 13.")
                if attempt < max_attempts - 1:
                    print("Try again...")
                    continue
        except (EOFError, KeyboardInterrupt):
            print("\nInput interrupted. Please try again...")
            if attempt < max_attempts - 1:
                continue
            else:
                print("Too many failed attempts. Exiting...")
                return None
    
    print("Invalid choice after multiple attempts. Please run the script again and enter 1-13.")
    return None


def extract_individual_vectors(data, original_filename, numvectors, clip_l_key, clip_g_key):
    """
    Extract individual vectors from a multi-vector SDXL TI file and save each as a separate .safetensors file
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        clip_l_key: Key name for CLIP-L embedding
        clip_g_key: Key name for CLIP-G embedding
    """
    print(f"\nExtracting {numvectors} individual vectors from '{original_filename}'...")
    
    # Get the original tensors
    clip_l_tensor = data[clip_l_key]
    clip_g_tensor = data[clip_g_key]
    
    # Create base filename (remove .safetensors extension if present)
    base_filename = original_filename.replace('.safetensors', '')
    
    # Ask user for confirmation and naming preference
    print(f"\nThis will create {numvectors} separate .safetensors files:")
    for i in range(numvectors):
        print(f"  {base_filename}_v_{i+1:02d}.safetensors")
    
    confirm = input(f"\nProceed with extraction? (y/n): ").lower().strip()
    if confirm != 'y' and confirm != 'yes':
        print("Extraction cancelled.")
        return
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Extract and save each vector
    successful_extractions = 0
    
    for i in range(numvectors):
        try:
            # Extract the i-th vector (keep as 2D with shape [1, dims])
            individual_clip_l = clip_l_tensor[i:i+1].clone()
            individual_clip_g = clip_g_tensor[i:i+1].clone()
            
            # Create new data dictionary for this individual vector
            individual_data = {
                clip_l_key: individual_clip_l,
                clip_g_key: individual_clip_g
            }
            
            # Generate filename
            vector_filename = f"{base_filename}_v_{i+1:02d}.safetensors"
            filepath = os.path.join(directory, vector_filename)
            
            # Save the individual vector file
            st.save_file(individual_data, filepath)
            
            print(f"✓ Saved vector {i+1}/{numvectors}: {vector_filename}")
            successful_extractions += 1
            
        except Exception as e:
            print(f"✗ Error saving vector {i+1}: {e}")
    
    print(f"\nExtraction complete! Successfully created {successful_extractions}/{numvectors} individual files.")
    
    if successful_extractions > 0:
        print(f"Files saved in '{directory}' directory.")


def select_top_n_vectors(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """
    Select and save the top N vectors based on combined absolute magnitude from both embeddings
    """
    print(f"\nSelecting top N vectors by absolute magnitude from '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    # Get N from user
    while True:
        try:
            n = int(input(f"Enter number of top vectors to keep (1-{numvectors}): "))
            if 1 <= n <= numvectors:
                break
            else:
                print(f"Please enter a number between 1 and {numvectors}")
        except ValueError:
            print("Please enter a valid integer")
    
    # Convert to numpy
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    # Calculate combined absolute magnitude for each vector (CLIP-L + CLIP-G)
    vector_magnitudes = []
    for i in range(numvectors):
        mag_l = np.linalg.norm(clip_l_np[i])
        mag_g = np.linalg.norm(clip_g_np[i])
        combined_mag = mag_l + mag_g  # Combined magnitude
        vector_magnitudes.append((i, combined_mag, mag_l, mag_g))
    
    # Sort by combined magnitude (descending)
    vector_magnitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N vector indices
    top_indices = [idx for idx, _, _, _ in vector_magnitudes[:n]]
    bottom_indices = [idx for idx, _, _, _ in vector_magnitudes[n:]]
    top_indices.sort()
    bottom_indices.sort()
    
    print(f"\nTop {n} vectors by combined magnitude:")
    for rank, (orig_idx, combined_mag, mag_l, mag_g) in enumerate(vector_magnitudes[:n], 1):
        print(f"  Rank {rank}: Vector {orig_idx + 1} (CLIP-L: {mag_l:.4f}, CLIP-G: {mag_g:.4f}, Combined: {combined_mag:.4f})")
    
    # Create new tensors with only top N vectors
    top_clip_l = clip_l_tensor[top_indices]
    top_clip_g = clip_g_tensor[top_indices]
    
    # Create new data dictionary for top vectors
    top_data = {
        clip_l_key: top_clip_l,
        clip_g_key: top_clip_g
    }
    
    print(f"\nTop vectors tensor shapes:")
    print(f"  CLIP-L: {top_clip_l.shape}")
    print(f"  CLIP-G: {top_clip_g.shape}")
    
    # Generate filenames
    base_filename = original_filename.replace('.safetensors', '')
    top_filename = f"{base_filename}_TOP{n}.safetensors"
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the top vectors file
    top_filepath = os.path.join(directory, top_filename)
    st.save_file(top_data, top_filepath)
    
    print(f"✅ Top vectors file '{top_filename}' has been saved to the '{directory}' directory.")
    
    # Handle bottom vectors if any
    if len(bottom_indices) > 0:
        bottom_clip_l = clip_l_tensor[bottom_indices]
        bottom_clip_g = clip_g_tensor[bottom_indices]
        
        bottom_data = {
            clip_l_key: bottom_clip_l,
            clip_g_key: bottom_clip_g
        }
        
        bottom_count = len(bottom_indices)
        bottom_filename = f"{base_filename}_BOTTOM{bottom_count}.safetensors"
        bottom_filepath = os.path.join(directory, bottom_filename)
        st.save_file(bottom_data, bottom_filepath)
        
        print(f"✅ Bottom vectors file '{bottom_filename}' has been saved to the '{directory}' directory.")
        
        print(f"\n📊 Summary:")
        print(f"   Original vectors: {numvectors}")
        print(f"   Top {n} vectors saved as: {top_filename}")
        print(f"   Bottom {bottom_count} vectors saved as: {bottom_filename}")
    else:
        print(f"\n📊 Summary:")
        print(f"   All {numvectors} vectors were selected as top vectors.")


def find_optimal_clusters_elbow_method(np_array, max_clusters=None, show_plots=True):
    """
    Find optimal number of clusters using elbow method and silhouette analysis
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("❌ Error: scikit-learn is required for clustering analysis.")
        print("Please install it with: pip install scikit-learn")
        return None
    
    n_vectors = np_array.shape[0]
    
    # Set reasonable range for testing
    if max_clusters is None:
        max_clusters = min(20, n_vectors - 1)
    
    max_clusters = min(max_clusters, n_vectors - 1)
    
    if max_clusters < 2:
        print("❌ Cannot perform clustering analysis: need at least 2 vectors")
        return None
    
    print(f"\n🔍 Analyzing optimal number of clusters...")
    print(f"   Testing clusters from 2 to {max_clusters}")
    print(f"   Vector count: {n_vectors}")
    
    # Store metrics for analysis
    cluster_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    
    print("\n   Progress:")
    for k in cluster_range:
        print(f"   Testing k={k}...", end=" ")
        
        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(np_array)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        sil_score = silhouette_score(np_array, cluster_labels)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        
        print(f"Inertia: {inertia:.2f}, Silhouette: {sil_score:.3f}")
    
    # Find elbow using simple derivative method
    def find_elbow_point(values):
        """Find elbow using rate of change analysis"""
        if len(values) < 3:
            return 0
        
        derivatives = []
        for i in range(1, len(values) - 1):
            derivative = values[i-1] - 2*values[i] + values[i+1]
            derivatives.append(derivative)
        
        max_derivative_idx = np.argmax(derivatives)
        return max_derivative_idx + 1
    
    # Find optimal points
    elbow_idx = find_elbow_point(inertias)
    best_silhouette_idx = np.argmax(silhouette_scores)
    
    optimal_clusters_elbow = list(cluster_range)[elbow_idx] if elbow_idx < len(cluster_range) else cluster_range[0]
    optimal_clusters_silhouette = list(cluster_range)[best_silhouette_idx]
    
    # Create analysis plots if requested
    if show_plots:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Elbow Method (Inertia)
        plt.subplot(1, 3, 1)
        plt.plot(cluster_range, inertias, 'bo-', markersize=8, linewidth=2)
        plt.axvline(x=optimal_clusters_elbow, color='red', linestyle='--', 
                   label=f'Elbow at k={optimal_clusters_elbow}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method')
        plt.grid(True, alpha=0.3)
        plt.xticks(cluster_range)
        plt.legend()
        
        # Plot 2: Silhouette Analysis
        plt.subplot(1, 3, 2)
        plt.plot(cluster_range, silhouette_scores, 'go-', markersize=8, linewidth=2)
        plt.axvline(x=optimal_clusters_silhouette, color='red', linestyle='--',
                   label=f'Best Silhouette at k={optimal_clusters_silhouette}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.grid(True, alpha=0.3)
        plt.xticks(cluster_range)
        plt.legend()
        
        # Plot 3: Combined Analysis
        plt.subplot(1, 3, 3)
        norm_inertias = [(max(inertias) - x) / (max(inertias) - min(inertias)) for x in inertias]
        norm_silhouettes = [(x - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) for x in silhouette_scores]
        
        plt.plot(cluster_range, norm_inertias, 'b-', label='Normalized Inertia (inverted)', linewidth=2)
        plt.plot(cluster_range, norm_silhouettes, 'g-', label='Normalized Silhouette', linewidth=2)
        plt.axvline(x=optimal_clusters_elbow, color='blue', linestyle='--', alpha=0.7)
        plt.axvline(x=optimal_clusters_silhouette, color='green', linestyle='--', alpha=0.7)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Normalized Score')
        plt.title('Combined Analysis')
        plt.xticks(cluster_range)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Prepare results
    results = {
        'cluster_range': list(cluster_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_elbow': optimal_clusters_elbow,
        'optimal_silhouette': optimal_clusters_silhouette,
        'best_silhouette_score': max(silhouette_scores),
        'elbow_inertia': inertias[elbow_idx] if elbow_idx < len(inertias) else inertias[0]
    }
    
    return results


def clustering_based_reduction(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """
    Apply K-means clustering to reduce vectors to N cluster centroids
    Operates on concatenated CLIP-L + CLIP-G embeddings for unified clustering
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("❌ Error: scikit-learn is required for clustering-based reduction.")
        print("Please install it with: pip install scikit-learn")
        return
    
    print(f"\nApplying Clustering-Based Reduction (K-means) to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    if numvectors < 2:
        print("❌ Error: Need at least 2 vectors for clustering analysis")
        return
    
    # Convert to numpy and concatenate both embeddings
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    # Concatenate CLIP-L and CLIP-G for unified clustering
    combined_np = np.concatenate([clip_l_np, clip_g_np], axis=1)
    print(f"Combined embedding shape for clustering: {combined_np.shape}")
    
    # Perform elbow method analysis
    print("\n" + "="*60)
    print("🔍 OPTIMAL CLUSTER ANALYSIS")
    print("="*60)
    
    analysis = find_optimal_clusters_elbow_method(combined_np, max_clusters=numvectors-1, show_plots=True)
    
    if analysis is None:
        print("❌ Could not perform cluster analysis")
        return
    
    # Display analysis results
    print(f"\n📊 CLUSTERING ANALYSIS RESULTS:")
    print(f"   Optimal clusters (Elbow method): {analysis['optimal_elbow']}")
    print(f"   Optimal clusters (Silhouette): {analysis['optimal_silhouette']}")
    print(f"   Best silhouette score: {analysis['best_silhouette_score']:.3f}")
    
    # Recommend optimal choice
    if analysis['optimal_elbow'] == analysis['optimal_silhouette']:
        recommended = analysis['optimal_elbow']
        print(f"\n🎯 STRONG RECOMMENDATION: {recommended} clusters")
    else:
        recommended = analysis['optimal_silhouette']
        print(f"\n🤔 MIXED SIGNALS - RECOMMENDED: {recommended} clusters (better silhouette)")
    
    # Get user choice
    print(f"\n" + "="*60)
    print("🎯 CLUSTER SELECTION")
    print("="*60)
    print(f"Recommended: {recommended} clusters")
    print(f"\nOptions:")
    print(f"1. Accept recommendation ({recommended} clusters)")
    print(f"2. Use elbow method ({analysis['optimal_elbow']} clusters)")
    print(f"3. Enter custom number")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-3): ").strip()
            if choice == "1":
                n_clusters = recommended
                break
            elif choice == "2":
                n_clusters = analysis['optimal_elbow']
                break
            elif choice == "3":
                n_clusters = int(input(f"Enter number of clusters (1-{numvectors}): "))
                if 1 <= n_clusters <= numvectors:
                    break
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid choice")
    
    print(f"\n🚀 Applying K-means clustering with {n_clusters} clusters...")
    
    # Apply K-means clustering on combined embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(combined_np)
    centroids_combined = kmeans.cluster_centers_
    
    # Split centroids back into CLIP-L and CLIP-G
    clip_l_dims = clip_l_np.shape[1]
    centroids_clip_l = centroids_combined[:, :clip_l_dims]
    centroids_clip_g = centroids_combined[:, clip_l_dims:]
    
    print(f"✓ Clustering complete!")
    print(f"✓ CLIP-L centroids shape: {centroids_clip_l.shape}")
    print(f"✓ CLIP-G centroids shape: {centroids_clip_g.shape}")
    
    # Create base filename
    base_filename = original_filename.replace('.safetensors', '')
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save combined centroids file
    print(f"\n💾 Saving combined centroids file...")
    centroids_clip_l_tensor = torch.tensor(centroids_clip_l, dtype=torch.float32)
    centroids_clip_g_tensor = torch.tensor(centroids_clip_g, dtype=torch.float32)
    
    combined_data = {
        clip_l_key: centroids_clip_l_tensor,
        clip_g_key: centroids_clip_g_tensor
    }
    
    combined_filename = f"{base_filename}_kmeans_{n_clusters}centroids.safetensors"
    combined_filepath = os.path.join(directory, combined_filename)
    st.save_file(combined_data, combined_filepath)
    
    print(f"✅ Combined centroids file '{combined_filename}' saved.")
    print(f"\n📊 K-MEANS CLUSTERING SUMMARY:")
    print(f"   Original vectors: {numvectors}")
    print(f"   Clusters created: {n_clusters}")
    print(f"   Compression ratio: {numvectors/n_clusters:.2f}:1")


def analyze_pca_components(np_array, show_plots=True):
    """
    Analyze PCA to recommend optimal number of components
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("❌ Error: scikit-learn is required for PCA analysis.")
        return None
    
    n_vectors = np_array.shape[0]
    
    print(f"\n🔍 Analyzing PCA components...")
    print(f"   Vector count: {n_vectors}")
    print(f"   Vector dimensions: {np_array.shape[1]}")
    
    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(np_array)
    
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find components for variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    variance_recommendations = {}
    
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        if n_components < len(cumulative_variance):
            variance_recommendations[f"{threshold*100:.0f}%"] = n_components
    
    # Elbow method
    if len(explained_variance_ratio) > 2:
        variance_drops = np.diff(explained_variance_ratio)
        elbow_point = np.argmax(np.abs(variance_drops)) + 1
    else:
        elbow_point = 1
    
    # Significant components
    significant_components = 0
    for i, var_ratio in enumerate(explained_variance_ratio):
        if var_ratio > 0.01 and cumulative_variance[i] < 0.99:
            significant_components = i + 1
        else:
            break
    significant_components = max(1, significant_components)
    
    # Plots
    if show_plots:
        fig = plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', markersize=6)
        plt.axvline(x=float(elbow_point), color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', markersize=6)
        for threshold in [0.90, 0.95]:
            plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.7)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Variance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Prepare results
    results = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'variance_recommendations': variance_recommendations,
        'elbow_point': elbow_point,
        'significant_components': significant_components
    }
    
    return results


def principal_component_analysis(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """
    Apply PCA to reduce the number of vectors using concatenated embeddings
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("❌ Error: scikit-learn is required for PCA.")
        return
    
    print(f"\nApplying PCA to '{original_filename}'...")
    
    if numvectors < 2:
        print("❌ Error: Need at least 2 vectors for PCA")
        return
    
    # Convert to numpy and concatenate
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    combined_np = np.concatenate([clip_l_np, clip_g_np], axis=1)
    
    # Perform PCA analysis
    analysis = analyze_pca_components(combined_np, show_plots=True)
    
    if analysis is None:
        return
    
    # Get user choice for number of components
    recommended = analysis['variance_recommendations'].get('90%', analysis['significant_components'])
    print(f"\n🌟 RECOMMENDED: {recommended} components")
    
    print(f"\nOptions:")
    print(f"1. Accept recommendation ({recommended})")
    print(f"2. Use 90% variance ({analysis['variance_recommendations'].get('90%', 'N/A')})")
    print(f"3. Use 95% variance ({analysis['variance_recommendations'].get('95%', 'N/A')})")
    print(f"4. Enter custom number")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-4): ").strip()
            if choice == "1":
                n_components = recommended
                break
            elif choice == "2" and '90%' in analysis['variance_recommendations']:
                n_components = analysis['variance_recommendations']['90%']
                break
            elif choice == "3" and '95%' in analysis['variance_recommendations']:
                n_components = analysis['variance_recommendations']['95%']
                break
            elif choice == "4":
                n_components = int(input(f"Enter number (1-{numvectors}): "))
                if 1 <= n_components <= numvectors:
                    break
        except ValueError:
            print("Please enter valid choice")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_vectors = pca.fit_transform(combined_np)
    reconstructed = pca.inverse_transform(pca_vectors)
    
    # Split back into CLIP-L and CLIP-G
    clip_l_dims = clip_l_np.shape[1]
    reconstructed_clip_l = reconstructed[:, :clip_l_dims]
    reconstructed_clip_g = reconstructed[:, clip_l_dims:]
    
    print(f"✓ PCA complete!")
    
    # Save files
    base_filename = original_filename.replace('.safetensors', '')
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    clip_l_tensor_new = torch.tensor(reconstructed_clip_l, dtype=torch.float32)
    clip_g_tensor_new = torch.tensor(reconstructed_clip_g, dtype=torch.float32)
    
    combined_data = {
        clip_l_key: clip_l_tensor_new,
        clip_g_key: clip_g_tensor_new
    }
    
    total_var = np.sum(pca.explained_variance_ratio_)
    filename = f"{base_filename}_pca_{n_components}comp_var{total_var*100:.1f}pct.safetensors"
    filepath = os.path.join(directory, filename)
    st.save_file(combined_data, filepath)
    
    print(f"✅ PCA file saved: {filename}")


def show_transformation_plots(original_data, transformed_data, title, filename):
    """Show before/after histograms"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    orig_flat = original_data.flatten()
    trans_flat = transformed_data.flatten()
    
    ax1.hist(orig_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Before: Original Data Distribution')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(trans_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('After: Transformed Data Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} Transformation\nFile: {filename}', fontsize=12)
    plt.tight_layout()
    plt.show()


def quantile_transform(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """Apply quantile transformation to both embeddings"""
    try:
        from scipy import stats
    except ImportError:
        print("❌ Error: scipy required")
        return
    
    print(f"\nApplying Quantile Transform to '{original_filename}'...")
    
    print("\nChoose distribution:")
    print("1. Uniform (0 to 1)")
    print("2. Gaussian (normal)")
    
    dist_choice = input("Enter choice (1 or 2): ").strip()
    
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    # Transform CLIP-L
    transformed_l = np.zeros_like(clip_l_np)
    for i in range(numvectors):
        vector = clip_l_np[i].copy()
        sorted_indices = np.argsort(vector)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(vector))
        empirical_cdf = (ranks + 1) / (len(vector) + 1)
        
        if dist_choice == '1':
            transformed_vector = empirical_cdf
        else:
            transformed_vector = stats.norm.ppf(empirical_cdf)
        
        min_val, max_val = transformed_vector.min(), transformed_vector.max()
        if max_val > min_val:
            transformed_vector = (transformed_vector - min_val) / (max_val - min_val) - 0.5
        
        transformed_l[i] = transformed_vector
    
    # Transform CLIP-G
    transformed_g = np.zeros_like(clip_g_np)
    for i in range(numvectors):
        vector = clip_g_np[i].copy()
        sorted_indices = np.argsort(vector)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(vector))
        empirical_cdf = (ranks + 1) / (len(vector) + 1)
        
        if dist_choice == '1':
            transformed_vector = empirical_cdf
        else:
            transformed_vector = stats.norm.ppf(empirical_cdf)
        
        min_val, max_val = transformed_vector.min(), transformed_vector.max()
        if max_val > min_val:
            transformed_vector = (transformed_vector - min_val) / (max_val - min_val) - 0.5
        
        transformed_g[i] = transformed_vector
    
    # Save
    suffix = "_quantile_uniform" if dist_choice == '1' else "_quantile_gaussian"
    save_transformed_file(data, original_filename, transformed_l, transformed_g, suffix, clip_l_key, clip_g_key)


def nonlinear_squashing_tanh(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """Apply tanh squashing"""
    print(f"\nApplying tanh squashing...")
    
    scale = float(input("Enter scale parameter (0.5-10.0): "))
    
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    transformed_l = np.tanh(clip_l_np * scale) * 0.5
    transformed_g = np.tanh(clip_g_np * scale) * 0.5
    
    suffix = f"_tanh_scale{scale}"
    save_transformed_file(data, original_filename, transformed_l, transformed_g, suffix, clip_l_key, clip_g_key)


def l2_normalization(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """Apply L2 normalization"""
    print(f"\nApplying L² Normalization...")
    
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    transformed_l = np.zeros_like(clip_l_np)
    transformed_g = np.zeros_like(clip_g_np)
    
    for i in range(numvectors):
        # CLIP-L
        vector_l = clip_l_np[i].copy()
        l2_norm = np.linalg.norm(vector_l)
        if l2_norm > 0:
            normalized = vector_l / l2_norm
            min_val, max_val = normalized.min(), normalized.max()
            if max_val > min_val:
                transformed_l[i] = (normalized - min_val) / (max_val - min_val) - 0.5
        
        # CLIP-G
        vector_g = clip_g_np[i].copy()
        l2_norm = np.linalg.norm(vector_g)
        if l2_norm > 0:
            normalized = vector_g / l2_norm
            min_val, max_val = normalized.min(), normalized.max()
            if max_val > min_val:
                transformed_g[i] = (normalized - min_val) / (max_val - min_val) - 0.5
    
    suffix = "_l2norm"
    save_transformed_file(data, original_filename, transformed_l, transformed_g, suffix, clip_l_key, clip_g_key)


def visualize_correlation_matrix(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """Visualize correlation matrix of vectors within the same TI file"""
    print(f"\nGenerating Correlation Matrix for '{original_filename}'...")
    print(f"Number of vectors: {numvectors}")
    
    if numvectors < 2:
        print("❌ Need at least 2 vectors to compute correlation matrix")
        return
    
    # Convert to numpy
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    # Compute correlation matrices
    # Each row is a vector, compute correlation between rows
    corr_matrix_l = np.corrcoef(clip_l_np)
    corr_matrix_g = np.corrcoef(clip_g_np)
    
    print(f"\n📊 CORRELATION STATISTICS")
    print("="*60)
    
    # CLIP-L statistics
    # Get off-diagonal elements (exclude self-correlation)
    mask = ~np.eye(numvectors, dtype=bool)
    off_diag_l = corr_matrix_l[mask]
    
    print(f"\nCLIP-L Correlations:")
    print(f"  Mean correlation: {np.mean(off_diag_l):.4f}")
    print(f"  Std deviation: {np.std(off_diag_l):.4f}")
    print(f"  Min correlation: {np.min(off_diag_l):.4f}")
    print(f"  Max correlation: {np.max(off_diag_l):.4f}")
    
    # CLIP-G statistics
    off_diag_g = corr_matrix_g[mask]
    
    print(f"\nCLIP-G Correlations:")
    print(f"  Mean correlation: {np.mean(off_diag_g):.4f}")
    print(f"  Std deviation: {np.std(off_diag_g):.4f}")
    print(f"  Min correlation: {np.min(off_diag_g):.4f}")
    print(f"  Max correlation: {np.max(off_diag_g):.4f}")
    
    # Find most and least correlated pairs
    print(f"\n🔍 MOST/LEAST CORRELATED PAIRS")
    print("="*60)
    
    # For CLIP-L
    max_idx_l = np.unravel_index(np.argmax(corr_matrix_l * (1 - np.eye(numvectors))), corr_matrix_l.shape)
    min_idx_l = np.unravel_index(np.argmin(corr_matrix_l + np.eye(numvectors) * 10), corr_matrix_l.shape)
    
    print(f"\nCLIP-L:")
    print(f"  Most correlated: Vector {max_idx_l[0]+1} & Vector {max_idx_l[1]+1} (r = {corr_matrix_l[max_idx_l]:.4f})")
    print(f"  Least correlated: Vector {min_idx_l[0]+1} & Vector {min_idx_l[1]+1} (r = {corr_matrix_l[min_idx_l]:.4f})")
    
    # For CLIP-G
    max_idx_g = np.unravel_index(np.argmax(corr_matrix_g * (1 - np.eye(numvectors))), corr_matrix_g.shape)
    min_idx_g = np.unravel_index(np.argmin(corr_matrix_g + np.eye(numvectors) * 10), corr_matrix_g.shape)
    
    print(f"\nCLIP-G:")
    print(f"  Most correlated: Vector {max_idx_g[0]+1} & Vector {max_idx_g[1]+1} (r = {corr_matrix_g[max_idx_g]:.4f})")
    print(f"  Least correlated: Vector {min_idx_g[0]+1} & Vector {min_idx_g[1]+1} (r = {corr_matrix_g[min_idx_g]:.4f})")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # CLIP-L correlation matrix
    im1 = ax1.imshow(corr_matrix_l, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'CLIP-L Correlation Matrix\n{original_filename}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Vector Number', fontsize=10)
    ax1.set_ylabel('Vector Number', fontsize=10)
    
    # Set ticks
    tick_positions = np.arange(numvectors)
    tick_labels = [str(i+1) for i in range(numvectors)]
    ax1.set_xticks(tick_positions)
    ax1.set_yticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticklabels(tick_labels)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Add text annotations for values
    for i in range(numvectors):
        for j in range(numvectors):
            text = ax1.text(j, i, f'{corr_matrix_l[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix_l[i, j]) < 0.5 else "white",
                          fontsize=8 if numvectors <= 10 else 6)
    
    # CLIP-G correlation matrix
    im2 = ax2.imshow(corr_matrix_g, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title(f'CLIP-G Correlation Matrix\n{original_filename}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Vector Number', fontsize=10)
    ax2.set_ylabel('Vector Number', fontsize=10)
    
    # Set ticks
    ax2.set_xticks(tick_positions)
    ax2.set_yticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticklabels(tick_labels)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Add text annotations for values
    for i in range(numvectors):
        for j in range(numvectors):
            text = ax2.text(j, i, f'{corr_matrix_g[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix_g[i, j]) < 0.5 else "white",
                          fontsize=8 if numvectors <= 10 else 6)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Correlation matrix visualization complete!")
    
    # Ask user if they want to remove redundant vectors
    print("\n" + "="*60)
    print("REMOVE REDUNDANT VECTORS")
    print("="*60)
    
    remove_redundant = input("\nRemove redundant/duplicate vectors? (y/n): ").strip().lower()
    
    if remove_redundant in ['y', 'yes']:
        # Get threshold from user
        while True:
            try:
                threshold_input = input("Enter correlation threshold (0.50-0.99, default 0.99): ").strip()
                if threshold_input == '':
                    threshold = 0.99
                    break
                threshold = float(threshold_input)
                if 0.50 <= threshold <= 0.99:
                    break
                else:
                    print("Please enter a value between 0.50 and 0.99")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"\nAnalyzing vectors with correlation threshold > {threshold}...")
        
        # Identify redundant vectors
        # We'll use the maximum correlation from both CLIP-L and CLIP-G
        # A vector is redundant if it has high correlation to ANY previous vector
        vectors_to_keep = []
        vectors_to_remove = []
        redundant_pairs = []
        
        for i in range(numvectors):
            is_redundant = False
            for j in vectors_to_keep:
                # Check both CLIP-L and CLIP-G correlations
                corr_l = corr_matrix_l[i, j]
                corr_g = corr_matrix_g[i, j]
                
                # Consider redundant if BOTH embeddings show high correlation
                if corr_l > threshold and corr_g > threshold:
                    is_redundant = True
                    vectors_to_remove.append(i)
                    redundant_pairs.append((i+1, j+1, corr_l, corr_g))
                    print(f"  Vector {i+1} is redundant with Vector {j+1} (CLIP-L: {corr_l:.4f}, CLIP-G: {corr_g:.4f})")
                    break
            
            if not is_redundant:
                vectors_to_keep.append(i)
        
        print(f"\n📊 REDUNDANCY ANALYSIS:")
        print(f"   Original vectors: {numvectors}")
        print(f"   Vectors to keep: {len(vectors_to_keep)}")
        print(f"   Vectors to remove: {len(vectors_to_remove)}")
        
        if len(vectors_to_remove) == 0:
            print("\n✅ No redundant vectors found! All vectors are unique.")
            return
        
        print(f"\n   Kept vector indices: {[i+1 for i in vectors_to_keep]}")
        print(f"   Removed vector indices: {[i+1 for i in vectors_to_remove]}")
        
        # Confirm with user
        confirm = input(f"\nSave streamlined TI file with {len(vectors_to_keep)} vectors? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            # Create new tensors with only the kept vectors
            clip_l_kept = clip_l_tensor[vectors_to_keep]
            clip_g_kept = clip_g_tensor[vectors_to_keep]
            
            # Create new data dictionary
            streamlined_data = {
                clip_l_key: clip_l_kept,
                clip_g_key: clip_g_kept
            }
            
            # Generate filename
            base_filename = original_filename.replace('.safetensors', '')
            new_filename = f"{base_filename}_no_redundants_th{threshold}.safetensors"
            
            # Create directory if it doesn't exist
            directory = "textual_inversions"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save the file
            filepath = os.path.join(directory, new_filename)
            st.save_file(streamlined_data, filepath)
            
            print(f"\n✅ Streamlined TI file saved: {new_filename}")
            print(f"   Location: {directory}/")
            print(f"   Reduction: {numvectors} → {len(vectors_to_keep)} vectors ({len(vectors_to_remove)} removed)")
        else:
            print("\n❌ Save cancelled.")


def compare_two_files():
    """Compare two SDXL TI files vector-by-vector"""
    print("\n" + "="*60)
    print("COMPARE TWO SDXL TI FILES")
    print("="*60)
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Load first file
    print("\nSelect FIRST file to compare...")
    file1_path = filedialog.askopenfilename(
        title="Select FIRST SDXL TI file",
        filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")],
        parent=root
    )
    
    if not file1_path:
        print("No file selected.")
        return
    
    # Load second file
    print("\nSelect SECOND file to compare...")
    file2_path = filedialog.askopenfilename(
        title="Select SECOND SDXL TI file",
        filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")],
        parent=root
    )
    
    if not file2_path:
        print("No file selected.")
        return
    
    # Load both files
    result1 = load_ti_file_sdxl(file1_path)
    result2 = load_ti_file_sdxl(file2_path)
    
    if not result1 or not result2:
        print("\n❌ Failed to load one or both files")
        return
    
    data1, clip_l_1, clip_g_1, clip_l_key_1, clip_g_key_1 = result1
    data2, clip_l_2, clip_g_2, clip_l_key_2, clip_g_key_2 = result2
    
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    
    print(f"\n📊 COMPARISON RESULTS")
    print("="*60)
    print(f"File 1: {file1_name}")
    print(f"  Vectors: {clip_l_1.shape[0]}")
    print(f"  CLIP-L shape: {clip_l_1.shape}")
    print(f"  CLIP-G shape: {clip_g_1.shape}")
    print(f"\nFile 2: {file2_name}")
    print(f"  Vectors: {clip_l_2.shape[0]}")
    print(f"  CLIP-L shape: {clip_l_2.shape}")
    print(f"  CLIP-G shape: {clip_g_2.shape}")
    
    # Convert to numpy
    clip_l_1_np = clip_l_1.cpu().detach().numpy()
    clip_g_1_np = clip_g_1.cpu().detach().numpy()
    clip_l_2_np = clip_l_2.cpu().detach().numpy()
    clip_g_2_np = clip_g_2.cpu().detach().numpy()
    
    # Check if dimensions match
    if clip_l_1.shape != clip_l_2.shape or clip_g_1.shape != clip_g_2.shape:
        print("\n⚠️  WARNING: Files have different shapes!")
        print("Can only perform partial comparison.")
        
        min_vectors = min(clip_l_1.shape[0], clip_l_2.shape[0])
        print(f"\nComparing first {min_vectors} vectors...")
    else:
        min_vectors = clip_l_1.shape[0]
        print(f"\n✓ Files have matching shapes")
    
    # Calculate differences for each vector
    print(f"\n" + "="*60)
    print("VECTOR-BY-VECTOR DIFFERENCES")
    print("="*60)
    
    diff_stats = []
    
    for i in range(min_vectors):
        # CLIP-L differences
        diff_l = clip_l_1_np[i] - clip_l_2_np[i]
        mae_l = np.mean(np.abs(diff_l))
        mse_l = np.mean(diff_l ** 2)
        max_diff_l = np.max(np.abs(diff_l))
        
        # CLIP-G differences
        diff_g = clip_g_1_np[i] - clip_g_2_np[i]
        mae_g = np.mean(np.abs(diff_g))
        mse_g = np.mean(diff_g ** 2)
        max_diff_g = np.max(np.abs(diff_g))
        
        # Combined metrics
        combined_mae = (mae_l + mae_g) / 2
        combined_mse = (mse_l + mse_g) / 2
        combined_max = max(max_diff_l, max_diff_g)
        
        diff_stats.append({
            'vector': i + 1,
            'mae_l': mae_l,
            'mae_g': mae_g,
            'mse_l': mse_l,
            'mse_g': mse_g,
            'max_diff_l': max_diff_l,
            'max_diff_g': max_diff_g,
            'combined_mae': combined_mae,
            'combined_mse': combined_mse,
            'combined_max': combined_max
        })
        
        print(f"\nVector {i+1}:")
        print(f"  CLIP-L - MAE: {mae_l:.6f}, MSE: {mse_l:.6f}, Max: {max_diff_l:.6f}")
        print(f"  CLIP-G - MAE: {mae_g:.6f}, MSE: {mse_g:.6f}, Max: {max_diff_g:.6f}")
        print(f"  Combined - MAE: {combined_mae:.6f}, MSE: {combined_mse:.6f}, Max: {combined_max:.6f}")
    
    # Overall statistics
    print(f"\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    avg_mae_l = np.mean([s['mae_l'] for s in diff_stats])
    avg_mae_g = np.mean([s['mae_g'] for s in diff_stats])
    avg_mse_l = np.mean([s['mse_l'] for s in diff_stats])
    avg_mse_g = np.mean([s['mse_g'] for s in diff_stats])
    
    print(f"Average MAE (CLIP-L): {avg_mae_l:.6f}")
    print(f"Average MAE (CLIP-G): {avg_mae_g:.6f}")
    print(f"Average MSE (CLIP-L): {avg_mse_l:.6f}")
    print(f"Average MSE (CLIP-G): {avg_mse_g:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    vectors = [s['vector'] for s in diff_stats]
    
    # MAE comparison
    axes[0, 0].plot(vectors, [s['mae_l'] for s in diff_stats], 'b-o', label='CLIP-L', markersize=6)
    axes[0, 0].plot(vectors, [s['mae_g'] for s in diff_stats], 'r-s', label='CLIP-G', markersize=6)
    axes[0, 0].set_title('Mean Absolute Error (MAE) per Vector')
    axes[0, 0].set_xlabel('Vector Number')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE comparison
    axes[0, 1].plot(vectors, [s['mse_l'] for s in diff_stats], 'b-o', label='CLIP-L', markersize=6)
    axes[0, 1].plot(vectors, [s['mse_g'] for s in diff_stats], 'r-s', label='CLIP-G', markersize=6)
    axes[0, 1].set_title('Mean Squared Error (MSE) per Vector')
    axes[0, 1].set_xlabel('Vector Number')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Max difference
    axes[1, 0].plot(vectors, [s['max_diff_l'] for s in diff_stats], 'b-o', label='CLIP-L', markersize=6)
    axes[1, 0].plot(vectors, [s['max_diff_g'] for s in diff_stats], 'r-s', label='CLIP-G', markersize=6)
    axes[1, 0].set_title('Maximum Difference per Vector')
    axes[1, 0].set_xlabel('Vector Number')
    axes[1, 0].set_ylabel('Max Diff')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics
    axes[1, 1].plot(vectors, [s['combined_mae'] for s in diff_stats], 'g-o', label='Combined MAE', markersize=6)
    axes[1, 1].plot(vectors, [s['combined_mse'] for s in diff_stats], 'purple', marker='s', label='Combined MSE', markersize=6)
    axes[1, 1].set_title('Combined Metrics per Vector')
    axes[1, 1].set_xlabel('Vector Number')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Comparison: {file1_name} vs {file2_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Comparison complete!")


def average_specified_vectors(data, original_filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key):
    """Average specified vectors"""
    print(f"\nAveraging specified vectors from '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    while True:
        try:
            user_input = input(f"Enter vector numbers to average (comma-separated): ").strip()
            vector_indices_str = [x.strip() for x in user_input.split(',')]
            vector_indices = [int(idx) - 1 for idx in vector_indices_str]
            
            if all(0 <= idx < numvectors for idx in vector_indices) and len(vector_indices) >= 2:
                break
            else:
                print(f"Please enter valid vector numbers (1-{numvectors})")
        except ValueError:
            print("Please enter valid numbers")
    
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    
    # Average specified vectors
    specified_l = clip_l_np[vector_indices]
    specified_g = clip_g_np[vector_indices]
    averaged_l = np.mean(specified_l, axis=0)
    averaged_g = np.mean(specified_g, axis=0)
    
    # Get remaining vectors
    all_indices = set(range(numvectors))
    remaining_indices = sorted(list(all_indices - set(vector_indices)))
    
    if remaining_indices:
        remaining_l = clip_l_np[remaining_indices]
        remaining_g = clip_g_np[remaining_indices]
        combined_l = np.vstack([remaining_l, averaged_l.reshape(1, -1)])
        combined_g = np.vstack([remaining_g, averaged_g.reshape(1, -1)])
    else:
        combined_l = averaged_l.reshape(1, -1)
        combined_g = averaged_g.reshape(1, -1)
    
    # Save
    base_filename = original_filename.replace('.safetensors', '')
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    vector_list_str = "_".join([str(i+1) for i in vector_indices])
    
    # Save combined
    combined_data = {
        clip_l_key: torch.tensor(combined_l, dtype=torch.float32),
        clip_g_key: torch.tensor(combined_g, dtype=torch.float32)
    }
    combined_filename = f"{base_filename}_combined_avg{vector_list_str}.safetensors"
    st.save_file(combined_data, os.path.join(directory, combined_filename))
    print(f"✅ Combined file saved: {combined_filename}")
    
    # Save averaged only
    averaged_data = {
        clip_l_key: torch.tensor(averaged_l.reshape(1, -1), dtype=torch.float32),
        clip_g_key: torch.tensor(averaged_g.reshape(1, -1), dtype=torch.float32)
    }
    averaged_filename = f"{base_filename}_avg{vector_list_str}.safetensors"
    st.save_file(averaged_data, os.path.join(directory, averaged_filename))
    print(f"✅ Averaged file saved: {averaged_filename}")


def save_transformed_file(data, original_filename, transformed_l, transformed_g, suffix, clip_l_key, clip_g_key):
    """Save transformed tensors"""
    clip_l_tensor = torch.tensor(transformed_l, dtype=torch.float32)
    clip_g_tensor = torch.tensor(transformed_g, dtype=torch.float32)
    
    new_data = {
        clip_l_key: clip_l_tensor,
        clip_g_key: clip_g_tensor
    }
    
    base_filename = original_filename.replace('.safetensors', '')
    final_filename = base_filename + suffix + ".safetensors"
    
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, final_filename)
    st.save_file(new_data, filepath)
    
    print(f"✅ Transformed file saved: {final_filename}")


def process_single_file():
    """Process a single SDXL TI file"""
    
    # Load file
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.geometry("800x600")
    
    print("Please select a .safetensors file to load...")
    filename_path = filedialog.askopenfilename(
        title="Select an SDXL TI .safetensors file",
        filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")],
        parent=root
    )
    
    if not filename_path:
        print("No file selected.")
        return False
    
    filename = os.path.basename(filename_path)
    
    # Analyze file
    print("\nAnalyzing selected file...")
    if not analyze_safetensors_file(filename_path):
        print("\n❌ Not a valid SDXL TI file")
        return False
    
    # Load file
    result = load_ti_file_sdxl(filename_path)
    if not result:
        print("❌ Failed to load file")
        return False
    
    data, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key = result
    
    # Convert to numpy
    clip_l_np = clip_l_tensor.cpu().detach().numpy()
    clip_g_np = clip_g_tensor.cpu().detach().numpy()
    numvectors = clip_l_np.shape[0]
    
    print(f'\n✅ SDXL TI loaded successfully')
    print(f'Number of vectors: {numvectors}')
    print(f'CLIP-L shape: {clip_l_np.shape}')
    print(f'CLIP-G shape: {clip_g_np.shape}')
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Statistics plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    vector_numbers = list(range(1, numvectors + 1))
    x_pos = np.arange(len(vector_numbers))
    
    # CLIP-L stats
    min_l = [np.min(clip_l_np[i]) for i in range(numvectors)]
    max_l = [np.max(clip_l_np[i]) for i in range(numvectors)]
    ax1.bar(x_pos, min_l, 0.7, label='Min', color='blue', alpha=0.6)
    ax1.bar(x_pos, max_l, 0.7, label='Max', color='red', alpha=0.6)
    ax1.set_title(f'CLIP-L Statistics ({filename})')
    ax1.set_xlabel('Vector Number')
    ax1.set_ylabel('Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(vector_numbers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # CLIP-G stats
    min_g = [np.min(clip_g_np[i]) for i in range(numvectors)]
    max_g = [np.max(clip_g_np[i]) for i in range(numvectors)]
    ax2.bar(x_pos, min_g, 0.7, label='Min', color='blue', alpha=0.6)
    ax2.bar(x_pos, max_g, 0.7, label='Max', color='red', alpha=0.6)
    ax2.set_title(f'CLIP-G Statistics ({filename})')
    ax2.set_xlabel('Vector Number')
    ax2.set_ylabel('Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(vector_numbers)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Get user choice
    user_input = get_user_choice()
    if user_input is None:
        return False
    
    print(f"\nSelected: Option {user_input}")
    
    # Execute operations
    if user_input == "4":
        extract_individual_vectors(data, filename, numvectors, clip_l_key, clip_g_key)
    elif user_input == "5":
        select_top_n_vectors(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "6":
        clustering_based_reduction(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "7":
        principal_component_analysis(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "8":
        quantile_transform(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "9":
        nonlinear_squashing_tanh(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "10":
        l2_normalization(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "11":
        average_specified_vectors(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "12":
        compare_two_files()
    elif user_input == "13":
        visualize_correlation_matrix(data, filename, numvectors, clip_l_tensor, clip_g_tensor, clip_l_key, clip_g_key)
    elif user_input == "1":
        # SMOOTHING
        print("You chose Option 1 - smoothing...")
        sm_kernel = int(input("Enter kernel size (must be odd, e.g., '3'), or '1' to skip: "))
        
        clip_l_np = clip_l_tensor.cpu().detach().numpy()
        clip_g_np = clip_g_tensor.cpu().detach().numpy()
        
        if sm_kernel == 1:
            processed_l = clip_l_np.copy()
            processed_g = clip_g_np.copy()
        else:
            # Smooth CLIP-L
            np_l_flat = clip_l_np.flatten()
            smooth_l = np.convolve(np_l_flat, np.ones(sm_kernel)/sm_kernel, mode='full')
            smooth_l = smooth_l[sm_kernel//2:len(smooth_l)-sm_kernel//2]
            processed_l = smooth_l.reshape(numvectors, -1)
            
            # Smooth CLIP-G
            np_g_flat = clip_g_np.flatten()
            smooth_g = np.convolve(np_g_flat, np.ones(sm_kernel)/sm_kernel, mode='full')
            smooth_g = smooth_g[sm_kernel//2:len(smooth_g)-sm_kernel//2]
            processed_g = smooth_g.reshape(numvectors, -1)
        
        suffix = f"_sm{sm_kernel}"
        save_transformed_file(data, filename, processed_l, processed_g, suffix, clip_l_key, clip_g_key)
        
    elif user_input == "2":
        # MEAN VECTOR
        print("You chose Option 2 - create single mean vector...")
        
        clip_l_np = clip_l_tensor.cpu().detach().numpy()
        clip_g_np = clip_g_tensor.cpu().detach().numpy()
        
        # Calculate mean vectors
        sd_values_l = [np.std(clip_l_np[i]) for i in range(numvectors)]
        meanSD_l = np.mean(sd_values_l)
        Xmean_l = np.mean(clip_l_np, axis=0)
        sd_val_l = np.std(Xmean_l)
        Xmean_l = Xmean_l * meanSD_l / sd_val_l
        
        sd_values_g = [np.std(clip_g_np[i]) for i in range(numvectors)]
        meanSD_g = np.mean(sd_values_g)
        Xmean_g = np.mean(clip_g_np, axis=0)
        sd_val_g = np.std(Xmean_g)
        Xmean_g = Xmean_g * meanSD_g / sd_val_g
        
        processed_l = Xmean_l.reshape(1, -1)
        processed_g = Xmean_g.reshape(1, -1)
        
        suffix = "_mean"
        save_transformed_file(data, filename, processed_l, processed_g, suffix, clip_l_key, clip_g_key)
        
    elif user_input == "3":
        # DIVISION
        print("You chose Option 4 - divide by scalar...")
        
        divisor = float(input("Enter the divisor for the tensor: "))
        
        clip_l_np = clip_l_tensor.cpu().detach().numpy()
        clip_g_np = clip_g_tensor.cpu().detach().numpy()
        
        processed_l = clip_l_np / divisor
        processed_g = clip_g_np / divisor
        
        print(f'TI divided successfully by {divisor}')
        
        suffix = f"_div{divisor}"
        save_transformed_file(data, filename, processed_l, processed_g, suffix, clip_l_key, clip_g_key)
    
    return True


def main():
    """Main function"""
    print("="*60)
    print("TI CHANGER SDXL - TEXTUAL INVERSION FILE PROCESSOR")
    print("="*60)
    print("This tool manipulates SDXL textual inversion files (.safetensors)")
    print("SDXL uses dual embeddings: CLIP-L (768) and CLIP-G (1280)")
    print("="*60)
    
    while True:
        success = process_single_file()
        if success:
            print("\n✅ Processing completed!")
        else:
            print("\n❌ Processing failed or cancelled")
        
        cont = input("\nProcess another file? (Y/N): ").strip().upper()
        if cont not in ['Y', 'YES']:
            return


if __name__ == "__main__":
    main()
