"""
TI Changer - Convert and manipulate textual inversion files
"""

# Load a .pt textual inversion file and show / manipulate it
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

# Configure matplotlib for better display (standalone Python version)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

# ========================================
# VISUALIZATION CONFIGURATION
# ========================================
# Using standard matplotlib divergent colormap for heatmap visualizations
# Available options: 'RdBu', 'coolwarm', 'seismic', 'RdYlBu', 'bwr'

# Heatmap dimensions (adjust based on your vector size preferences)
HEATMAP_HEIGHT = 24
HEATMAP_WIDTH = 36


def analyze_pt_file(filepath):
    """
    Analyze a .pt file to determine its type and structure with flexible TI detection
    Now supports both single-vector (1D) and multi-vector (2D) files
    """
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"\nFile Analysis: {os.path.basename(filepath)}")
        print("=" * 50)
        
        # Check basic structure
        if isinstance(data, dict):
            print("✓ File is a dictionary-based PyTorch file")
            print(f"Top-level keys: {list(data.keys())}")
            
            # Try to find embedding tensor with flexible key matching
            embedding_info = find_embedding_tensor(data)
            
            if embedding_info:
                tensor_path, tensor = embedding_info
                print(f"✓ Found embedding tensor at: {tensor_path}")
                print(f"✓ Tensor shape: {tensor.shape}")
                
                if len(tensor.shape) == 1:
                    print(f"✓ Single vector file (1D tensor)")
                    print(f"✓ Vector dimensions: {tensor.shape[0]}")
                elif len(tensor.shape) == 2:
                    print(f"✓ Multi-vector file (2D tensor)")
                    print(f"✓ Number of vectors: {tensor.shape[0]}")
                    print(f"✓ Vector dimensions: {tensor.shape[1]}")
                
                return True
            else:
                print("✗ No embedding tensor found with common TI patterns")
                print("This might be a different type of PyTorch file")
                
                # Try to identify what type of file it might be
                if 'state_dict' in data:
                    print("→ Might be a model checkpoint file")
                elif 'model' in data:
                    print("→ Might be a saved model file")
                elif any('weight' in str(k).lower() or 'bias' in str(k).lower() for k in data.keys()):
                    print("→ Might be a neural network weights file")
                else:
                    print("→ Unknown PyTorch file type")
                    
        elif hasattr(data, 'state_dict'):
            print("✓ File contains a model with state_dict")
            print("✗ This is a model file, not a textual inversion")
        elif torch.is_tensor(data):
            print(f"✓ File contains a raw tensor with shape: {data.shape}")
            if len(data.shape) == 1:
                print(f"✓ Single vector file! Vector dimensions: {data.shape[0]}")
                return True
            elif len(data.shape) == 2:
                print(f"✓ Multi-vector file! This could be a direct embedding tensor!")
                print(f"✓ Number of vectors: {data.shape[0]}")
                print(f"✓ Vector dimensions: {data.shape[1]}")
                return True
            else:
                print("✗ Tensor shape doesn't match typical embedding format")
        else:
            print(f"✗ File contains: {type(data)}")
            print("✗ Expected a dictionary structure or tensor for textual inversions")
            
        return False
        
    except Exception as e:
        print(f"✗ Error analyzing file: {e}")
        return False


def find_embedding_tensor(data):
    """
    Flexibly search for embedding tensors in various TI file formats
    Handles both 1D (single vector) and 2D (multi-vector) tensors
    Returns: (path_description, tensor) if found, None if not found
    """
    if not isinstance(data, dict):
        return None
    
    # Common patterns for textual inversion files
    search_patterns = [
        # Standard Automatic1111 format
        (['string_to_param', '*'], "string_to_param['*']"),
        (['string_to_param', 'embedding'], "string_to_param['embedding']"),
        
        # Alternative key names
        (['emb_params', '*'], "emb_params['*']"),
        (['embeddings', '*'], "embeddings['*']"),
        (['embedding', '*'], "embedding['*']"),
        
        # Direct embedding keys
        (['*'], "root level '*'"),
        (['embedding'], "root level 'embedding'"),
        (['tensor'], "root level 'tensor'"),
        (['vectors'], "root level 'vectors'"),
        (['weights'], "root level 'weights'"),
        
        # Token-based patterns
        (['string_to_param'], "string_to_param (checking all keys)"),
        (['embeddings'], "embeddings (checking all keys)"),
        (['emb_params'], "emb_params (checking all keys)"),
    ]
    
    # Try each pattern
    for key_path, description in search_patterns:
        try:
            current = data
            
            # Navigate through the key path
            for key in key_path[:-1]:
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    break
            else:
                # We successfully navigated to the parent, now check the final key
                final_key = key_path[-1]
                
                if final_key in current:
                    tensor_candidate = current[final_key]
                    # Accept both 1D and 2D tensors
                    if torch.is_tensor(tensor_candidate) and len(tensor_candidate.shape) in [1, 2]:
                        return (description, tensor_candidate)
                elif len(key_path) == 1 and final_key in ['string_to_param', 'embeddings', 'emb_params']:
                    # For containers, check all their contents
                    container = current[final_key]
                    if isinstance(container, dict):
                        for sub_key, sub_value in container.items():
                            # Accept both 1D and 2D tensors
                            if torch.is_tensor(sub_value) and len(sub_value.shape) in [1, 2]:
                                return (f"{description}['{sub_key}']", sub_value)
        except:
            continue
    
    # If no standard patterns work, search more broadly
    return search_all_tensors(data)


def search_all_tensors(data, path="", max_depth=3):
    """
    Recursively search for 1D or 2D tensors that could be embeddings
    """
    if max_depth <= 0:
        return None
        
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Accept both 1D and 2D tensors
            if torch.is_tensor(value) and len(value.shape) in [1, 2]:
                return (f"deep search: {current_path}", value)
            elif isinstance(value, dict):
                result = search_all_tensors(value, current_path, max_depth - 1)
                if result:
                    return result
    
    return None


def load_ti_file_flexible(filepath):
    """
    Load a TI file with flexible structure detection
    Handles both single-vector (1D) and multi-vector (2D) tensors
    Returns: (data_dict, tensor, tensor_path, is_single_vector) or None if not a valid TI file
    """
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Handle raw tensor files
        if torch.is_tensor(data):
            if len(data.shape) == 2:
                # Create a minimal TI structure for 2D tensor
                ti_data = {
                    'string_to_param': {'*': data},
                    'string_to_token': {'*': 49408},  # Default token
                    'name': os.path.splitext(os.path.basename(filepath))[0],
                    'step': 0
                }
                return (ti_data, data, "raw tensor converted to standard format", False)
            elif len(data.shape) == 1:
                # Convert 1D tensor to 2D for consistent processing
                data_2d = data.unsqueeze(0)  # Add batch dimension
                ti_data = {
                    'string_to_param': {'*': data_2d},
                    'string_to_token': {'*': 49408},  # Default token
                    'name': os.path.splitext(os.path.basename(filepath))[0],
                    'step': 0
                }
                return (ti_data, data_2d, "raw 1D tensor converted to 2D format", True)
        
        # Handle dictionary files
        if isinstance(data, dict):
            embedding_info = find_embedding_tensor(data)
            if embedding_info:
                tensor_path, tensor = embedding_info
                is_single_vector = False
                
                # Handle 1D tensors (single vector)
                if len(tensor.shape) == 1:
                    is_single_vector = True
                    # Convert to 2D for consistent processing
                    tensor = tensor.unsqueeze(0)
                    print(f"🔄 Converted single vector (1D) to 2D format: {tensor.shape}")
                
                # Ensure we have a complete TI structure
                if 'string_to_param' not in data:
                    # Create missing structure
                    data['string_to_param'] = {'*': tensor}
                else:
                    # Update the tensor in the existing structure (might have been converted from 1D)
                    if isinstance(data['string_to_param'], dict):
                        # Find the key that contains our tensor and update it
                        for key, value in data['string_to_param'].items():
                            if torch.is_tensor(value) and torch.equal(value.squeeze(), tensor.squeeze()):
                                data['string_to_param'][key] = tensor
                                break
                    if '*' not in data['string_to_param']:
                        data['string_to_param']['*'] = tensor
                
                # Add missing metadata if needed
                if 'string_to_token' not in data:
                    data['string_to_token'] = {'*': 49408}
                if 'name' not in data:
                    data['name'] = os.path.splitext(os.path.basename(filepath))[0]
                if 'step' not in data:
                    data['step'] = 0
                
                return (data, tensor, tensor_path, is_single_vector)
        
        return None
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def get_user_choice(is_single_vector=False):
    """
    User input function with enhanced feedback for single-vector files
    """
    import sys
    
    # Clear any potential input buffer issues
    if hasattr(sys.stdin, 'flush'):
        sys.stdin.flush()
    
    print("\n" + "="*60)
    if is_single_vector:
        print("TI CHANGER - SINGLE VECTOR FILE OPERATIONS")
        print("="*60)
        print("🔵 SINGLE VECTOR DETECTED: Some operations limited or modified")
    else:
        print("TI CHANGER - MULTI VECTOR FILE OPERATIONS") 
        print("="*60)
        print("🟢 MULTI VECTOR FILE: All operations available")
    print("="*60)
    
    # Mark operations that work differently for single vectors
    sv_note = " (⚠️  Single vector: creates duplicate)" if is_single_vector else ""
    mv_only = " (❌ Multi-vector only)" if is_single_vector else ""
    
    print(f"1. Apply smoothing to all vectors{sv_note}")
    print(f"2. Create single mean vector (condensed){mv_only}")
    print(f"3. Apply threshold-based decimation (zero small values)")
    print(f"4. Divide all vectors by scalar")
    print(f"5. Roll/shift all vectors")
    print(f"6. Extract individual vectors to separate files{mv_only}")
    print(f"7. Save top N vectors by absolute magnitude{mv_only}")
    print(f"8. Clustering-Based Reduction (K-means with Elbow Method){mv_only}")
    print(f"9. Principal Component Analysis (PCA){mv_only}")
    print(f"10. Quantile Transform (Uniform/Gaussian)")
    print(f"11. Nonlinear Squashing (tanh)")
    print(f"12. L² Normalization")
    print(f"13. Max/Min Averaging (single vector from extremes){mv_only}")
    print(f"14. Average specified vectors and combine with remaining{mv_only}")
    print("15. Compare two TI files (vector-by-vector analysis)")
    print("="*60)
    
    if is_single_vector:
        print("\n📌 NOTE: This is a single-vector file. Some operations that require")
        print("   multiple vectors are unavailable or will work on duplicated vectors.")
        print("   Recommended operations: 3, 4, 5, 10, 11, 12")
    
    # Define which operations are available for single vectors
    single_vector_ops = ['1', '3', '4', '5', '10', '11', '12', '15']
    multi_vector_ops = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    
    valid_ops = single_vector_ops if is_single_vector else multi_vector_ops
    
    # Improved input handling with retry logic
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Add a small delay to ensure console is ready
            import time
            time.sleep(0.1)
            
            print(f"\nAttempt {attempt + 1}: ", end="", flush=True)
            user_input = input("Choose operation (1-15): ").strip()
            
            # Debug: Show what was actually captured (remove this after fixing)
            print(f"Debug: Captured input: '{user_input}' (length: {len(user_input)})")
            
            if user_input in valid_ops:
                return user_input
            elif user_input in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']:
                if is_single_vector:
                    print(f"❌ Operation {user_input} is not available for single-vector files.")
                    print(f"   Available operations: {', '.join(single_vector_ops)}")
                else:
                    return user_input
            else:
                print(f"Invalid input: '{user_input}'. Please enter a number from 1 to 15.")
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
    
    print("Invalid choice after multiple attempts. Please run the script again and enter 1-15.")
    return None


def extract_individual_vectors(data, original_filename, numvectors):
    """
    Extract individual vectors from a multi-vector TI file and save each as a separate .pt file
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
    """
    print(f"\nExtracting {numvectors} individual vectors from '{original_filename}'...")
    
    # Get the original tensor
    original_tensor = data['string_to_param']['*']
    
    # Create base filename (remove .pt extension if present)
    base_filename = original_filename.replace('.pt', '')
    
    # Ask user for confirmation and naming preference
    print(f"\nThis will create {numvectors} separate .pt files:")
    for i in range(numvectors):
        print(f"  {base_filename}_vector_{i+1:02d}.pt")
    
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
            # Extract the i-th vector (keep as 2D with shape [1, 768])
            individual_vector = original_tensor[i:i+1].clone()
            
            # Create new data dictionary for this individual vector
            individual_data = data.copy()
            individual_data['string_to_param'] = {'*': individual_vector}
            
            # Generate filename
            vector_filename = f"{base_filename}_vector_{i+1:02d}.pt"
            filepath = os.path.join(directory, vector_filename)
            
            # Save the individual vector file
            torch.save(individual_data, filepath)
            
            print(f"✓ Saved vector {i+1}/{numvectors}: {vector_filename}")
            successful_extractions += 1
            
        except Exception as e:
            print(f"✗ Error saving vector {i+1}: {e}")
    
    print(f"\nExtraction complete! Successfully created {successful_extractions}/{numvectors} individual vector files.")
    
    if successful_extractions > 0:
        print(f"Files saved in '{directory}' directory.")
        
        # Show summary of created files
        print("\nCreated files:")
        for i in range(successful_extractions):
            vector_filename = f"{base_filename}_vector_{i+1:02d}.pt"
            print(f"  - {vector_filename}")


def select_top_n_vectors(data, original_filename, numvectors, np_array):
    """
    Select and save the top N vectors based on absolute magnitude
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        np_array: NumPy array of the tensor data
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
    
    # Calculate absolute magnitude for each vector
    vector_magnitudes = []
    for i in range(numvectors):
        magnitude = np.linalg.norm(np_array[i])  # L2 norm (magnitude)
        vector_magnitudes.append((i, magnitude))
    
    # Sort by magnitude (descending)
    vector_magnitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N vector indices and remaining indices
    top_indices = [idx for idx, mag in vector_magnitudes[:n]]
    bottom_indices = [idx for idx, mag in vector_magnitudes[n:]]
    top_indices.sort()  # Sort indices for consistent ordering
    bottom_indices.sort()  # Sort indices for consistent ordering
    
    print(f"\nTop {n} vectors by magnitude:")
    for rank, (orig_idx, magnitude) in enumerate(vector_magnitudes[:n], 1):
        print(f"  Rank {rank}: Vector {orig_idx + 1} (magnitude: {magnitude:.6f})")
    
    # Create new tensor with only top N vectors
    top_vectors = np_array[top_indices]
    
    # Convert back to tensor
    top_tensor = torch.tensor(top_vectors, device='cpu', requires_grad=True)
    
    # Create new data dictionary for top vectors
    top_data = data.copy()
    top_data['string_to_param'] = {'*': top_tensor}
    
    print(f"\nTop vectors tensor shape: {top_tensor.shape} (reduced from {np_array.shape})")
    
    # Generate filenames
    base_filename = original_filename.replace('.pt', '')
    top_filename_suffix = f"_TOP{n}.pt"
    top_final_filename = base_filename + top_filename_suffix
    
    # Specify the directory path
    directory = "textual_inversions"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the top vectors file
    top_filepath = os.path.join(directory, top_final_filename)
    torch.save(top_data, top_filepath)
    
    print(f"✅ Top vectors file '{top_final_filename}' has been saved to the '{directory}' directory.")
    
    # Handle bottom vectors (remaining vectors)
    if len(bottom_indices) > 0:
        bottom_vectors = np_array[bottom_indices]
        bottom_tensor = torch.tensor(bottom_vectors, device='cpu', requires_grad=True)
        
        # Create new data dictionary for bottom vectors
        bottom_data = data.copy()
        bottom_data['string_to_param'] = {'*': bottom_tensor}
        
        print(f"Bottom vectors tensor shape: {bottom_tensor.shape}")
        
        # Generate bottom filename
        bottom_count = len(bottom_indices)
        bottom_filename_suffix = f"_BOTTOM{bottom_count}.pt"
        bottom_final_filename = base_filename + bottom_filename_suffix
        
        # Save the bottom vectors file
        bottom_filepath = os.path.join(directory, bottom_final_filename)
        torch.save(bottom_data, bottom_filepath)
        
        print(f"✅ Bottom vectors file '{bottom_final_filename}' has been saved to the '{directory}' directory.")
        
        print(f"\n📊 Summary:")
        print(f"   Original vectors: {numvectors}")
        print(f"   Top {n} vectors saved as: {top_final_filename}")
        print(f"   Bottom {bottom_count} vectors saved as: {bottom_final_filename}")
    else:
        print(f"\n📊 Summary:")
        print(f"   All {numvectors} vectors were selected as top vectors.")
        print(f"   Top {n} vectors saved as: {top_final_filename}")
        print("   No bottom vectors to save.")
    
    print(f"Successfully processed vectors based on absolute magnitude.")


def find_optimal_clusters_elbow_method(np_array, max_clusters=None, show_plots=True):
    """
    Find optimal number of clusters using elbow method and silhouette analysis
    
    Args:
        np_array: NumPy array of the tensor data
        max_clusters: Maximum number of clusters to test (default: min(20, n_vectors-1))
        show_plots: Whether to display plots for analysis
        
    Returns:
        dict: Analysis results including optimal clusters and metrics
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
        
        # Calculate rate of change (derivative)
        derivatives = []
        for i in range(1, len(values) - 1):
            derivative = values[i-1] - 2*values[i] + values[i+1]
            derivatives.append(derivative)
        
        # Find point with maximum rate of change
        max_derivative_idx = np.argmax(derivatives)
        return max_derivative_idx + 1  # +1 because derivatives start from index 1
    
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
        plt.xticks(cluster_range)  # Only show ticks for actual cluster values
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
        plt.xticks(cluster_range)  # Only show ticks for actual cluster values
        plt.legend()
        
        # Plot 3: Combined Analysis
        plt.subplot(1, 3, 3)
        # Normalize both metrics for comparison
        norm_inertias = [(max(inertias) - x) / (max(inertias) - min(inertias)) for x in inertias]
        norm_silhouettes = [(x - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) for x in silhouette_scores]
        
        plt.plot(cluster_range, norm_inertias, 'b-', label='Normalized Inertia (inverted)', linewidth=2)
        plt.plot(cluster_range, norm_silhouettes, 'g-', label='Normalized Silhouette', linewidth=2)
        plt.axvline(x=optimal_clusters_elbow, color='blue', linestyle='--', alpha=0.7)
        plt.axvline(x=optimal_clusters_silhouette, color='green', linestyle='--', alpha=0.7)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Normalized Score')
        plt.title('Combined Analysis')
        plt.xticks(cluster_range)  # Only show ticks for actual cluster values
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


def clustering_based_reduction(data, original_filename, numvectors, np_array):
    """
    Apply K-means clustering to reduce vectors to N cluster centroids with elbow method analysis
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        np_array: NumPy array of the tensor data
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
    
    # Perform elbow method analysis
    print("\n" + "="*60)
    print("🔍 OPTIMAL CLUSTER ANALYSIS")
    print("="*60)
    
    analysis = find_optimal_clusters_elbow_method(np_array, max_clusters=numvectors-1, show_plots=True)
    
    if analysis is None:
        print("❌ Could not perform cluster analysis")
        return
    
    # Display analysis results
    print(f"\n📊 CLUSTERING ANALYSIS RESULTS:")
    print(f"   Optimal clusters (Elbow method): {analysis['optimal_elbow']}")
    print(f"   Optimal clusters (Silhouette): {analysis['optimal_silhouette']}")
    print(f"   Best silhouette score: {analysis['best_silhouette_score']:.3f}")
    
    # Create detailed statistics table
    print(f"\n📋 DETAILED CLUSTER STATISTICS:")
    print(f"{'Clusters':<10} {'Inertia':<12} {'Silhouette':<12} {'Recommendation':<15}")
    print("-" * 55)
    
    for i, k in enumerate(analysis['cluster_range']):
        inertia = analysis['inertias'][i]
        silhouette = analysis['silhouette_scores'][i]
        
        recommendation = ""
        if k == analysis['optimal_elbow']:
            recommendation += "ELBOW "
        if k == analysis['optimal_silhouette']:
            recommendation += "BEST_SIL"
        
        print(f"{k:<10} {inertia:<12.2f} {silhouette:<12.3f} {recommendation:<15}")
    
    # Recommend optimal choice
    if analysis['optimal_elbow'] == analysis['optimal_silhouette']:
        recommended = analysis['optimal_elbow']
        print(f"\n🎯 STRONG RECOMMENDATION: {recommended} clusters")
        print(f"   Both elbow method and silhouette analysis agree on {recommended} clusters")
    else:
        print(f"\n🤔 MIXED SIGNALS:")
        print(f"   Elbow method suggests: {analysis['optimal_elbow']} clusters")
        print(f"   Silhouette analysis suggests: {analysis['optimal_silhouette']} clusters")
        
        # Use silhouette as tiebreaker for better cluster quality
        recommended = analysis['optimal_silhouette']
        print(f"   RECOMMENDED: {recommended} clusters (better silhouette score)")
    
    print(f"\n" + "="*60)
    print("🎯 CLUSTER SELECTION")
    print("="*60)
    print(f"Recommended number of clusters: {recommended}")
    print(f"\nOptions:")
    print(f"1. Accept recommendation ({recommended} clusters)")
    print(f"2. Use elbow method suggestion ({analysis['optimal_elbow']} clusters)")
    print(f"3. Enter custom number of clusters")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                n_clusters = recommended
                print(f"✓ Using recommended {n_clusters} clusters")
                break
            elif choice == "2":
                n_clusters = analysis['optimal_elbow']
                print(f"✓ Using elbow method suggestion: {n_clusters} clusters")
                break
            elif choice == "3":
                while True:
                    try:
                        n_clusters = int(input(f"Enter number of clusters (1-{numvectors}): "))
                        if 1 <= n_clusters <= numvectors:
                            print(f"✓ Using custom selection: {n_clusters} clusters")
                            break
                        else:
                            print(f"Please enter a number between 1 and {numvectors}")
                    except ValueError:
                        print("Please enter a valid integer")
                break
            else:
                print("Please enter 1, 2, or 3")
        except (ValueError, KeyboardInterrupt):
            print("Please enter a valid choice")
    
    print(f"\n🚀 Applying K-means clustering with {n_clusters} clusters...")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(np_array)
    centroids = kmeans.cluster_centers_
    
    # Calculate final metrics for the chosen number of clusters
    final_inertia = kmeans.inertia_
    if n_clusters > 1:
        final_silhouette = silhouette_score(np_array, cluster_labels)
    else:
        final_silhouette = 0  # Silhouette score undefined for 1 cluster
    
    print(f"✓ Clustering complete!")
    print(f"✓ Original vectors: {numvectors}")
    print(f"✓ Reduced to centroids: {n_clusters}")
    print(f"✓ Final inertia: {final_inertia:.2f}")
    if n_clusters > 1:
        print(f"✓ Final silhouette score: {final_silhouette:.3f}")
    print(f"✓ Centroid tensor shape: {centroids.shape}")
    
    # Show cluster information
    print(f"\n📊 FINAL CLUSTER ASSIGNMENTS:")
    for cluster_id in range(n_clusters):
        vectors_in_cluster = np.where(cluster_labels == cluster_id)[0]
        print(f"  Cluster {cluster_id + 1}: {len(vectors_in_cluster)} vectors (indices: {vectors_in_cluster.tolist()})")
    
    # Create base filename
    base_filename = original_filename.replace('.pt', '')
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save individual centroid files
    print(f"\n💾 Saving individual centroid files...")
    for i in range(n_clusters):
        # Extract the i-th centroid (keep as 2D with shape [1, dimensions])
        individual_centroid = centroids[i:i+1]
        
        # Convert to tensor
        centroid_tensor = torch.tensor(individual_centroid, device='cpu', requires_grad=True)
        
        # Create new data dictionary for this individual centroid
        individual_data = data.copy()
        individual_data['string_to_param'] = {'*': centroid_tensor}
        
        # Generate filename
        centroid_filename = f"{base_filename}_kmeans_centroid_{i+1:02d}.pt"
        filepath = os.path.join(directory, centroid_filename)
        
        # Save the individual centroid file
        torch.save(individual_data, filepath)
        print(f"✓ Saved centroid {i+1}/{n_clusters}: {centroid_filename}")
    
    # Save combined centroids file
    print(f"\n💾 Saving combined centroids file...")
    combined_tensor = torch.tensor(centroids, device='cpu', requires_grad=True)
    combined_data = data.copy()
    combined_data['string_to_param'] = {'*': combined_tensor}
    
    combined_filename = f"{base_filename}_kmeans_{n_clusters}centroids.pt"
    combined_filepath = os.path.join(directory, combined_filename)
    torch.save(combined_data, combined_filepath)
    
    print(f"✅ Combined centroids file '{combined_filename}' saved to '{directory}' directory.")
    
    print(f"\n📊 K-MEANS CLUSTERING SUMMARY:")
    print(f"   Original vectors: {numvectors}")
    print(f"   Clusters created: {n_clusters}")
    print(f"   Compression ratio: {numvectors/n_clusters:.2f}:1")
    print(f"   Final inertia: {final_inertia:.2f}")
    if n_clusters > 1:
        print(f"   Final silhouette score: {final_silhouette:.3f}")
    print(f"   Individual centroids: {n_clusters} files")
    print(f"   Combined file: {combined_filename}")
    print(f"   Dimensionality preserved: {centroids.shape[1]}")
    
    # Save analysis report
    analysis_filename = f"{base_filename}_kmeans_analysis.txt"
    analysis_filepath = os.path.join(directory, analysis_filename)
    
    with open(analysis_filepath, 'w') as f:
        f.write(f"K-means Clustering Analysis Report\n")
        f.write(f"=================================\n\n")
        f.write(f"Original file: {original_filename}\n")
        f.write(f"Original vectors: {numvectors}\n")
        f.write(f"Selected clusters: {n_clusters}\n")
        f.write(f"Compression ratio: {numvectors/n_clusters:.2f}:1\n\n")
        
        f.write(f"Analysis Results:\n")
        f.write(f"Elbow method optimal: {analysis['optimal_elbow']}\n")
        f.write(f"Silhouette optimal: {analysis['optimal_silhouette']}\n")
        f.write(f"Best silhouette score: {analysis['best_silhouette_score']:.3f}\n\n")
        
        f.write(f"Final Metrics:\n")
        f.write(f"Final inertia: {final_inertia:.2f}\n")
        if n_clusters > 1:
            f.write(f"Final silhouette: {final_silhouette:.3f}\n")
        
        f.write(f"\nDetailed Statistics:\n")
        f.write(f"{'Clusters':<10} {'Inertia':<12} {'Silhouette':<12}\n")
        f.write("-" * 40 + "\n")
        for i, k in enumerate(analysis['cluster_range']):
            f.write(f"{k:<10} {analysis['inertias'][i]:<12.2f} {analysis['silhouette_scores'][i]:<12.3f}\n")
    
    print(f"📋 Analysis report saved: {analysis_filename}")


def analyze_pca_components(np_array, show_plots=True):
    """
    Analyze PCA to recommend optimal number of components
    
    Args:
        np_array: NumPy array of the tensor data
        show_plots: Whether to display analysis plots
        
    Returns:
        dict: Analysis results including recommendations and metrics
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
    
    # Fit PCA with all components to get full variance analysis
    pca_full = PCA()
    pca_full.fit(np_array)
    
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find components that capture different variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    variance_recommendations = {}
    
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        if n_components < len(cumulative_variance):  # Valid component count
            variance_recommendations[f"{threshold*100:.0f}%"] = n_components
    
    # Method 1: Elbow method (largest drop in variance)
    if len(explained_variance_ratio) > 2:
        variance_drops = np.diff(explained_variance_ratio)
        elbow_point = np.argmax(np.abs(variance_drops)) + 1
    else:
        elbow_point = 1
    
    # Method 2: Kaiser criterion (eigenvalues > 1/n_features threshold)
    # For normalized data, keep components with variance > average
    kaiser_threshold = 1.0 / len(explained_variance_ratio)
    kaiser_components = np.sum(explained_variance_ratio > kaiser_threshold)
    
    # Method 3: Significant components (variance > 1% and cumulative < 99%)
    significant_components = 0
    for i, var_ratio in enumerate(explained_variance_ratio):
        if var_ratio > 0.01 and cumulative_variance[i] < 0.99:
            significant_components = i + 1
        else:
            break
    significant_components = max(1, significant_components)
    
    # Calculate reconstruction errors for different numbers of components
    reconstruction_errors = []
    test_range = range(1, min(n_vectors, 15) + 1)
    
    for n_comp in test_range:
        pca_test = PCA(n_components=n_comp)
        transformed = pca_test.fit_transform(np_array)
        reconstructed = pca_test.inverse_transform(transformed)
        error = np.mean((np_array - reconstructed) ** 2)
        reconstruction_errors.append(error)
    
    # Create analysis plots if requested
    if show_plots:
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Scree plot
        plt.subplot(2, 3, 1)
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', markersize=6)
        plt.axvline(x=float(elbow_point), color='red', linestyle='--', alpha=0.7, label=f'Elbow at {elbow_point}')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Cumulative variance plot
        plt.subplot(2, 3, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', markersize=6)
        for threshold in [0.90, 0.95]:
            plt.axhline(y=threshold, color='g', linestyle='--', alpha=0.7, label=f'{threshold*100:.0f}% threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Reconstruction error
        plt.subplot(2, 3, 3)
        plt.plot(test_range, reconstruction_errors, 'go-', markersize=6)
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error vs Components')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 4: Component importance bar chart
        plt.subplot(2, 3, 4)
        top_components = min(15, len(explained_variance_ratio))
        plt.bar(range(1, top_components + 1), explained_variance_ratio[:top_components], alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f'Top {top_components} Component Importance')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Variance analysis breakdown
        plt.subplot(2, 3, 5)
        categories = ['First\nComponent', 'First 2\nComponents', 'First 5\nComponents', '90%\nThreshold', '95%\nThreshold']
        percentages = [
            cumulative_variance[0] * 100,
            cumulative_variance[1] * 100 if len(cumulative_variance) > 1 else 0,
            cumulative_variance[4] * 100 if len(cumulative_variance) > 4 else cumulative_variance[-1] * 100,
            variance_recommendations.get('90%', 0) and cumulative_variance[variance_recommendations['90%'] - 1] * 100,
            variance_recommendations.get('95%', 0) and cumulative_variance[variance_recommendations['95%'] - 1] * 100
        ]
        
        bars = plt.bar(categories, percentages, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
        plt.ylabel('Cumulative Variance Explained (%)')
        plt.title('Variance Capture Analysis')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: Recommendation comparison
        plt.subplot(2, 3, 6)
        rec_methods = ['Elbow', 'Kaiser', 'Significant', '90%', '95%']
        rec_values = [
            elbow_point,
            kaiser_components,
            significant_components,
            variance_recommendations.get('90%', 0),
            variance_recommendations.get('95%', 0)
        ]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        bars = plt.bar(rec_methods, rec_values, alpha=0.7, color=colors)
        plt.ylabel('Recommended Components')
        plt.title('Method Comparison')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, rec_values):
            if val > 0:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{val}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    # Prepare results
    results = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'variance_recommendations': variance_recommendations,
        'elbow_point': elbow_point,
        'kaiser_components': kaiser_components,
        'significant_components': significant_components,
        'reconstruction_errors': reconstruction_errors,
        'test_range': list(test_range)
    }
    
    return results


def principal_component_analysis(data, original_filename, numvectors, np_array):
    """
    Apply PCA to reduce the number of vectors to N principal components with intelligent analysis
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        np_array: NumPy array of the tensor data
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("❌ Error: scikit-learn is required for Principal Component Analysis.")
        print("Please install it with: pip install scikit-learn")
        return
    
    print(f"\nApplying Principal Component Analysis (PCA) to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    if numvectors < 2:
        print("❌ Error: Need at least 2 vectors for PCA analysis")
        return
    
    # Perform comprehensive PCA analysis
    print("\n" + "="*60)
    print("🔍 PCA COMPONENT ANALYSIS")
    print("="*60)
    
    analysis = analyze_pca_components(np_array, show_plots=True)
    
    if analysis is None:
        print("❌ Could not perform PCA analysis")
        return
    
    # Display analysis results
    print(f"\n📊 PCA ANALYSIS RESULTS:")
    print(f"   Total possible components: {numvectors}")
    print(f"   Vector dimensions: {np_array.shape[1]}")
    
    # Show detailed component breakdown
    print(f"\n📋 COMPONENT BREAKDOWN (Top 10):")
    print(f"{'Component':<10} {'Variance %':<12} {'Cumulative %':<15} {'Significance':<15}")
    print("-" * 60)
    
    for i in range(min(10, len(analysis['explained_variance_ratio']))):
        var_pct = analysis['explained_variance_ratio'][i] * 100
        cum_pct = analysis['cumulative_variance'][i] * 100
        
        significance = ""
        if i == 0:
            significance = "PRIMARY"
        elif var_pct > 10:
            significance = "MAJOR"
        elif var_pct > 5:
            significance = "MODERATE"
        elif var_pct > 1:
            significance = "MINOR"
        else:
            significance = "MINIMAL"
        
        print(f"{i+1:<10} {var_pct:<12.2f} {cum_pct:<15.2f} {significance:<15}")
    
    # Show recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   Elbow method: {analysis['elbow_point']} components")
    print(f"   Kaiser criterion: {analysis['kaiser_components']} components") 
    print(f"   Significant components: {analysis['significant_components']} components")
    
    for threshold, n_comp in analysis['variance_recommendations'].items():
        var_captured = analysis['cumulative_variance'][n_comp-1] * 100
        print(f"   {threshold} variance: {n_comp} components (captures {var_captured:.1f}%)")
    
    # Determine best recommendation
    recs = [analysis['elbow_point'], analysis['significant_components']]
    if '90%' in analysis['variance_recommendations']:
        recs.append(analysis['variance_recommendations']['90%'])
    if '95%' in analysis['variance_recommendations']:
        recs.append(analysis['variance_recommendations']['95%'])
    
    # Use most common recommendation, or median if tied
    from collections import Counter
    rec_counts = Counter(recs)
    if len(rec_counts) == 1:
        recommended = list(rec_counts.keys())[0]
    else:
        recommended = analysis['variance_recommendations'].get('90%', analysis['significant_components'])
    
    print(f"\n🌟 BEST RECOMMENDATION: {recommended} components")
    if recommended <= len(analysis['cumulative_variance']):
        var_captured = analysis['cumulative_variance'][recommended-1] * 100
        print(f"   Captures {var_captured:.1f}% of variance")
    
    print(f"\n" + "="*60)
    print("🎯 COMPONENT SELECTION")
    print("="*60)
    print(f"Recommended number of components: {recommended}")
    print(f"\nOptions:")
    print(f"1. Accept best recommendation ({recommended} components)")
    print(f"2. Use 90% variance threshold ({analysis['variance_recommendations'].get('90%', 'N/A')} components)")
    print(f"3. Use 95% variance threshold ({analysis['variance_recommendations'].get('95%', 'N/A')} components)")
    print(f"4. Use elbow method ({analysis['elbow_point']} components)")
    print(f"5. Enter custom number of components")
    print(f"6. Show detailed analysis again")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                n_components = recommended
                print(f"✓ Using best recommendation: {n_components} components")
                break
            elif choice == "2":
                if '90%' in analysis['variance_recommendations']:
                    n_components = analysis['variance_recommendations']['90%']
                    print(f"✓ Using 90% variance threshold: {n_components} components")
                    break
                else:
                    print("90% variance threshold not available")
            elif choice == "3":
                if '95%' in analysis['variance_recommendations']:
                    n_components = analysis['variance_recommendations']['95%']
                    print(f"✓ Using 95% variance threshold: {n_components} components")
                    break
                else:
                    print("95% variance threshold not available")
            elif choice == "4":
                n_components = analysis['elbow_point']
                print(f"✓ Using elbow method: {n_components} components")
                break
            elif choice == "5":
                while True:
                    try:
                        n_components = int(input(f"Enter number of components (1-{numvectors}): "))
                        if 1 <= n_components <= numvectors:
                            print(f"✓ Using custom selection: {n_components} components")
                            break
                        else:
                            print(f"Please enter a number between 1 and {numvectors}")
                    except ValueError:
                        print("Please enter a valid integer")
                break
            elif choice == "6":
                # Re-show analysis
                analyze_pca_components(np_array, show_plots=True)
                continue
            else:
                print("Please enter 1-6")
        except (ValueError, KeyboardInterrupt):
            print("Please enter a valid choice")
    
    print(f"\n🚀 Applying PCA with {n_components} components...")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_vectors = pca.fit_transform(np_array)
    
    # Get the principal components (these are the new vectors)
    # Note: pca.components_ gives us the components, but we want the transformed data
    # The transformed data is in the original vector space dimension
    # We need to transform back to get vectors in the original space
    reconstructed_vectors = pca.inverse_transform(pca_vectors)
    
    # Calculate final metrics
    total_variance_explained = np.sum(pca.explained_variance_ratio_)
    reconstruction_error = np.mean((np_array - reconstructed_vectors) ** 2)
    compression_ratio = numvectors / n_components
    
    print(f"✓ PCA complete!")
    print(f"✓ Original vectors: {numvectors}")
    print(f"✓ Principal components: {n_components}")
    print(f"✓ Total explained variance: {total_variance_explained:.4f} ({total_variance_explained*100:.2f}%)")
    print(f"✓ Reconstruction error: {reconstruction_error:.6f}")
    print(f"✓ Compression ratio: {compression_ratio:.2f}:1")
    print(f"✓ PCA vectors shape: {reconstructed_vectors.shape}")
    
    # Show individual component information
    print(f"\n📋 INDIVIDUAL COMPONENT ANALYSIS:")
    print(f"{'Component':<10} {'Variance %':<12} {'Cumulative %':<15} {'Importance':<15}")
    print("-" * 55)
    
    cumulative_var = 0
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        cumulative_var += var_ratio
        var_pct = var_ratio * 100
        cum_pct = cumulative_var * 100
        
        if var_pct > 20:
            importance = "CRITICAL"
        elif var_pct > 10:
            importance = "MAJOR"
        elif var_pct > 5:
            importance = "SIGNIFICANT"
        elif var_pct > 1:
            importance = "MINOR"
        else:
            importance = "MINIMAL"
        
        print(f"{i+1:<10} {var_pct:<12.2f} {cum_pct:<15.2f} {importance:<15}")
    
    # Create base filename
    base_filename = original_filename.replace('.pt', '')
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save individual PCA component files
    print(f"\n💾 Saving individual PCA component files...")
    for i in range(n_components):
        # Extract the i-th component (keep as 2D with shape [1, dimensions])
        individual_component = reconstructed_vectors[i:i+1]
        
        # Convert to tensor
        component_tensor = torch.tensor(individual_component, device='cpu', requires_grad=True)
        
        # Create new data dictionary for this individual component
        individual_data = data.copy()
        individual_data['string_to_param'] = {'*': component_tensor}
        
        # Generate filename with variance info
        variance_pct = pca.explained_variance_ratio_[i] * 100
        component_filename = f"{base_filename}_pca_comp{i+1:02d}_var{variance_pct:.1f}pct.pt"
        filepath = os.path.join(directory, component_filename)
        
        # Save the individual component file
        torch.save(individual_data, filepath)
        print(f"✓ Saved component {i+1}/{n_components}: {component_filename}")
    
    # Save combined PCA components file
    print(f"\n💾 Saving combined PCA components file...")
    combined_tensor = torch.tensor(reconstructed_vectors, device='cpu', requires_grad=True)
    combined_data = data.copy()
    combined_data['string_to_param'] = {'*': combined_tensor}
    
    combined_filename = f"{base_filename}_pca_{n_components}comp_var{total_variance_explained*100:.1f}pct.pt"
    combined_filepath = os.path.join(directory, combined_filename)
    torch.save(combined_data, combined_filepath)
    
    print(f"✅ Combined PCA file '{combined_filename}' saved to '{directory}' directory.")
    
    # Save detailed analysis report
    analysis_filename = f"{base_filename}_pca_analysis.txt"
    analysis_filepath = os.path.join(directory, analysis_filename)
    
    with open(analysis_filepath, 'w') as f:
        f.write(f"Principal Component Analysis Report\n")
        f.write(f"==================================\n\n")
        f.write(f"Original file: {original_filename}\n")
        f.write(f"Original vectors: {numvectors}\n")
        f.write(f"Selected components: {n_components}\n")
        f.write(f"Compression ratio: {compression_ratio:.2f}:1\n")
        f.write(f"Total variance explained: {total_variance_explained:.4f} ({total_variance_explained*100:.2f}%)\n")
        f.write(f"Reconstruction error: {reconstruction_error:.6f}\n\n")
        
        f.write(f"Component Analysis:\n")
        f.write(f"{'Component':<10} {'Variance':<12} {'Cumulative':<12} {'Eigenvalue':<12}\n")
        f.write("-" * 50 + "\n")
        
        cumulative_var = 0
        for i, (var_ratio, eigenval) in enumerate(zip(pca.explained_variance_ratio_, pca.explained_variance_)):
            cumulative_var += var_ratio
            f.write(f"{i+1:<10} {var_ratio:<12.6f} {cumulative_var:<12.6f} {eigenval:<12.6f}\n")
        
        f.write(f"\nRecommendation Analysis:\n")
        f.write(f"Elbow method suggested: {analysis['elbow_point']} components\n")
        f.write(f"Kaiser criterion suggested: {analysis['kaiser_components']} components\n")
        f.write(f"Significant components: {analysis['significant_components']} components\n")
        
        for threshold, n_comp in analysis['variance_recommendations'].items():
            f.write(f"{threshold} variance threshold: {n_comp} components\n")
    
    print(f"📋 Analysis report saved: {analysis_filename}")
    
    print(f"\n📊 PCA SUMMARY:")
    print(f"   Original vectors: {numvectors}")
    print(f"   Principal components: {n_components}")
    print(f"   Variance captured: {total_variance_explained*100:.2f}%")
    print(f"   Information loss: {(1-total_variance_explained)*100:.2f}%")
    print(f"   Compression ratio: {compression_ratio:.2f}:1")
    print(f"   Reconstruction error: {reconstruction_error:.6f}")
    print(f"   Individual components: {n_components} files")
    print(f"   Combined file: {combined_filename}")
    print(f"   Analysis report: {analysis_filename}")
    print(f"   Dimensionality preserved: {reconstructed_vectors.shape[1]}")
    
    # Show quality assessment
    if total_variance_explained >= 0.95:
        quality = "EXCELLENT"
        color = "🟢"
    elif total_variance_explained >= 0.90:
        quality = "VERY GOOD"
        color = "🟡"
    elif total_variance_explained >= 0.80:
        quality = "GOOD"
        color = "🟠"
    else:
        quality = "FAIR"
        color = "🔴"
    
    print(f"\n{color} Quality Assessment: {quality}")
    print(f"   Data preservation: {total_variance_explained*100:.1f}% of original variance retained")
    
    if total_variance_explained < 0.90:
        print(f"   💡 Consider using more components for better quality")
    if compression_ratio < 2:
        print(f"   💡 Consider using fewer components for better compression")
    
    print(f"✅ PCA analysis and reduction completed successfully!")
    


def show_transformation_plots(original_data, transformed_data, title, filename):
    """
    Show before/after histograms and vector plots for transformations
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Flatten data for histograms
    orig_flat = original_data.flatten()
    trans_flat = transformed_data.flatten()
    
    # Original histogram
    ax1.hist(orig_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Before: Original Data Distribution', fontsize=10)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Transformed histogram
    ax2.hist(trans_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('After: Transformed Data Distribution', fontsize=10)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Original vectors plot (first few vectors)
    num_vectors_to_show = min(5, original_data.shape[0])
    x_dims = np.arange(original_data.shape[1])
    
    for i in range(num_vectors_to_show):
        ax3.plot(x_dims, original_data[i], alpha=0.7, label=f'Vector {i+1}')
    
    ax3.set_title('Before: Original Vector Values', fontsize=10)
    ax3.set_xlabel('Dimension Index')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Transformed vectors plot
    for i in range(num_vectors_to_show):
        ax4.plot(x_dims, transformed_data[i], alpha=0.7, label=f'Vector {i+1}')
    
    ax4.set_title('After: Transformed Vector Values', fontsize=10)
    ax4.set_xlabel('Dimension Index')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle(f'{title} Transformation Analysis\nFile: {filename}', fontsize=12, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    
    # Print statistics
    print(f"\n📊 {title} Statistics:")
    print(f"   Original range: [{orig_flat.min():.6f}, {orig_flat.max():.6f}]")
    print(f"   Original mean: {orig_flat.mean():.6f}")
    print(f"   Original std: {orig_flat.std():.6f}")
    print(f"   Transformed range: [{trans_flat.min():.6f}, {trans_flat.max():.6f}]")
    print(f"   Transformed mean: {trans_flat.mean():.6f}")
    print(f"   Transformed std: {trans_flat.std():.6f}")


def quantile_transform(data, original_filename, numvectors, np_array):
    """
    Apply quantile transformation (uniform or Gaussian) to TI data
    """
    try:
        from scipy import stats
    except ImportError:
        print("❌ Error: scipy is required for quantile transform.")
        print("Please install it with: pip install scipy")
        return
    
    print(f"\nApplying Quantile Transform to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    # Choose distribution type
    print("\nChoose target distribution:")
    print("1. Uniform (0 to 1)")
    print("2. Gaussian (normal/bell curve)")
    
    while True:
        try:
            dist_choice = input("Enter choice (1 or 2): ").strip()
            if dist_choice in ['1', '2']:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter 1 or 2")
    
    # Process each vector independently
    transformed_array = np.zeros_like(np_array)
    
    # Set suffix based on choice
    suffix = "_quantile_uniform" if dist_choice == '1' else "_quantile_gaussian"
    
    for i in range(numvectors):
        vector = np_array[i].copy()
        
        # Sort values and get ranks
        sorted_indices = np.argsort(vector)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(vector))
        
        # Convert ranks to empirical CDF (0 to 1)
        empirical_cdf = (ranks + 1) / (len(vector) + 1)
        
        if dist_choice == '1':
            # Uniform distribution (use CDF directly)
            transformed_vector = empirical_cdf
        else:
            # Gaussian distribution (inverse normal CDF)
            transformed_vector = stats.norm.ppf(empirical_cdf)
        
        # Scale to [-0.5, 0.5] range
        min_val, max_val = transformed_vector.min(), transformed_vector.max()
        if max_val > min_val:  # Avoid division by zero
            transformed_vector = (transformed_vector - min_val) / (max_val - min_val) - 0.5
        
        transformed_array[i] = transformed_vector
    
    # Show visualization
    show_transformation_plots(np_array, transformed_array, 
                            f"Quantile {'Uniform' if dist_choice == '1' else 'Gaussian'}", 
                            original_filename)
    
    # Save the transformed file
    save_transformed_file(data, original_filename, transformed_array, suffix)


def nonlinear_squashing_tanh(data, original_filename, numvectors, np_array):
    """
    Apply tanh nonlinear squashing to TI data
    """
    print(f"\nApplying Nonlinear Squashing (tanh) to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    # Get scale/temperature parameter
    print("\nTanh scaling parameter controls the 'sharpness' of the transformation:")
    print("  - Small scale (0.1-1.0): Even moderate values hit boundaries quickly")
    print("  - Medium scale (1.0-5.0): Balanced transformation")
    print("  - Large scale (5.0-20.0): More linear behavior, gradual curve")
    
    while True:
        try:
            scale = float(input("Enter scale parameter (recommended: 0.5-10.0): "))
            if scale > 0:
                break
            else:
                print("Please enter a positive value")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nApplying tanh transformation with scale {scale}...")
    
    # Apply tanh transformation
    transformed_array = np.tanh(np_array * scale)
    
    # Scale to [-0.5, 0.5] range
    # tanh already outputs [-1, 1], so we just scale by 0.5
    transformed_array = transformed_array * 0.5
    
    # Show visualization
    show_transformation_plots(np_array, transformed_array, f"Tanh (scale={scale})", original_filename)
    
    # Save the transformed file
    suffix = f"_tanh_scale{scale}"
    save_transformed_file(data, original_filename, transformed_array, suffix)


def l2_normalization(data, original_filename, numvectors, np_array):
    """
    Apply L2 normalization to each vector in TI data
    """
    print(f"\nApplying L² Normalization to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    # Apply L2 normalization to each vector
    transformed_array = np.zeros_like(np_array)
    
    for i in range(numvectors):
        vector = np_array[i].copy()
        
        # Calculate L2 norm (Euclidean length)
        l2_norm = np.linalg.norm(vector)
        
        if l2_norm > 0:  # Avoid division by zero
            # Normalize vector to unit length
            normalized_vector = vector / l2_norm
        else:
            # Handle zero vector case
            normalized_vector = vector
        
        # Scale to [-0.5, 0.5] range
        # Since normalized vector is in approximately [-1, 1] range, scale by 0.5
        min_val, max_val = normalized_vector.min(), normalized_vector.max()
        if max_val > min_val:
            # First normalize to [0, 1], then shift to [-0.5, 0.5]
            scaled_vector = (normalized_vector - min_val) / (max_val - min_val) - 0.5
        else:
            scaled_vector = normalized_vector * 0.5
        
        transformed_array[i] = scaled_vector
        
        # Print L2 norm info for first few vectors
        if i < 3:
            original_norm = np.linalg.norm(vector)
            new_norm = np.linalg.norm(normalized_vector)
            final_norm = np.linalg.norm(scaled_vector)
            print(f"  Vector {i+1}: Original norm={original_norm:.6f}, "
                  f"Normalized norm={new_norm:.6f}, Final norm={final_norm:.6f}")
    
    # Show visualization
    show_transformation_plots(np_array, transformed_array, "L² Normalization", original_filename)
    
    # Save the transformed file
    suffix = "_l2norm"
    save_transformed_file(data, original_filename, transformed_array, suffix)


def max_min_averaging(data, original_filename, numvectors, np_array):
    """
    Create a single vector where each datapoint is either the max (if average is positive) 
    or min (if average is negative) across all vectors at that position.
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        np_array: NumPy array of the tensor data
    """
    print(f"\nApplying Max/Min Averaging to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    print(f"Vector dimensions: {np_array.shape[1]}")
    
    # Get dimensions
    num_dimensions = np_array.shape[1]
    
    # Create output vector
    result_vector = np.zeros(num_dimensions)
    
    # Statistics tracking
    positive_avg_count = 0
    negative_avg_count = 0
    zero_avg_count = 0
    
    print(f"\nProcessing {num_dimensions} datapoints...")
    
    # Process each dimension (datapoint position) across all vectors
    for dim_idx in range(num_dimensions):
        # Extract all values at this dimension across all vectors
        values_at_dimension = np_array[:, dim_idx]
        
        # Calculate average for this dimension
        avg_value = np.mean(values_at_dimension)
        
        if avg_value > 0:
            # Use maximum value if average is positive
            result_vector[dim_idx] = np.max(values_at_dimension)
            positive_avg_count += 1
        elif avg_value < 0:
            # Use minimum value if average is negative
            result_vector[dim_idx] = np.min(values_at_dimension)
            negative_avg_count += 1
        else:
            # Handle exact zero case (rare but possible)
            result_vector[dim_idx] = 0.0
            zero_avg_count += 1
        
        # Print progress for first few and every 100th dimension
        if dim_idx < 5 or dim_idx % 100 == 0:
            print(f"  Dim {dim_idx:3d}: avg={avg_value:8.5f} → {'MAX' if avg_value > 0 else 'MIN' if avg_value < 0 else 'ZERO'} = {result_vector[dim_idx]:8.5f}")
    
    # Create the result as a 2D array (single vector)
    result_array = result_vector.reshape(1, -1)
    
    print(f"\n✓ Max/Min Averaging complete!")
    print(f"✓ Original vectors: {numvectors}")
    print(f"✓ Result: 1 vector with {num_dimensions} dimensions")
    print(f"✓ Statistics:")
    print(f"   Positive averages (used MAX): {positive_avg_count} ({positive_avg_count/num_dimensions*100:.1f}%)")
    print(f"   Negative averages (used MIN): {negative_avg_count} ({negative_avg_count/num_dimensions*100:.1f}%)")
    print(f"   Zero averages: {zero_avg_count} ({zero_avg_count/num_dimensions*100:.1f}%)")
    
    # Show statistics of the result vector
    print(f"\n📊 Result Vector Statistics:")
    print(f"   Range: [{result_vector.min():.6f}, {result_vector.max():.6f}]")
    print(f"   Mean: {result_vector.mean():.6f}")
    print(f"   Std: {result_vector.std():.6f}")
    print(f"   L2 norm: {np.linalg.norm(result_vector):.6f}")
    
    # Show transformation visualization
    # Create a comparison with the mean vector for context
    mean_vector = np.mean(np_array, axis=0).reshape(1, -1)
    
    # Show both mean and max/min result for comparison
    comparison_data = np.vstack([mean_vector, result_array])
    comparison_labels = ["Mean Vector", "Max/Min Result"]
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data histogram (all vectors combined)
    orig_flat = np_array.flatten()
    ax1.hist(orig_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Original: All Vector Values Distribution', fontsize=10)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Result vector histogram
    ax2.hist(result_vector, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('Result: Max/Min Vector Distribution', fontsize=10)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Vector comparison plot
    x_dims = np.arange(num_dimensions)
    ax3.plot(x_dims, mean_vector[0], alpha=0.7, label='Mean Vector', color='blue', linewidth=1)
    ax3.plot(x_dims, result_vector, alpha=0.8, label='Max/Min Vector', color='red', linewidth=1.5)
    ax3.set_title('Comparison: Mean vs Max/Min Vector', fontsize=10)
    ax3.set_xlabel('Dimension Index')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Statistics bar chart
    categories = ['Positive Avg\n(MAX used)', 'Negative Avg\n(MIN used)', 'Zero Avg']
    counts = [positive_avg_count, negative_avg_count, zero_avg_count]
    colors = ['green', 'red', 'gray']
    
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Max/Min Selection Statistics', fontsize=10)
    ax4.set_ylabel('Number of Dimensions')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Max/Min Averaging Analysis\nFile: {original_filename}', fontsize=12, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    
    # Save the result file
    suffix = "_maxmin_avg"
    save_transformed_file(data, original_filename, result_array, suffix)


def average_specified_vectors(data, original_filename, numvectors, np_array):
    """
    Average specified vectors and save both combined file and averaged vector separately
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        np_array: NumPy array of the tensor data
    """
    print(f"\nAveraging specified vectors from '{original_filename}'...")
    print(f"Available vectors: {numvectors} (numbered 1-{numvectors})")
    
    # Get vector indices from user
    while True:
        try:
            user_input = input(f"Enter vector numbers to average (comma-separated, e.g., '2,5,10'): ").strip()
            if not user_input:
                print("Please enter at least one vector number")
                continue
            
            # Parse the input
            vector_indices_str = [x.strip() for x in user_input.split(',')]
            vector_indices = []
            
            for idx_str in vector_indices_str:
                idx = int(idx_str)
                if 1 <= idx <= numvectors:
                    vector_indices.append(idx - 1)  # Convert to 0-based indexing
                else:
                    print(f"Vector {idx} is out of range. Please use numbers 1-{numvectors}")
                    raise ValueError("Invalid vector index")
            
            if len(vector_indices) < 2:
                print("Please specify at least 2 vectors to average")
                continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in vector_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            vector_indices = unique_indices
            
            break
            
        except ValueError:
            print("Please enter valid vector numbers separated by commas")
    
    # Convert back to 1-based for display
    vector_numbers = [idx + 1 for idx in vector_indices]
    print(f"\nVectors to average: {vector_numbers}")
    print(f"Number of vectors to average: {len(vector_indices)}")
    
    # Calculate average of specified vectors
    specified_vectors = np_array[vector_indices]
    averaged_vector = np.mean(specified_vectors, axis=0)
    
    print(f"✓ Calculated average of {len(vector_indices)} vectors")
    
    # Get remaining vector indices (not in the averaging list)
    all_indices = set(range(numvectors))
    remaining_indices = sorted(list(all_indices - set(vector_indices)))
    remaining_numbers = [idx + 1 for idx in remaining_indices]
    
    print(f"Remaining vectors: {remaining_numbers}")
    print(f"Number of remaining vectors: {len(remaining_indices)}")
    
    # Create combined array: remaining vectors + averaged vector
    if remaining_indices:
        remaining_vectors = np_array[remaining_indices]
        combined_array = np.vstack([remaining_vectors, averaged_vector.reshape(1, -1)])
    else:
        # Edge case: all vectors were averaged
        combined_array = averaged_vector.reshape(1, -1)
    
    total_combined_vectors = len(remaining_indices) + 1
    print(f"Combined file will have: {total_combined_vectors} vectors")
    
    # Create base filename
    base_filename = original_filename.replace('.pt', '')
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save combined file (remaining + averaged)
    print(f"\nSaving combined file...")
    combined_tensor = torch.tensor(combined_array, device='cpu', requires_grad=True)
    combined_data = data.copy()
    combined_data['string_to_param'] = {'*': combined_tensor}
    
    # Generate combined filename
    vector_list_str = "_".join(map(str, vector_numbers))
    combined_filename = f"{base_filename}_combined_avg{vector_list_str}.pt"
    combined_filepath = os.path.join(directory, combined_filename)
    torch.save(combined_data, combined_filepath)
    
    print(f"✅ Combined file '{combined_filename}' saved to '{directory}' directory.")
    print(f"   Contains {len(remaining_indices)} original vectors + 1 averaged vector = {total_combined_vectors} total vectors")
    
    # Save averaged vector only
    print(f"\nSaving averaged vector file...")
    averaged_tensor = torch.tensor(averaged_vector.reshape(1, -1), device='cpu', requires_grad=True)
    averaged_data = data.copy()
    averaged_data['string_to_param'] = {'*': averaged_tensor}
    
    # Generate averaged filename
    averaged_filename = f"{base_filename}_avg{vector_list_str}.pt"
    averaged_filepath = os.path.join(directory, averaged_filename)
    torch.save(averaged_data, averaged_filepath)
    
    print(f"✅ Averaged vector file '{averaged_filename}' saved to '{directory}' directory.")
    print(f"   Contains 1 vector (average of vectors {vector_numbers})")
    
    # Show statistics
    print(f"\n📊 Vector Averaging Summary:")
    print(f"   Original file: {numvectors} vectors")
    print(f"   Vectors averaged: {vector_numbers} ({len(vector_indices)} vectors)")
    print(f"   Remaining vectors: {remaining_numbers} ({len(remaining_indices)} vectors)")
    print(f"   Combined file: {combined_filename} ({total_combined_vectors} vectors)")
    print(f"   Averaged vector file: {averaged_filename} (1 vector)")
    
    # Show vector statistics
    print(f"\n📊 Averaged Vector Statistics:")
    print(f"   Range: [{averaged_vector.min():.6f}, {averaged_vector.max():.6f}]")
    print(f"   Mean: {averaged_vector.mean():.6f}")
    print(f"   Std: {averaged_vector.std():.6f}")
    print(f"   L2 norm: {np.linalg.norm(averaged_vector):.6f}")
    
    # Compare with original vectors being averaged
    print(f"\n📊 Individual Vector Statistics (being averaged):")
    for i, orig_idx in enumerate(vector_indices):
        vector = np_array[orig_idx]
        vector_num = orig_idx + 1
        print(f"   Vector {vector_num}: Range=[{vector.min():.6f}, {vector.max():.6f}], "
              f"Mean={vector.mean():.6f}, L2 norm={np.linalg.norm(vector):.6f}")
    
    print(f"Successfully processed vector averaging and combination.")


def save_transformed_file(data, original_filename, transformed_array, suffix):
    """
    Save transformed array as a new TI file
    """
    # Convert back to tensor
    transformed_tensor = torch.tensor(transformed_array, device='cpu', requires_grad=True)
    
    # Create new data dictionary
    new_data = data.copy()
    new_data['string_to_param'] = {'*': transformed_tensor}
    
    # Generate filename
    base_filename = original_filename.replace('.pt', '')
    final_filename = base_filename + suffix + ".pt"
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the file to the directory
    filepath = os.path.join(directory, final_filename)
    torch.save(new_data, filepath)
    
    print(f"✅ Transformed file '{final_filename}' saved to '{directory}' directory.")
    print(f"   Tensor shape: {transformed_tensor.shape}")


def process_single_file():
    """Process a single TI file - main workflow"""
    
    # ========================================
    # 1. LOAD TEXTUAL INVERSION FILE
    # ========================================
    """
    1. load a PyTorch tensor from a TI file located in the 'textual_inversions' directory. The tensor is loaded onto the CPU.
    2. print the keys of the loaded data, which is expected to be a dictionary, then print the key/value pairs.
    3. extract the tensor associated with the key '*' from the dictionary under the 'string_to_param' key.
    4. convert the extracted tensor to a NumPy array and detaches it from the GPU. All subsequent ops are performed on this NumPy array.
    5. retrieve and print the number of vectors and the shape of the tensor.
    """
    
    # Load the dictionary from the .pt file
    # First check if file exists in current directory, then in textual_inversions subdirectory
    
    # Use a file chooser dialog to select the .pt file

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Make the dialog larger and more prominent
    root.attributes('-topmost', True)  # Bring to front
    root.geometry("800x600")  # Set a larger initial size

    print("Please select a .pt file to load...")
    filename = filedialog.askopenfilename(
        title="Select a TI .pt file",
        filetypes=[("PyTorch TI files", "*.pt"), ("All files", "*.*")],
        parent=root
    )

    if not filename:
        print("No file selected. Returning to main menu.")
        return False

    # Get just the filename for later use
    original_selected_path = filename
    filename = os.path.basename(filename)
    
    # First, analyze the file with flexible TI detection
    print("\nAnalyzing selected file...")
    if not analyze_pt_file(original_selected_path):
        print("\n" + "="*50)
        print("ERROR: The selected file does not appear to contain textual inversion data.")
        print("="*50)
        print("\nExpected patterns for TI files:")
        print("- Dictionary with embedding tensors")
        print("- 2D tensors with shape [num_vectors, dimensions]")
        print("- Common key patterns like 'string_to_param', 'embeddings', etc.")
        print("\nTrying flexible loading anyway...")
        
        # Try flexible loading as a last resort
        flexible_result = load_ti_file_flexible(original_selected_path)
        if not flexible_result:
            print("❌ Flexible loading also failed.")
            print("\nPlease verify this is a textual inversion file.")
            input("\nPress Enter to continue...")
            return False
        else:
            print("✅ Flexible loading succeeded! Proceeding...")
    
    # Load the file using flexible approach
    flexible_result = load_ti_file_flexible(original_selected_path)
    if not flexible_result:
        print("❌ Failed to load file even with flexible parsing.")
        input("\nPress Enter to continue...")
        return False
    
    data, tensor, tensor_path, is_single_vector = flexible_result
    print(f'\n✅ TI loaded successfully from {original_selected_path}')
    print(f'📍 Embedding tensor found at: {tensor_path}')
    print(f'📊 Final data structure keys: {list(data.keys())}')
    
    if is_single_vector:
        print(f'🔵 SINGLE VECTOR FILE: Original 1D tensor converted to 2D for processing')
        print(f'🔵 Vector will be duplicated where multiple vectors are needed')
    else:
        print(f'🟢 MULTI VECTOR FILE: {tensor.shape[0]} vectors available')
    
    # Now we can safely access the tensor since we've validated and normalized the structure
    
    # Convert the tensor to a NumPy array. It's always a good idea to detach the tensor from the GPU
    # all operations will be done on the numpy array
    np_array = tensor.cpu().detach().numpy()
    numvectors = np_array.shape[0] # get number of vectors
    print('\nNumber of vectors: ',numvectors,'    Shape of the tensor: ',np_array.shape)
    
    # ========================================
    # 2. VECTOR STATISTICS VISUALIZATION
    # ========================================
    """
    Display statistics for each vector using vertical bars
    """
    
    print("\nGenerating vector statistics visualization...")
    
    # Calculate statistics for each vector
    vector_numbers = list(range(1, numvectors + 1))
    min_values = [np.min(np_array[i]) for i in range(numvectors)]
    max_values = [np.max(np_array[i]) for i in range(numvectors)]
      # Create the plot with wider figure for bars and better spacing (smaller size)
    bar_width = 0.7  # Width of each bar (wider since we only have 2 bars)
    fig, ax = plt.subplots(figsize=(max(8, numvectors * 0.5), 5))  # Smaller figure
    
    # Calculate x positions - both bars at the same x position for perfect alignment
    x_pos = np.arange(len(vector_numbers))
    
    # Create the bars at the same x positions
    bars1 = ax.bar(x_pos, min_values, bar_width, label='Min', color='blue', alpha=0.6)
    bars2 = ax.bar(x_pos, max_values, bar_width, label='Max', color='red', alpha=0.6)
    
    # Calculate symmetric y-axis limits
    # Find the maximum absolute value across min and max values
    all_values = min_values + max_values
    max_abs_value = max(abs(min(all_values)), abs(max(all_values)))
    
    # Add 10% padding and ensure we have a reasonable minimum range
    y_limit = max(max_abs_value * 1.1, 0.1)
    
    # Set symmetric y-axis limits
    ax.set_ylim(-y_limit, y_limit)
    
    # Customize the plot with better spacing
    ax.set_title(f' Vector Statistics for {filename}\n({numvectors} vectors, {np_array.shape[1]} dimensions each)', 
                fontsize=10, pad=15)
    ax.set_xlabel('Vector Number', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    
    # Set x-axis labels and positions
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in vector_numbers])
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Improve layout with more white space
    plt.tight_layout(pad=2.0)
    
    # Show the first plot
    plt.show()
    
    
    # ========================================
    # 2D HEATMAP GRID VISUALIZATION
    # ========================================
    """
    Display each vector as a 2D heatmap reshaped to 36x24 for better visualization
    """
    
    print("\nGenerating 2D heatmap grid visualization...")
    
    # Calculate grid dimensions for subplots
    import math
    grid_cols = min(4, numvectors)  # Max 4 columns
    grid_rows = math.ceil(numvectors / grid_cols)
    
    # Create figure for heatmap grid
    fig_size_factor = 3  # Size factor for each subplot
    fig3, axes = plt.subplots(grid_rows, grid_cols, 
                             figsize=(grid_cols * fig_size_factor, grid_rows * fig_size_factor))
    
    # Handle case where we only have one subplot
    if numvectors == 1:
        axes = [axes]
    elif grid_rows == 1:
        axes = [axes] if numvectors == 1 else axes
    else:
        axes = axes.flatten()
    
    # Define heatmap dimensions using configuration
    heatmap_height = HEATMAP_HEIGHT
    heatmap_width = HEATMAP_WIDTH
    target_size = heatmap_height * heatmap_width
    
    # Process each vector
    for i in range(numvectors):
        vector_data = np_array[i].copy()
        
        # Pad or truncate to target size
        if len(vector_data) < target_size:
            # Pad with zeros if vector is smaller
            padded_data = np.zeros(target_size)
            padded_data[:len(vector_data)] = vector_data
            vector_data = padded_data
        elif len(vector_data) > target_size:
            # Truncate if vector is larger
            vector_data = vector_data[:target_size]
        
        # Reshape to 2D heatmap
        heatmap_data = vector_data.reshape(heatmap_height, heatmap_width)
        
        # Use standard matplotlib divergent colormap with bilinear smoothing
        
        # Create heatmap with bilinear interpolation for smooth visualization
        ax = axes[i]
        im = ax.imshow(heatmap_data, cmap='RdBu', aspect='auto', 
                      interpolation='bilinear')  # Smooth bilinear interpolation
        
        # Customize each subplot
        ax.set_title(f'V {i+1}', fontsize=10, pad=5)
        #ax.set_xlabel(f'Dim Width (0-{heatmap_width-1})', fontsize=8)
        #ax.set_ylabel(f'Dim Height (0-{heatmap_height-1})', fontsize=8)
        
        # Reduce tick density for cleaner look
        ax.set_xticks(np.linspace(0, heatmap_width-1, 5).astype(int))
        ax.set_yticks(np.linspace(0, heatmap_height-1, 5).astype(int))
        ax.tick_params(labelsize=7)
        
        # Add colorbar for the first subplot as reference
        if i == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Value', fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    
    # Hide empty subplots if we have fewer vectors than grid spaces
    for i in range(numvectors, len(axes)):
        axes[i].set_visible(False)
    
    # Set overall title
    fig3.suptitle(f' Vector Heatmaps for {filename}\n({numvectors} vectors reshaped for visualization)', 
                 fontsize=12, y=0.98)
    
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    #-----------
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Overall Min: {min(min_values):.6f}")
    print(f"   Overall Max: {max(max_values):.6f}")
    print(f"   Range: {max(max_values) - min(min_values):.6f}")
    print(f"   Y-axis range: ±{y_limit:.6f}")
    print(f"   Heatmap grid: {grid_rows}×{grid_cols} layout")
    print(f"   Each heatmap: {heatmap_height}×{heatmap_width} dimensions")
    
    # ========================================
    # 3. MENU SELECTION
    # ========================================
    """
    Present user with operation choices upfront
    """
    
    user_input = get_user_choice(is_single_vector)
    if user_input is None:
        input("Press Enter to continue...")
        return False
    
    print(f"\nSelected: Option {user_input}")
    
    # Handle single-vector restrictions
    if is_single_vector and user_input in ['2', '6', '7', '8', '9', '13', '14']:
        print(f"❌ Option {user_input} is not available for single-vector files.")
        print("   Please select a different operation.")
        input("Press Enter to continue...")
        return False
    
    # Handle Option 6 (extract vectors) immediately since it doesn't need processing
    if user_input == "6":
        print("You chose Option 6 - extract individual vectors to separate files...")
        extract_individual_vectors(data, filename, numvectors)
        return True  # Successfully completed processing
    
    # Handle Option 7 (top N vectors) immediately since it doesn't need processing
    if user_input == "7":
        print("You chose Option 7 - save top N vectors by absolute magnitude...")
        select_top_n_vectors(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 8 (clustering-based reduction) immediately since it doesn't need processing
    if user_input == "8":
        print("You chose Option 8 - clustering-based reduction (K-means)...")
        clustering_based_reduction(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 9 (PCA) immediately since it doesn't need processing
    if user_input == "9":
        print("You chose Option 9 - Principal Component Analysis (PCA)...")
        principal_component_analysis(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 10 (Quantile Transform) immediately since it doesn't need processing
    if user_input == "10":
        print("You chose Option 10 - Quantile Transform (Uniform/Gaussian)...")
        quantile_transform(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 11 (Tanh Squashing) immediately since it doesn't need processing
    if user_input == "11":
        print("You chose Option 11 - Nonlinear Squashing (tanh)...")
        nonlinear_squashing_tanh(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 12 (L2 Normalization) immediately since it doesn't need processing
    if user_input == "12":
        print("You chose Option 12 - L² Normalization...")
        l2_normalization(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 13 (Max/Min Averaging) immediately since it doesn't need processing
    if user_input == "13":
        print("You chose Option 13 - Max/Min Averaging (single vector from extremes)...")
        max_min_averaging(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 14 (Average Specified Vectors) immediately since it doesn't need processing
    if user_input == "14":
        print("You chose Option 14 - Average specified vectors and combine with remaining...")
        average_specified_vectors(data, filename, numvectors, np_array)
        return True  # Successfully completed processing
    
    # Handle Option 15 (Compare two TI files) immediately since it doesn't need the current file
    if user_input == "15":
        print("You chose Option 15 - Compare two TI files (vector-by-vector analysis)...")
        print("Note: This will open new file dialogs to select two files for comparison.")
        success = compare_two_ti_files()
        if success:
            print("✅ File comparison completed successfully!")
        else:
            print("❌ File comparison failed or was cancelled.")
        return True  # Successfully completed processing
    
    # ========================================
    # 4. EXECUTE SELECTED OPERATION
    # ========================================
    """
    Only perform the processing needed for the selected operation
    """
    
    print(f"\nProcessing with Option {user_input}...")
    
    # Initialize variables
    filename_suffix = ""
    processed_array = None
    
    if user_input == "1":
        # SMOOTHING OPERATION
        print("You chose Option 1 - retain all vectors, but with SMOOTHING...")
        
        if is_single_vector:
            print("\n⚠️  SINGLE VECTOR NOTICE:")
            print("   Since this is a single-vector file, smoothing will be applied to the one vector.")
            print("   The result will still be a single vector (no duplication for smoothing).")
            
            # Ask user if they want to continue
            continue_choice = input("\nContinue with smoothing? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("Operation cancelled.")
                return False
        
        # Get smoothing parameters
        sm_kernel = int(input("SMOOTHING: Enter kernel size (MUST BE ODD eg '3'), or enter '1' to skip smoothing: "))
        print('smoothing kernel used: ', sm_kernel)
        
        if sm_kernel == 1:
            # No smoothing
            processed_array = np_array.copy()
        else:
            # Apply smoothing
            np_smoothed = np_array.flatten()
            smooth_tmp = np.convolve(np_smoothed, np.ones(sm_kernel)/sm_kernel, mode='full')
            smooth_tmp = smooth_tmp[sm_kernel//2:len(smooth_tmp)-sm_kernel//2]
            processed_array = smooth_tmp.reshape(numvectors, -1)
        
        filename_suffix = f"_sm{sm_kernel}.pt"
        
    elif user_input == "2":
        # MEAN VECTOR OPERATION
        print("You chose Option 2 - condense to a SINGLE (scaled) MEAN vector...")
        
        # Calculate statistics for scaling
        sd_values = [np.std(np_array[i]) for i in range(np_array.shape[0])]
        meanSD = np.mean(sd_values)
        
        # Calculate mean vector
        Xmean = np.mean(np_array, axis=0)
        sd_val = np.std(Xmean)
          # Scale to match original SD
        Xmean = Xmean * meanSD/sd_val
        processed_array = Xmean.reshape(1, -1)
        
        filename_suffix = "_mean.pt"
        
    elif user_input == "3":
        # DECIMATION OPERATION
        print("You chose Option 3 - retain all vectors, but decimated with zeros...")
        
        # Get threshold values from user
        print("\nThreshold-based decimation:")
        print("Values will be set to zero if they fall within the threshold ranges")
        
        # Get positive threshold
        while True:
            try:
                pos_threshold = float(input("Enter positive threshold (positive values < this will become 0): "))
                if pos_threshold > 0:
                    break
                else:
                    print("Please enter a positive value")
            except ValueError:
                print("Please enter a valid number")
        
        # Get negative threshold  
        while True:
            try:
                neg_threshold = float(input("Enter negative threshold (negative values > this will become 0): "))
                if neg_threshold < 0:
                    break
                else:
                    print("Please enter a negative value")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"\nApplying thresholds:")
        print(f"  Positive values < {pos_threshold} → 0")
        print(f"  Negative values > {neg_threshold} → 0")
        
        # Apply decimation thresholds
        processed_array = np_array.copy()
        
        # Count affected values for reporting
        pos_count = np.sum((processed_array > 0) & (processed_array < pos_threshold))
        neg_count = np.sum((processed_array < 0) & (processed_array > neg_threshold))
        
        # Apply positive threshold: zero out positive values below threshold
        processed_array[(processed_array > 0) & (processed_array < pos_threshold)] = 0
        
        # Apply negative threshold: zero out negative values above threshold (closer to zero)
        processed_array[(processed_array < 0) & (processed_array > neg_threshold)] = 0
        
        print(f"✓ Decimation complete!")
        print(f"  {pos_count} positive values set to zero")
        print(f"  {neg_count} negative values set to zero")
        print(f"  Total values affected: {pos_count + neg_count}")
        
        filename_suffix = f"_dec_pos{pos_threshold}_neg{neg_threshold}.pt"
        
    elif user_input == "4":
        # DIVISION OPERATION
        print("You chose Option 4 - retain all vectors, but divided by a scalar...")
        
        divisor = float(input("Enter the divisor for the tensor: "))
        processed_array = np_array / divisor
        print(f'TI divided successfully by {divisor}')
        
        filename_suffix = f"_div{divisor}.pt"
        
    elif user_input == "5":
        # ROLLING OPERATION
        print("You chose Option 5 - retain all vectors, but ROLLED...")
        
        roll_amount = int(input("ROLLING: Enter eg '3', or enter '0' to skip rolling (can be +ve or -ve): "))
        print('roll_amount used: ', roll_amount)
        
        if roll_amount == 0:
            processed_array = np_array.copy()
        else:
            np_rolled = np_array.flatten()
            np_rolled = np.roll(np_rolled, roll_amount)
            processed_array = np_rolled.reshape(numvectors, -1)
        
        filename_suffix = f"_roll{roll_amount}.pt"
    
    # Convert processed array back to tensor
    tensor = torch.tensor(processed_array, device='cpu', requires_grad=True)
    data['string_to_param']['*'] = tensor
    
    # ========================================
    # 5. SAVE THE PROCESSED TI
    # ========================================
    """
    Save the processed file
    """
    
    print(f"\nProcessing complete! Tensor shape: {tensor.shape}")
    
    # Ask the user for a filename
    base_filename = input(f"Please enter a filename ('{filename_suffix}' will be appended): ")
    final_filename = base_filename + filename_suffix
    
    # Specify the directory path
    directory = "textual_inversions"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the file to the directory
    filepath = os.path.join(directory, final_filename)
    torch.save(data, filepath)
    
    print(f"✅ The file '{final_filename}' has been saved to the '{directory}' directory.")
    print("\nOperation completed successfully!")
    return True  # Successfully completed processing


def compare_two_ti_files():
    """
    Compare two TI files vector-by-vector with visualization and save options
    """
    import tkinter as tk
    from tkinter import filedialog
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    print("="*60)
    print("TI FILE COMPARISON - VECTOR BY VECTOR ANALYSIS")
    print("="*60)
    print("This tool loads two TI files and performs detailed comparison.")
    print("Provides visualization and options to save averaged/difference files.")
    print("="*60)
    
    # Set up file dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.geometry("800x600")
    
    # Load first file
    print("\n📁 Select FIRST TI file to compare...")
    file1_path = filedialog.askopenfilename(
        title="Select FIRST TI .pt file",
        filetypes=[("PyTorch TI files", "*.pt"), ("All files", "*.*")],
        parent=root
    )
    
    if not file1_path:
        print("❌ No first file selected. Returning to main menu.")
        return False
    
    # Load second file
    print("\n📁 Select SECOND TI file to compare...")
    file2_path = filedialog.askopenfilename(
        title="Select SECOND TI .pt file",
        filetypes=[("PyTorch TI files", "*.pt"), ("All files", "*.*")],
        parent=root
    )
    
    if not file2_path:
        print("❌ No second file selected. Returning to main menu.")
        return False
    
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    
    print(f"\n🔍 Loading and analyzing files...")
    print(f"   File 1: {file1_name}")
    print(f"   File 2: {file2_name}")
    
    # Load first file
    result1 = load_ti_file_flexible(file1_path)
    if not result1:
        print(f"❌ Failed to load first file: {file1_name}")
        return False
    
    data1, tensor1, path1, is_single1 = result1
    np_array1 = tensor1.cpu().detach().numpy()
    
    # Load second file
    result2 = load_ti_file_flexible(file2_path)
    if not result2:
        print(f"❌ Failed to load second file: {file2_name}")
        return False
    
    data2, tensor2, path2, is_single2 = result2
    np_array2 = tensor2.cpu().detach().numpy()
    
    print(f"\n✅ Files loaded successfully!")
    print(f"   File 1: {np_array1.shape} - {'Single vector' if is_single1 else 'Multi-vector'}")
    print(f"   File 2: {np_array2.shape} - {'Single vector' if is_single2 else 'Multi-vector'}")
    
    # Check compatibility
    if np_array1.shape != np_array2.shape:
        print(f"\n⚠️  WARNING: Files have different shapes!")
        print(f"   File 1: {np_array1.shape}")
        print(f"   File 2: {np_array2.shape}")
        
        # Try to make them compatible
        min_vectors = min(np_array1.shape[0], np_array2.shape[0])
        min_dims = min(np_array1.shape[1], np_array2.shape[1])
        
        print(f"\n🔧 Attempting to make compatible by using:")
        print(f"   Vectors: {min_vectors} (minimum of both)")
        print(f"   Dimensions: {min_dims} (minimum of both)")
        
        np_array1 = np_array1[:min_vectors, :min_dims]
        np_array2 = np_array2[:min_vectors, :min_dims]
        
        print(f"✅ Trimmed both arrays to: {np_array1.shape}")
    
    num_vectors = np_array1.shape[0]
    num_dims = np_array1.shape[1]
    
    # Calculate statistics
    print(f"\n📊 COMPARISON STATISTICS:")
    print(f"   Vectors to compare: {num_vectors}")
    print(f"   Dimensions per vector: {num_dims}")
    
    # Calculate differences and averages
    difference_array = np_array2 - np_array1
    average_array = (np_array1 + np_array2) / 2.0
    
    # Overall statistics
    file1_range = [np_array1.min(), np_array1.max()]
    file2_range = [np_array2.min(), np_array2.max()]
    diff_range = [difference_array.min(), difference_array.max()]
    avg_range = [average_array.min(), average_array.max()]
    
    print(f"\n📈 VALUE RANGES:")
    print(f"   File 1 range: [{file1_range[0]:.6f}, {file1_range[1]:.6f}]")
    print(f"   File 2 range: [{file2_range[0]:.6f}, {file2_range[1]:.6f}]")
    print(f"   Difference range: [{diff_range[0]:.6f}, {diff_range[1]:.6f}]")
    print(f"   Average range: [{avg_range[0]:.6f}, {avg_range[1]:.6f}]")
    
    # Calculate similarity metrics
    mse = np.mean((np_array1 - np_array2) ** 2)
    mae = np.mean(np.abs(np_array1 - np_array2))
    max_diff = np.max(np.abs(np_array1 - np_array2))
    
    # Correlation coefficient
    correlation = np.corrcoef(np_array1.flatten(), np_array2.flatten())[0, 1]
    
    print(f"\n🔍 SIMILARITY METRICS:")
    print(f"   Mean Squared Error (MSE): {mse:.8f}")
    print(f"   Mean Absolute Error (MAE): {mae:.8f}")
    print(f"   Maximum Difference: {max_diff:.8f}")
    print(f"   Correlation Coefficient: {correlation:.6f}")
    
    # Visualize the comparison
    print(f"\n🎨 Generating vector-by-vector visualization...")
    
    # Create heatmap visualization
    heatmap_height = HEATMAP_HEIGHT  # 24
    heatmap_width = HEATMAP_WIDTH    # 36
    target_size = heatmap_height * heatmap_width
    
    # Calculate grid for subplots
    cols = min(4, num_vectors)
    rows = math.ceil(num_vectors / cols)
    
    # Create figure with 3 rows per vector (file1, file2, difference)
    fig_height = rows * 9  # 3 subplots per row * 3 inches each
    fig_width = cols * 4   # 4 inches per column
    
    fig, axes = plt.subplots(rows * 3, cols, figsize=(fig_width, fig_height))
    
    # Handle case where we have only one column or row
    if num_vectors == 1:
        axes = axes.reshape(-1, 1) if axes.ndim == 1 else axes
    elif rows == 1:
        axes = axes.reshape(3, -1)
    
    # Use single diverging colormap for all plots
    cmap_diverging = plt.get_cmap('RdBu_r')
    
    # Calculate global min/max for consistent scaling across all plots
    global_min = min(np_array1.min(), np_array2.min(), difference_array.min())
    global_max = max(np_array1.max(), np_array2.max(), difference_array.max())
    
    # Ensure symmetric scaling around zero for diverging colormap
    global_abs_max = max(abs(global_min), abs(global_max))
    vmin, vmax = -global_abs_max, global_abs_max
    
    # Process each vector
    for i in range(num_vectors):
        col = i % cols
        base_row = (i // cols) * 3
        
        # Get vector data
        vector1 = np_array1[i].copy()
        vector2 = np_array2[i].copy()
        vector_diff = difference_array[i].copy()
        
        # Resize vectors to heatmap dimensions
        if num_dims != target_size:
            # Pad or truncate to target size
            if num_dims < target_size:
                # Pad with zeros
                padded1 = np.zeros(target_size)
                padded2 = np.zeros(target_size)
                padded_diff = np.zeros(target_size)
                padded1[:num_dims] = vector1
                padded2[:num_dims] = vector2
                padded_diff[:num_dims] = vector_diff
                vector1, vector2, vector_diff = padded1, padded2, padded_diff
            else:
                # Truncate
                vector1 = vector1[:target_size]
                vector2 = vector2[:target_size]
                vector_diff = vector_diff[:target_size]
        
        # Reshape to 2D heatmap
        heatmap1 = vector1.reshape(heatmap_height, heatmap_width)
        heatmap2 = vector2.reshape(heatmap_height, heatmap_width)
        heatmap_diff = vector_diff.reshape(heatmap_height, heatmap_width)
        
        # Plot File 1 vector with consistent scaling
        ax1 = axes[base_row, col]
        im1 = ax1.imshow(heatmap1, cmap=cmap_diverging, aspect='auto', vmin=vmin, vmax=vmax)
        ax1.set_title(f'F1,V{i+1}\n[{vector1.min():.2f}, {vector1.max():.2f}]', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # Plot File 2 vector with consistent scaling
        ax2 = axes[base_row + 1, col]
        im2 = ax2.imshow(heatmap2, cmap=cmap_diverging, aspect='auto', vmin=vmin, vmax=vmax)
        ax2.set_title(f'F2,V{i+1}\n[{vector2.min():.2f}, {vector2.max():.2f}]', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        # Plot Difference with consistent scaling
        ax3 = axes[base_row + 2, col]
        im3 = ax3.imshow(heatmap_diff, cmap=cmap_diverging, aspect='auto', vmin=vmin, vmax=vmax)
        ax3.set_title(f'Diff,V{i+1}\n[{vector_diff.min():.2f}, {vector_diff.max():.2f}]', fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.colorbar(im3, ax=ax3, shrink=0.6)
    
    # Hide unused subplots
    total_subplots = (rows * 3) * cols
    used_subplots = num_vectors * 3
    for i in range(used_subplots, total_subplots):
        row = i // cols
        col = i % cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)
    
    # Set overall title with 3 decimal places for statistics
    fig.suptitle(f'TI File Comparison: {file1_name} vs {file2_name}\n'
                f'MSE: {mse:.3f}, MAE: {mae:.3f}, Correlation: {correlation:.3f}', 
                fontsize=12, y=0.98)
    
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    
    # Ask user about saving results
    print(f"\n💾 SAVE OPTIONS:")
    print(f"1. Save AVERAGE of both files")
    print(f"2. Save DIFFERENCE of files (File2 - File1)")
    print(f"3. Save BOTH average and difference")
    print(f"4. Skip saving and return to menu")
    
    while True:
        try:
            save_choice = input(f"\nEnter your choice (1-4): ").strip()
            if save_choice in ['1', '2', '3', '4']:
                break
            else:
                print("Please enter 1, 2, 3, or 4")
        except (ValueError, KeyboardInterrupt):
            print("Please enter a valid choice")
    
    if save_choice == '4':
        print("✅ Comparison completed without saving.")
        return True
    
    # Create base filename from the two input files
    base1 = file1_name.replace('.pt', '')
    base2 = file2_name.replace('.pt', '')
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    saved_files = []
    
    # Save average file
    if save_choice in ['1', '3']:
        print(f"\n💾 Saving AVERAGE file...")
        avg_tensor = torch.tensor(average_array, device='cpu', requires_grad=True)
        avg_data = data1.copy()  # Use structure from first file
        avg_data['string_to_param'] = {'*': avg_tensor}
        
        avg_filename = f"{base1}_AVG_{base2}.pt"
        avg_filepath = os.path.join(directory, avg_filename)
        torch.save(avg_data, avg_filepath)
        
        print(f"✅ Average file saved: {avg_filename}")
        print(f"   Shape: {avg_tensor.shape}")
        print(f"   Range: [{average_array.min():.6f}, {average_array.max():.6f}]")
        saved_files.append(avg_filename)
    
    # Save difference file
    if save_choice in ['2', '3']:
        print(f"\n💾 Saving DIFFERENCE file...")
        diff_tensor = torch.tensor(difference_array, device='cpu', requires_grad=True)
        diff_data = data1.copy()  # Use structure from first file
        diff_data['string_to_param'] = {'*': diff_tensor}
        
        diff_filename = f"{base1}_DIFF_{base2}.pt"
        diff_filepath = os.path.join(directory, diff_filename)
        torch.save(diff_data, diff_filepath)
        
        print(f"✅ Difference file saved: {diff_filename}")
        print(f"   Shape: {diff_tensor.shape}")
        print(f"   Range: [{difference_array.min():.6f}, {difference_array.max():.6f}]")
        saved_files.append(diff_filename)
    
    # Summary
    print(f"\n🎉 COMPARISON SUMMARY:")
    print(f"   Files compared: {file1_name} vs {file2_name}")
    print(f"   Vectors compared: {num_vectors}")
    print(f"   Dimensions: {num_dims}")
    print(f"   Similarity (correlation): {correlation:.6f}")
    print(f"   Files saved: {len(saved_files)}")
    for filename in saved_files:
        print(f"     - {filename}")
    
    print(f"\n✅ Vector-by-vector comparison completed successfully!")
    return True


def main():
    """Main function with loop for processing multiple files"""
    print("="*60)
    print("TI CHANGER MULTIPLE - TEXTUAL INVERSION FILE PROCESSOR")
    print("="*60)
    print("This tool allows you to manipulate and transform textual inversion files.")
    print("You can process multiple files in sequence or compare two files.")
    print("="*60)
    
    while True:
        print(f"\n{'='*60}")
        print("MAIN MENU - CHOOSE OPERATION TYPE")
        print("="*60)
        print("1. Process a single TI file (load, analyze, transform)")
        print("2. Compare two TI files (vector-by-vector analysis)")
        print("3. Exit")
        print("="*60)
        
        # Get user choice for operation type
        while True:
            try:
                main_choice = input("Choose operation type (1-3): ").strip()
                if main_choice in ['1', '2', '3']:
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{'='*60}")
                print("SESSION INTERRUPTED BY USER")
                print("="*60)
                print("Goodbye!")
                return
        
        if main_choice == '3':
            print(f"\n{'='*60}")
            print("THANK YOU FOR USING TI CHANGER MULTIPLE!")
            print("="*60)
            print("Session completed. Goodbye!")
            return
        elif main_choice == '2':
            # Compare two files directly
            print(f"\n{'='*60}")
            print("COMPARING TWO TI FILES")
            print("="*60)
            print("This will open file dialogs to select two files for comparison.")
            success = compare_two_ti_files()
            if success:
                print("✅ File comparison completed successfully!")
            else:
                print("❌ File comparison failed or was cancelled.")
        else:
            # Process a single file (main_choice == '1')
            print(f"\n{'='*60}")
            print("STARTING NEW FILE PROCESSING SESSION")
            print("="*60)
            
            # Process a single file
            success = process_single_file()
            
            if success:
                print(f"\n{'='*60}")
                print("FILE PROCESSING COMPLETED SUCCESSFULLY!")
                print("="*60)
            else:
                print(f"\n{'='*60}")
                print("FILE PROCESSING WAS CANCELLED OR FAILED")
                print("="*60)
        
        # Ask user if they want to continue
        print("\nWould you like to perform another operation?")
        while True:
            try:
                continue_choice = input("Enter 'Y' for Yes or 'N' for No: ").strip().upper()
                if continue_choice in ['Y', 'YES']:
                    print("\nReturning to main menu...")
                    break
                elif continue_choice in ['N', 'NO']:
                    print(f"\n{'='*60}")
                    print("THANK YOU FOR USING TI CHANGER MULTIPLE!")
                    print("="*60)
                    print("Session completed. Goodbye!")
                    return
                else:
                    print("Please enter 'Y' for Yes or 'N' for No")
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{'='*60}")
                print("SESSION INTERRUPTED BY USER")
                print("="*60)
                print("Goodbye!")
                return


if __name__ == "__main__":
    main()
