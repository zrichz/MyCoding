#!/usr/bin/env python3
"""
TI Changer Multiple - Convert and manipulate textual inversion files
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
from scipy.ndimage import zoom

# Configure matplotlib for better display (standalone Python version)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

# ========================================
# VISUALIZATION CONFIGURATION
# ========================================
# Customizable colormap for heatmap visualizations
HEATMAP_COLORS = ['#FF0000', "#BE6103", "#000000", "#000000", "#000000", 
                  "#000000", "#000000", "#8D8A00", "#D6C400", "#FFF200"]

# Alternative color schemes you can use by changing HEATMAP_COLORS above:
# COOL_COLORS = ['#000080', '#0000FF', '#0080FF', '#00FFFF', '#80FFFF', '#FFFFFF', '#FFFF80', '#FFFF00', '#FF8000', '#FF0000']
# WARM_COLORS = ['#000000', '#330000', '#660000', '#990000', '#CC0000', '#FF0000', '#FF3300', '#FF6600', '#FF9900', '#FFCC00']
# RAINBOW_COLORS = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000', '#FF1493', '#00CED1', '#32CD32']

# Heatmap dimensions (adjust based on your vector size preferences)
HEATMAP_HEIGHT = 36
HEATMAP_WIDTH = 24


def analyze_pt_file(filepath):
    """
    Analyze a .pt file to determine its type and structure with flexible TI detection
    """
    try:
        data = torch.load(filepath, map_location='cpu')
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
            if len(data.shape) == 2:
                print(f"✓ This could be a direct embedding tensor!")
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
                    if torch.is_tensor(tensor_candidate) and len(tensor_candidate.shape) == 2:
                        return (description, tensor_candidate)
                elif len(key_path) == 1 and final_key in ['string_to_param', 'embeddings', 'emb_params']:
                    # For containers, check all their contents
                    container = current[final_key]
                    if isinstance(container, dict):
                        for sub_key, sub_value in container.items():
                            if torch.is_tensor(sub_value) and len(sub_value.shape) == 2:
                                return (f"{description}['{sub_key}']", sub_value)
        except:
            continue
    
    # If no standard patterns work, search more broadly
    return search_all_tensors(data)


def search_all_tensors(data, path="", max_depth=3):
    """
    Recursively search for 2D tensors that could be embeddings
    """
    if max_depth <= 0:
        return None
        
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if torch.is_tensor(value) and len(value.shape) == 2:
                # Found a 2D tensor - likely an embedding
                return (f"deep search: {current_path}", value)
            elif isinstance(value, dict):
                result = search_all_tensors(value, current_path, max_depth - 1)
                if result:
                    return result
    
    return None


def load_ti_file_flexible(filepath):
    """
    Load a TI file with flexible structure detection
    Returns: (data_dict, tensor, tensor_path) or None if not a valid TI file
    """
    try:
        data = torch.load(filepath, map_location='cpu')
        
        # Handle raw tensor files
        if torch.is_tensor(data) and len(data.shape) == 2:
            # Create a minimal TI structure
            ti_data = {
                'string_to_param': {'*': data},
                'string_to_token': {'*': 49408},  # Default token
                'name': os.path.splitext(os.path.basename(filepath))[0],
                'step': 0
            }
            return (ti_data, data, "raw tensor converted to standard format")
        
        # Handle dictionary files
        if isinstance(data, dict):
            embedding_info = find_embedding_tensor(data)
            if embedding_info:
                tensor_path, tensor = embedding_info
                
                # Ensure we have a complete TI structure
                if 'string_to_param' not in data:
                    # Create missing structure
                    data['string_to_param'] = {'*': tensor}
                elif '*' not in data['string_to_param']:
                    data['string_to_param']['*'] = tensor
                
                # Add missing metadata if needed
                if 'string_to_token' not in data:
                    data['string_to_token'] = {'*': 49408}
                if 'name' not in data:
                    data['name'] = os.path.splitext(os.path.basename(filepath))[0]
                if 'step' not in data:
                    data['step'] = 0
                
                return (data, tensor, tensor_path)
        
        return None
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def get_user_choice():
    """
    user input function
    """
    import sys
    
    # Clear any potential input buffer issues
    if hasattr(sys.stdin, 'flush'):
        sys.stdin.flush()
    
    print("\n" + "="*60)
    print("TI CHANGER - SELECT OPERATION")
    print("="*60)
    print("1. Apply smoothing to all vectors")
    print("2. Create single mean vector (condensed)")
    print("3. Apply threshold-based decimation (zero small values)")
    print("4. Divide all vectors by scalar")
    print("5. Roll/shift all vectors")
    print("6. Extract individual vectors to separate files")
    print("7. Save top N vectors by absolute magnitude")
    print("8. Clustering-Based Reduction (K-means)")
    print("9. Principal Component Analysis (PCA)")
    print("10. Quantile Transform (Uniform/Gaussian)")
    print("11. Nonlinear Squashing (tanh)")
    print("12. L² Normalization")
    print("13. Max/Min Averaging (single vector from extremes)")
    print("14. Average specified vectors and combine with remaining")
    print("="*60)
    
    # Improved input handling with retry logic
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Add a small delay to ensure console is ready
            import time
            time.sleep(0.1)
            
            print(f"\nAttempt {attempt + 1}: ", end="", flush=True)
            user_input = input("Choose operation (1-14): ").strip()
            
            # Debug: Show what was actually captured (remove this after fixing)
            print(f"Debug: Captured input: '{user_input}' (length: {len(user_input)})")
            
            if user_input in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']:
                return user_input
            else:
                print(f"Invalid input: '{user_input}'. Please enter a number from 1 to 14.")
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
    
    print("Invalid choice after multiple attempts. Please run the script again and enter 1-14.")
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


def clustering_based_reduction(data, original_filename, numvectors, np_array):
    """
    Apply K-means clustering to reduce vectors to N cluster centroids
    
    Args:
        data: The original TI data dictionary
        original_filename: The original filename (without path)
        numvectors: Number of vectors in the TI file
        np_array: NumPy array of the tensor data
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("❌ Error: scikit-learn is required for clustering-based reduction.")
        print("Please install it with: pip install scikit-learn")
        return
    
    print(f"\nApplying Clustering-Based Reduction (K-means) to '{original_filename}'...")
    print(f"Available vectors: {numvectors}")
    
    # Get number of clusters from user
    while True:
        try:
            n_clusters = int(input(f"Enter number of clusters (1-{numvectors}): "))
            if 1 <= n_clusters <= numvectors:
                break
            else:
                print(f"Please enter a number between 1 and {numvectors}")
        except ValueError:
            print("Please enter a valid integer")
    
    print(f"\nApplying K-means clustering with {n_clusters} clusters...")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(np_array)
    centroids = kmeans.cluster_centers_
    
    print(f"✓ Clustering complete!")
    print(f"✓ Original vectors: {numvectors}")
    print(f"✓ Reduced to centroids: {n_clusters}")
    print(f"✓ Centroid tensor shape: {centroids.shape}")
    
    # Show cluster information
    print(f"\nCluster assignments:")
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
    print(f"\nSaving individual centroid files...")
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
    print(f"\nSaving combined centroids file...")
    combined_tensor = torch.tensor(centroids, device='cpu', requires_grad=True)
    combined_data = data.copy()
    combined_data['string_to_param'] = {'*': combined_tensor}
    
    combined_filename = f"{base_filename}_kmeans_{n_clusters}centroids.pt"
    combined_filepath = os.path.join(directory, combined_filename)
    torch.save(combined_data, combined_filepath)
    
    print(f"✅ Combined centroids file '{combined_filename}' saved to '{directory}' directory.")
    
    print(f"\n📊 K-means Clustering Summary:")
    print(f"   Original vectors: {numvectors}")
    print(f"   Clusters created: {n_clusters}")
    print(f"   Individual centroids: {n_clusters} files")
    print(f"   Combined file: {combined_filename}")
    print(f"   Dimensionality preserved: {centroids.shape[1]}")


def principal_component_analysis(data, original_filename, numvectors, np_array):
    """
    Apply PCA to reduce the number of vectors to N principal components
    
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
    
    # Get number of components from user
    while True:
        try:
            n_components = int(input(f"Enter number of principal components (1-{numvectors}): "))
            if 1 <= n_components <= numvectors:
                break
            else:
                print(f"Please enter a number between 1 and {numvectors}")
        except ValueError:
            print("Please enter a valid integer")
    
    print(f"\nApplying PCA with {n_components} components...")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_vectors = pca.fit_transform(np_array)
    
    # Get the principal components (these are the new vectors)
    # Note: pca.components_ gives us the components, but we want the transformed data
    # The transformed data is in the original vector space dimension
    # We need to transform back to get vectors in the original space
    reconstructed_vectors = pca.inverse_transform(pca_vectors)
    
    print(f"✓ PCA complete!")
    print(f"✓ Original vectors: {numvectors}")
    print(f"✓ Principal components: {n_components}")
    print(f"✓ Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"✓ Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    print(f"✓ PCA vectors shape: {reconstructed_vectors.shape}")
    
    # Show variance information
    print(f"\nVariance explained by each component:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  Component {i + 1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    
    # Create base filename
    base_filename = original_filename.replace('.pt', '')
    
    # Create directory if it doesn't exist
    directory = "textual_inversions"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save individual PCA component files
    print(f"\nSaving individual PCA component files...")
    for i in range(n_components):
        # Extract the i-th component (keep as 2D with shape [1, dimensions])
        individual_component = reconstructed_vectors[i:i+1]
        
        # Convert to tensor
        component_tensor = torch.tensor(individual_component, device='cpu', requires_grad=True)
        
        # Create new data dictionary for this individual component
        individual_data = data.copy()
        individual_data['string_to_param'] = {'*': component_tensor}
        
        # Generate filename
        component_filename = f"{base_filename}_pca_component_{i+1:02d}.pt"
        filepath = os.path.join(directory, component_filename)
        
        # Save the individual component file
        torch.save(individual_data, filepath)
        variance_pct = pca.explained_variance_ratio_[i] * 100
        print(f"✓ Saved component {i+1}/{n_components}: {component_filename} (explains {variance_pct:.2f}% variance)")
    
    # Save combined PCA components file
    print(f"\nSaving combined PCA components file...")
    combined_tensor = torch.tensor(reconstructed_vectors, device='cpu', requires_grad=True)
    combined_data = data.copy()
    combined_data['string_to_param'] = {'*': combined_tensor}
    
    combined_filename = f"{base_filename}_pca_{n_components}components.pt"
    combined_filepath = os.path.join(directory, combined_filename)
    torch.save(combined_data, combined_filepath)
    
    print(f"✅ Combined PCA components file '{combined_filename}' saved to '{directory}' directory.")
    
    print(f"\n📊 PCA Analysis Summary:")
    print(f"   Original vectors: {numvectors}")
    print(f"   Principal components: {n_components}")
    print(f"   Total variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"   Individual components: {n_components} files")
    print(f"   Combined file: {combined_filename}")
    print(f"   Dimensionality preserved: {reconstructed_vectors.shape[1]}")
    


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
            suffix = "_quantile_uniform"
        else:
            # Gaussian distribution (inverse normal CDF)
            transformed_vector = stats.norm.ppf(empirical_cdf)
            suffix = "_quantile_gaussian"
        
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
    
    # Save the file
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
    
    data, tensor, tensor_path = flexible_result
    print(f'\n✅ TI loaded successfully from {original_selected_path}')
    print(f'📍 Embedding tensor found at: {tensor_path}')
    print(f'📊 Final data structure keys: {list(data.keys())}')
    
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
    ax.set_title(f'Vector Statistics for {filename}\n({numvectors} vectors, {np_array.shape[1]} dimensions each)', 
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
    # NEW: INDIVIDUAL VECTOR VALUES LINE PLOT
    # ========================================
    """
    Display all vector values as line plots with x-axis 0-768 and different colored lines
    """
    
    print("\nGenerating individual vector values visualization...")
    
    # Create figure for vector line plots with better spacing
    fig2, ax2 = plt.subplots(figsize=(12, 6))  # Wider for better dimension visibility
      # Generate colors for each vector
    import matplotlib.cm as cm
    
    # Use different color strategies based on number of vectors
    if numvectors <= 10:
        # Use distinct colors for up to 10 vectors
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, 10))[:numvectors]
    else:
        # Use a continuous colormap for many vectors
        colors = cm.get_cmap('viridis')(np.linspace(0, 1, numvectors))
    
    # X-axis represents dimensions (0 to vector_length-1)
    x_dims = np.arange(np_array.shape[1])
    
    # Plot each vector as a separate line
    for i in range(numvectors):
        vector_values = np_array[i]
        color = colors[i]
        ax2.plot(x_dims, vector_values, 
                label=f'Vector {i+1}', 
                color=color, 
                linewidth=1.5, 
                alpha=0.8)
    
    # Customize the vector plot
    ax2.set_title(f'Individual Vector Values for {filename}\n({numvectors} vectors × {np_array.shape[1]} dimensions)', 
                 fontsize=10, pad=15)
    ax2.set_xlabel('Dimension Index (0-{})'.format(np_array.shape[1]-1), fontsize=9)
    ax2.set_ylabel('Vector Value', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend (but limit it if too many vectors)
    if numvectors <= 10:
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        ax2.text(1.02, 0.5, f'{numvectors} vectors\n(too many for legend)', 
                transform=ax2.transAxes, fontsize=8, verticalalignment='center')
    
    # Add a horizontal line at y=0 for reference
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Improve layout with more white space
    plt.tight_layout(pad=2.0)
    
    # Show the second plot
    plt.show()
    
    # ========================================
    # NEW: 2D HEATMAP GRID VISUALIZATION
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
        
        # Use raw heatmap data without smoothing
        
        # Create quantized colormap using configuration
        bright_colors = HEATMAP_COLORS
        n_colors = len(bright_colors)
        quantized_cmap = mcolors.ListedColormap(bright_colors)
        
        # Create quantized normalization to enforce discrete color levels
        boundaries = np.linspace(np.min(vector_data), np.max(vector_data), n_colors + 1)
        norm = mcolors.BoundaryNorm(boundaries, quantized_cmap.N)
        
        # Create heatmap
        ax = axes[i]
        im = ax.imshow(heatmap_data, cmap=quantized_cmap, norm=norm, aspect='auto', 
                      interpolation='nearest')  # Use raw values without smoothing
        
        # Customize each subplot
        ax.set_title(f'Vector {i+1}', fontsize=10, pad=5)
        ax.set_xlabel(f'Dim Width (0-{heatmap_width-1})', fontsize=8)
        ax.set_ylabel(f'Dim Height (0-{heatmap_height-1})', fontsize=8)
        
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
    fig3.suptitle(f'Vector Heatmaps for {filename}\n({numvectors} vectors reshaped to {heatmap_height}×{heatmap_width} for visualization)', 
                 fontsize=12, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    # Show the third plot
    plt.show()
    
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
    
    user_input = get_user_choice()
    if user_input is None:
        input("Press Enter to continue...")
        return False
    
    print(f"\nSelected: Option {user_input}")
    
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
    
    # Handle Option 14 (Average specified vectors) immediately since it doesn't need processing
    if user_input == "14":
        print("You chose Option 14 - Average specified vectors and combine with remaining...")
        average_specified_vectors(data, filename, numvectors, np_array)
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


def main():
    """Main function with loop for processing multiple files"""
    print("="*60)
    print("TI CHANGER MULTIPLE - TEXTUAL INVERSION FILE PROCESSOR")
    print("="*60)
    print("This tool allows you to manipulate and transform textual inversion files.")
    print("You can process multiple files in sequence.")
    print("="*60)
    
    while True:
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
        
        # Ask user if they want to process another file
        print("\nWould you like to process another textual inversion file?")
        while True:
            try:
                continue_choice = input("Enter 'Y' for Yes or 'N' for No: ").strip().upper()
                if continue_choice in ['Y', 'YES']:
                    print("\nStarting new file selection...")
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
