#!/usr/bin/env python3
"""
TI Changer Multiple - Convert and manipulate textual inversion files
Converted from Jupyter notebook to Python script

New Features Added:
- Option 6: Extract individual vectors from multi-vector TI files
  This option allows you to:
  * Load a .pt file containing multiple vectors (e.g., 8 vectors)
  * Display the number of vectors present
  * Extract each vector as a separate .pt file
  * Save with numbered suffixes (e.g., filename_vector_01.pt, filename_vector_02.pt, etc.)
  
Usage for Option 6:
1. Run the script
2. Choose option '6' when prompted
3. Confirm the extraction when shown the list of files to be created
4. Each vector will be saved as an individual, fully functional TI file

Example: An 8-vector TI file named "my_embedding.pt" will create:
- my_embedding_vector_01.pt
- my_embedding_vector_02.pt
- my_embedding_vector_03.pt
- my_embedding_vector_04.pt
- my_embedding_vector_05.pt
- my_embedding_vector_06.pt
- my_embedding_vector_07.pt
- my_embedding_vector_08.pt
"""

# Load a .pt textual inversion file and show / manipulate it
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configure matplotlib for better display (standalone Python version)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150


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
    Robust user input function that handles Windows batch file input issues
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
    print("3. Apply decimation with zeros")
    print("4. Divide all vectors by scalar")
    print("5. Roll/shift all vectors")
    print("6. Extract individual vectors to separate files")
    print("7. Save top N vectors by absolute magnitude")
    print("8. Clustering-Based Reduction (K-means)")
    print("9. Principal Component Analysis (PCA)")
    print("="*60)
    
    # Improved input handling with retry logic
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Add a small delay to ensure console is ready
            import time
            time.sleep(0.1)
            
            print(f"\nAttempt {attempt + 1}: ", end="", flush=True)
            user_input = input("Choose operation (1-9): ").strip()
            
            # Debug: Show what was actually captured (remove this after fixing)
            print(f"Debug: Captured input: '{user_input}' (length: {len(user_input)})")
            
            if user_input in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return user_input
            else:
                print(f"Invalid input: '{user_input}'. Please enter a number from 1 to 9.")
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
    
    print("Invalid choice after multiple attempts. Please run the script again and enter 1-9.")
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
    


def main():
    """Main function to execute the TI manipulation workflow"""
    
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

    print("Please select a .pt file to load...")
    filename = filedialog.askopenfilename(
        title="Select a TI .pt file",
        filetypes=[("PyTorch TI files", "*.pt"), ("All files", "*.*")]
    )

    if not filename:
        print("No file selected. Exiting.")
        return

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
            input("\nPress Enter to exit...")
            return
        else:
            print("✅ Flexible loading succeeded! Proceeding...")
    
    # Load the file using flexible approach
    flexible_result = load_ti_file_flexible(original_selected_path)
    if not flexible_result:
        print("❌ Failed to load file even with flexible parsing.")
        input("\nPress Enter to exit...")
        return
    
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
    
    # Create the plot with wider figure for bars and better spacing
    bar_width = 0.7  # Width of each bar (wider since we only have 2 bars)
    fig, ax = plt.subplots(figsize=(max(10, numvectors * 1.5), 7))
    
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
    
    # Customize the plot
    ax.set_title(f'Vector Statistics for {filename}\n({numvectors} vectors, {np_array.shape[1]} dimensions each)', fontsize=12, pad=20)
    ax.set_xlabel('Vector Number', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='best', framealpha=0.9)
    
    # Set x-axis labels and positions
    ax.set_xticks(x_pos)
    ax.set_xticklabels(vector_numbers)
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Overall Min: {min(min_values):.6f}")
    print(f"   Overall Max: {max(max_values):.6f}")
    print(f"   Range: {max(max_values) - min(min_values):.6f}")
    print(f"   Y-axis range: ±{y_limit:.6f}")
    
    # ========================================
    # 3. MENU SELECTION
    # ========================================
    """
    Present user with operation choices upfront
    """
    
    user_input = get_user_choice()
    if user_input is None:
        input("Press Enter to exit...")
        return
    
    print(f"\nSelected: Option {user_input}")
    
    # Handle Option 6 (extract vectors) immediately since it doesn't need processing
    if user_input == "6":
        print("You chose Option 6 - extract individual vectors to separate files...")
        extract_individual_vectors(data, filename, numvectors)
        return  # Exit early since we're not modifying the original file
    
    # Handle Option 7 (top N vectors) immediately since it doesn't need processing
    if user_input == "7":
        print("You chose Option 7 - save top N vectors by absolute magnitude...")
        select_top_n_vectors(data, filename, numvectors, np_array)
        return  # Exit early since we're not modifying the original file
    
    # Handle Option 8 (clustering-based reduction) immediately since it doesn't need processing
    if user_input == "8":
        print("You chose Option 8 - clustering-based reduction (K-means)...")
        clustering_based_reduction(data, filename, numvectors, np_array)
        return  # Exit early since we're not modifying the original file
    
    # Handle Option 9 (PCA) immediately since it doesn't need processing
    if user_input == "9":
        print("You chose Option 9 - Principal Component Analysis (PCA)...")
        principal_component_analysis(data, filename, numvectors, np_array)
        return  # Exit early since we're not modifying the original file
    
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
        
        # For now, using placeholder decimation (you can enhance this)
        processed_array = np_array.copy()
        # TODO: Add your specific decimation logic here
        
        filename_suffix = "_dec.pt"
        
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


if __name__ == "__main__":
    main()
