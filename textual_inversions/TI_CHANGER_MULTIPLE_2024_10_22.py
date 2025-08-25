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
    # 2. MENU SELECTION
    # ========================================
    """
    Present user with operation choices upfront
    """
    
    print("\n" + "="*60)
    print("TI CHANGER - SELECT OPERATION")
    print("="*60)
    print("1. Apply smoothing to all vectors")
    print("2. Create single mean vector (condensed)")
    print("3. Apply decimation with zeros")
    print("4. Divide all vectors by scalar")
    print("5. Roll/shift all vectors")
    print("6. Extract individual vectors to separate files")
    print("="*60)
    
    user_input = input("Choose operation (1-6): ").strip()
    
    if user_input not in ['1', '2', '3', '4', '5', '6']:
        print("Invalid choice. Please run the script again and enter 1-6.")
        input("Press Enter to exit...")
        return
    
    print(f"\nSelected: Option {user_input}")
    
    # Handle Option 6 (extract vectors) immediately since it doesn't need processing
    if user_input == "6":
        print("You chose Option 6 - extract individual vectors to separate files...")
        extract_individual_vectors(data, filename, numvectors)
        return  # Exit early since we're not modifying the original file
    
    # ========================================
    # 3. EXECUTE SELECTED OPERATION
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
    # 4. SAVE THE PROCESSED TI
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
