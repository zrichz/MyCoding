#https://30fps.net/notebooks/pcacolors/
#License (MIT)
#Copyright (c) 2024 Pekka Väänänen
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
RGB Channel Pair Reduction Demo with GUI File Selection

This program demonstrates how different RGB channel pairs (RG, RB, GB) can be used to 
reduce color dimensionality compared to PCA-determined components. Here's what it does:

1. FILE SELECTION: Opens a GUI dialog to let the user select any image file
2. IMAGE PROCESSING: Loads and converts the image to RGB format, resizes if needed
3. CHANNEL PAIR ANALYSIS: Tests three specific channel combinations:
   - RG (Red-Green): Drops the Blue channel
   - RB (Red-Blue): Drops the Green channel  
   - GB (Green-Blue): Drops the Red channel
4. RECONSTRUCTION: Reconstructs images using only the selected channel pairs
5. COMPARISON VISUALIZATION: Creates displays showing:
   - Original image
   - Reconstructions using RG, RB, and GB pairs
   - Error analysis for each channel pair
   - Channel pair distributions in 2D space

This approach differs from PCA by using fixed channel combinations rather than 
mathematically optimal principal components. It's useful for:
- Understanding the contribution of individual color channels
- Comparing manual channel selection vs PCA optimization
- Educational demonstration of color space relationships
- Analysis of which channels are most important for specific images
"""

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def select_image_file():
    """
    Open a file dialog to let the user select an image file
    Returns the selected file path or None if cancelled
    """
    # Create a root window (but don't show it)
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Define supported image file types
    filetypes = [
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image file for PCA color reduction",
        filetypes=filetypes
    )
    
    root.destroy()  # Clean up the root window
    return file_path

def load_and_process_image(file_path):
    """
    Load and process the selected image
    Returns processed image data or None if error
    """
    try:
        # Load image with PIL
        image = Image.open(file_path)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original size for display
        original_size = image.size
        
        # Resize for processing (but not too small for very small images)
        max_dimension = max(image.size)
        if max_dimension > 400:
            # Scale down large images for faster processing
            scale_factor = 400 / max_dimension
            new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        print(f"Processing image: {os.path.basename(file_path)}")
        print(f"Original size: {original_size}")
        print(f"Processing size: {image.size}")
        
        # Convert to numpy array and normalize
        image_data = np.array(image) / 255.0
        
        return image_data, os.path.basename(file_path)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        return None, None

# Get image file from user
print("PCA Color Reduction Demo")
print("Please select an image file to process...")

file_path = select_image_file()

if not file_path:
    print("No file selected. Exiting.")
    exit()

# Load and process the selected image
image_data, filename = load_and_process_image(file_path)

if image_data is None:
    print("Failed to load image. Exiting.")
    exit()

flattened_image_data = image_data.reshape(-1, 3)

# Define the three channel pairs to test
channel_pairs = {
    'RG': ([0, 1], 'Red-Green'),  # Red and Green channels
    'RB': ([0, 2], 'Red-Blue'),   # Red and Blue channels  
    'GB': ([1, 2], 'Green-Blue')  # Green and Blue channels
}

# Store results for each channel pair
results = {}

print(f"\nAnalyzing channel pairs for {filename}:")
print("=" * 50)

for pair_name, (channels, full_name) in channel_pairs.items():
    print(f"Processing {pair_name} ({full_name})...")
    
    # Extract only the selected channels
    two_channel_data = flattened_image_data[:, channels]
    
    # Create reconstruction by setting missing channel to average of the two selected channels
    reconstructed_flat = np.zeros_like(flattened_image_data)
    reconstructed_flat[:, channels] = two_channel_data
    
    # For the missing channel, use the average of the two selected channels
    missing_channel = [ch for ch in [0, 1, 2] if ch not in channels][0]
    reconstructed_flat[:, missing_channel] = np.mean(two_channel_data, axis=1)
    
    # Reshape back to image dimensions
    reconstruction = reconstructed_flat.reshape(image_data.shape)
    
    # Calculate reconstruction error
    error = np.mean(np.abs(image_data - reconstruction))
    mse_error = np.mean((image_data - reconstruction) ** 2)
    
    # Store results
    results[pair_name] = {
        'data': two_channel_data,
        'reconstruction': reconstruction,
        'error_l1': error,
        'error_mse': mse_error,
        'channels': channels,
        'full_name': full_name,
        'colors': [tuple(flattened_image_data[i]) for i in range(flattened_image_data.shape[0])]
    }
    
    print(f"  L1 Error: {error:.6f}")
    print(f"  MSE Error: {mse_error:.6f}")

# Find the best performing channel pair
best_pair = min(results.keys(), key=lambda k: results[k]['error_l1'])
print(f"\nBest performing pair: {best_pair} ({results[best_pair]['full_name']}) with L1 error: {results[best_pair]['error_l1']:.6f}")

# Create comprehensive visualization with larger plots
fig = plt.figure(figsize=(16, 10))

# Create a 2x4 grid for better organization (removing distribution plots)
gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.25)

# Original image (top left)
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.imshow(image_data)
ax_orig.set_title("Original Image", fontsize=14, fontweight='bold')
ax_orig.axis('off')

# Channel pair reconstructions (top row)
for i, (pair_name, result) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, i+1])
    ax.imshow(result['reconstruction'].clip(0, 1))
    ax.set_title(f"{pair_name} Reconstruction\nL1 Error: {result['error_l1']:.4f}", 
                fontsize=12, fontweight='bold' if pair_name == best_pair else 'normal')
    ax.axis('off')

# Error maps (second row)
ax_orig_label = fig.add_subplot(gs[1, 0])
ax_orig_label.text(0.5, 0.5, 'Error Maps →', ha='center', va='center', 
                  fontsize=16, fontweight='bold', transform=ax_orig_label.transAxes)
ax_orig_label.axis('off')

for i, (pair_name, result) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i+1])
    error_map = np.mean(np.abs(image_data - result['reconstruction']), axis=2)
    im = ax.imshow(error_map, cmap='hot')
    ax.set_title(f"{pair_name} Error Map\nMax Error: {error_map.max():.3f}", 
                fontsize=12, fontweight='bold' if pair_name == best_pair else 'normal')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Overall title
plt.suptitle(f"RGB Channel Pair Analysis: {filename}", fontsize=18, fontweight='bold')

plt.show()

# Print detailed analysis
print(f"\n" + "="*60)
print("DETAILED CHANNEL PAIR ANALYSIS")
print("="*60)
print(f"Image: {filename}")
print(f"Processing size: {image_data.shape[:2]}")
print(f"Total pixels: {flattened_image_data.shape[0]:,}")

print(f"\nChannel Pair Performance:")
print(f"{'Pair':<4} {'Channels':<12} {'L1 Error':<12} {'MSE Error':<12} {'Ranking'}")
print("-" * 55)

# Sort by L1 error for ranking
sorted_pairs = sorted(results.items(), key=lambda x: x[1]['error_l1'])
for rank, (pair_name, result) in enumerate(sorted_pairs, 1):
    channels_str = f"{['R','G','B'][result['channels'][0]]}{['R','G','B'][result['channels'][1]]}"
    star = " ⭐" if rank == 1 else ""
    print(f"{pair_name:<4} {channels_str:<12} {result['error_l1']:<12.6f} {result['error_mse']:<12.6f} #{rank}{star}")

print(f"\nAnalysis Summary:")
print(f"• Best pair: {best_pair} ({results[best_pair]['full_name']})")
print(f"• Lowest error: {results[best_pair]['error_l1']:.6f}")
print(f"• This suggests {results[best_pair]['full_name'].lower()} channels contain the most visual information for this image")

# Calculate channel importance
channel_errors = {'Red': [], 'Green': [], 'Blue': []}
for pair_name, result in results.items():
    # Channels present in this pair have "contributed" to its performance
    for ch_idx in result['channels']:
        channel_names = ['Red', 'Green', 'Blue']
        channel_errors[channel_names[ch_idx]].append(result['error_l1'])

print(f"\nChannel Importance Analysis:")
for color in ['Red', 'Green', 'Blue']:
    avg_error = np.mean(channel_errors[color])
    print(f"• {color} channel average error when present: {avg_error:.6f}")

print(f"\nNote: Lower error indicates the channel pair better preserves the original image information.")