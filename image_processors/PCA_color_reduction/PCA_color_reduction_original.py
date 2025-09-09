#https://30fps.net/notebooks/pcacolors/
#License (MIT)
#Copyright (c) 2024 Pekka Väänänen
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
PCA Color Reduction Demo with GUI File Selection

This program demonstrates how Principal Component Analysis (PCA) can be used to reduce 
the color dimensionality of any image. Here's what it does:

1. FILE SELECTION: Opens a GUI dialog to let the user select any image file
2. IMAGE PROCESSING: Loads and converts the image to RGB format, resizes if needed
3. COLOR EXTRACTION: Converts the RGB image into a flat array of color pixels
4. PCA ANALYSIS: Applies PCA to reduce 3 color channels (RGB) down to 2 components
5. RECONSTRUCTION: Uses the 2 PCA components to reconstruct an approximation of the original image
6. VISUALIZATION: Creates a 6-panel display showing:
   - Original image
   - PCA-reconstructed image  
   - The two encoded PCA channels as grayscale images
   - A scatter plot showing how colors are distributed in the 2D PCA space
   - The reconstruction error (difference between original and reconstructed)

Supported image formats: JPEG, PNG, BMP, TIFF, GIF, WebP

The key insight is that most images don't use the full RGB color space uniformly - 
colors tend to cluster, so PCA can find the 2 most important directions in color space
and represent the image using just those 2 dimensions instead of 3, achieving compression
while maintaining much of the visual information.

This is useful for:
- Image compression analysis
- Understanding color relationships in images
- Dimensionality reduction for computer vision tasks
- Educational demonstration of PCA concepts
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

colors = [tuple(flattened_image_data[i]) for i in range(flattened_image_data.shape[0])]

# Perform PCA:
pca = PCA(n_components=2)
model = pca.fit(flattened_image_data)
reduced_data = model.transform(flattened_image_data)
result = reduced_data.reshape((*image_data.shape[:2],2))
reconstruction = pca.inverse_transform(result).reshape((*image_data.shape[:2],3))

fig, ax = plt.subplots(3,2,figsize=(14,16))

ax_in, ax_reco, ax_out_a, ax_out_b, ax_pri1, ax_diff = ax.flatten()
ax_in.imshow(image_data)
ax_reco.imshow(reconstruction.clip(0,1))
ax_out_a.imshow(result[...,0])
ax_out_b.imshow(result[...,1])

ax_in.set_title("Input image")
ax_reco.set_title("Reconstruction")
ax_out_a.set_title("Encoded channel 1")
ax_out_b.set_title("Encoded channel 2")

ax_pri1.scatter(reduced_data[...,0], reduced_data[...,1], c=colors)
ax_pri1.set_xlabel("Principal direction 1")
ax_pri1.set_ylabel("Principal direction 2 ")
ax_pri1.set_title("Image data distribution in PCA space")

ax_diff.imshow(np.mean(np.abs(image_data - reconstruction), axis=2))
ax_diff.set_title("L1 error between input and reconstruction")

for a in ax.flatten()[:4]:
    a.axis('off')

ax_diff.axis('off')

plt.suptitle(f"PCA Color Reduction Analysis: {filename}")

plt.show()

print(f'Image: {filename}')
print('Principal components:\n', model.components_)
print(f'Explained variance ratio: {model.explained_variance_ratio_}')
print(f'Total variance explained: {sum(model.explained_variance_ratio_):.3f} ({sum(model.explained_variance_ratio_)*100:.1f}%)')