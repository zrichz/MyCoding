import os
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# Helper to load and preprocess images for CLIP
from torchvision.transforms import InterpolationMode

preprocess = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def load_images_from_folder(folder):
    images = []
    paths = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return images, paths

def load_multiple_clip_models(device):
    """Load multiple CLIP models for ensemble approach"""
    models = []
    model_names = ["ViT-B/32", "ViT-B/16", "RN50"]
    
    for model_name in model_names:
        try:
            print(f"Loading {model_name}...")
            model, _ = clip.load(model_name, device=device)
            models.append((model, model_name))
            print(f"✓ {model_name} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    if not models:
        raise RuntimeError("No CLIP models could be loaded!")
    
    return models

def compute_clip_features(images, model, device):
    features = []
    for img in tqdm(images, desc="Encoding images with CLIP"):
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image_input)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.cpu().numpy()[0])
    return np.stack(features)

def compute_ensemble_features(images, models, device):
    """Compute features using multiple CLIP models and concatenate them"""
    all_features = []
    
    for model, model_name in models:
        print(f"Computing features with {model_name}...")
        features = compute_clip_features(images, model, device)
        all_features.append(features)
    
    # Concatenate features from all models
    ensemble_features = np.concatenate(all_features, axis=1)
    print(f"Ensemble features shape: {ensemble_features.shape}")
    return ensemble_features

def compute_advanced_similarity(features, method='ensemble'):
    """Compute similarity matrix using advanced methods"""
    if method == 'euclidean':
        distances = pdist(features, 'euclidean')
        sim_matrix = 1 - squareform(distances) / np.max(distances)
        print("Using Euclidean distance similarity")
    elif method == 'correlation':
        sim_matrix = np.corrcoef(features)
        print("Using correlation coefficient similarity")
    elif method == 'ensemble':
        # Combine euclidean and correlation for best results
        euclidean_distances = pdist(features, 'euclidean')
        euclidean_sim = 1 - squareform(euclidean_distances) / np.max(euclidean_distances)
        correlation_sim = np.corrcoef(features)
        
        # Handle NaN values in correlation matrix
        correlation_sim = np.nan_to_num(correlation_sim, nan=0.0)
        
        # Weighted ensemble (60% euclidean, 40% correlation)
        sim_matrix = 0.6 * euclidean_sim + 0.4 * correlation_sim
        print("Using ensemble similarity (Euclidean + Correlation)")
    else:  # cosine (fallback)
        sim_matrix = 1 - squareform(pdist(features, 'cosine'))
        print("Using cosine similarity")
    
    return sim_matrix

def plot_similarity_matrix(sim_matrix, paths):
    """Plot the full similarity matrix as a 2D heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Create labels from filenames (shortened)
    labels = [os.path.basename(path)[:15] + "..." if len(os.path.basename(path)) > 15 
              else os.path.basename(path) for path in paths]
    
    # Plot the matrix
    im = plt.imshow(sim_matrix, cmap='jet', interpolation='nearest')
    plt.colorbar(im, label='Similarity Score')
    
    # Set ticks and labels
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    
    # Add values to cells (for smaller matrices)
    if len(sim_matrix) <= 20:
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix)):
                plt.text(j, i, f'{sim_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=6, 
                        color='white' if sim_matrix[i, j] < 0.5 else 'black')
    
    plt.title('Image Similarity Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Images', fontsize=12)
    plt.ylabel('Images', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_top_pairs(sim_matrix, paths, top_k=8):
    N = sim_matrix.shape[0]
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    sim_scores = sim_matrix[mask]
    idx_pairs = np.argwhere(mask)
    top_indices = np.argsort(sim_scores)[-top_k:][::-1]
    top_pairs = [idx_pairs[i] for i in top_indices]
    top_scores = [sim_scores[i] for i in top_indices]

    # Optimize for 1920x1080 screen - larger thumbnails
    fig, axes = plt.subplots(top_k, 2, figsize=(16, 20))
    
    # Handle single pair case
    if top_k == 1:
        axes = axes.reshape(1, -1)
    
    for i, ((idx1, idx2), score) in enumerate(zip(top_pairs, top_scores)):
        img1 = Image.open(paths[idx1])
        img2 = Image.open(paths[idx2])
        axes[i, 0].imshow(img1)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(img2)
        axes[i, 1].axis('off')
        # Add similarity score as a centered title for the pair
        fig.text(0.5, 1 - (i + 0.5) / top_k, f"Similarity: {score:.3f}", 
                ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.suptitle('Top Similar Image Pairs', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLIP Image Similarity Matrix")
    parser.add_argument("--input_dir", type=str, help="Directory of input images")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            root = tk.Tk()
            root.withdraw()
            input_dir = filedialog.askdirectory(title="Select directory of input images")
            if not input_dir:
                messagebox.showinfo("No directory selected", "No directory selected. Exiting.")
                return
        except Exception as e:
            print(f"Error with GUI selection: {e}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load multiple CLIP models for ensemble approach
    models = load_multiple_clip_models(device)
    print(f"Loaded {len(models)} CLIP models")

    images, paths = load_images_from_folder(input_dir)
    if len(images) < 2:
        print("Need at least 2 images in the directory.")
        return
    
    print(f"Found {len(images)} images")

    # Compute ensemble features from multiple models
    features = compute_ensemble_features(images, models, device)
    
    # Use advanced similarity computation
    sim_matrix = compute_advanced_similarity(features, method='ensemble')

    print("Advanced similarity matrix:")
    print(sim_matrix)

    # Plot the full similarity matrix first
    plot_similarity_matrix(sim_matrix, paths)
    
    # Then plot the top similar pairs with larger thumbnails
    plot_top_pairs(sim_matrix, paths, top_k=8)

if __name__ == "__main__":
    main()
