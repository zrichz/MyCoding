#!/home/rich/MyCoding/textual_inversions/.venv/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os
from PIL import Image

# --- 1. DATASET GENERATION ---
IMG_SIZE = 32
N_IMAGES = 1200
MIN_RINGS = 5
MAX_RINGS = 15
DATASET_DIR = "ring_dataset"

# --- 2. MODEL CONFIGURATION ---
LATENT_DIM = 512  # Change this value to adjust the latent space dimension

class RingsDataset(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR, generate_new=False, n_images=N_IMAGES, img_size=IMG_SIZE, min_rings=MIN_RINGS, max_rings=MAX_RINGS):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Check if dataset already exists
        existing_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        
        if generate_new or len(existing_files) < n_images:
            print(f"Generating {n_images} new ring images...")
            self._generate_dataset(n_images, img_size, min_rings, max_rings)
        else:
            print(f"Loading existing dataset with {len(existing_files)} images...")
        
        # Load image paths
        self.image_paths = [os.path.join(dataset_dir, f) for f in sorted(os.listdir(dataset_dir)) if f.endswith('.png')]
        
    def _generate_dataset(self, n_images, img_size, min_rings, max_rings):
        """Generate and save colored ring images to disk"""
        for i in range(n_images):
            img = np.zeros((img_size, img_size, 3), dtype=np.float32)  # RGB image
            n_rings = random.randint(min_rings, max_rings)
            
            for _ in range(n_rings):
                # Random center (can be outside image)
                cx = random.uniform(-img_size, 2*img_size)
                cy = random.uniform(-img_size, 2*img_size)
                r = random.uniform(img_size*0.2, img_size*0.8)
                thickness = random.uniform(1.0, 7.0)
                
                # Random bright color for each ring
                color = np.array([
                    random.uniform(0.3, 1.0),  # Red
                    random.uniform(0.3, 1.0),  # Green  
                    random.uniform(0.3, 1.0)   # Blue
                ])
                
                y, x = np.ogrid[:img_size, :img_size]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                ring_mask = ((dist >= r - thickness/2) & (dist <= r + thickness/2))
                
                # Apply color to all three channels
                for c in range(3):
                    img[:, :, c] = np.maximum(img[:, :, c], ring_mask.astype(np.float32) * color[c])
            
            # Save as RGB PNG
            img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
            img_pil.save(os.path.join(self.dataset_dir, f"ring_{i:04d}.png"))
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_images} images...")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load RGB image from disk
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        # Convert from HWC to CHW format for PyTorch
        img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # shape: (3, H, W)
        return img_tensor

# --- 2. AUTOENCODER MODEL ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), # 16x16 -> 8x8  
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*4, latent_dim)  # 4x4x64 = 1024
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64*4*4)  # Match encoder: 64 channels, 4x4
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 16x16 -> 32x32 (RGB output)
            nn.Sigmoid()
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 64, 4, 4)  # Reshape to 64 channels
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)

def show_examples(dataset, n=8):
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        # Convert from CHW to HWC for display
        img_display = dataset[i].permute(1, 2, 0).numpy()
        plt.imshow(img_display)
        plt.axis('off')
    plt.suptitle('Sample 32x32 RGB Ring Images', fontsize=16)
    plt.show()

def show_reconstructions(model, dataset, n=8, device='cpu'):
    model.eval()
    idxs = np.random.choice(len(dataset), n, replace=False)
    imgs = torch.stack([dataset[i] for i in idxs]).to(device)
    with torch.no_grad():
        recons = model(imgs).cpu()
    plt.figure(figsize=(16, 8))
    for i in range(n):
        plt.subplot(2, n, i+1)
        # Convert from CHW to HWC for display
        img_display = imgs[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_display)
        plt.axis('off')
        if i == 0:
            plt.title('Originals (32x32)', fontsize=12)
        plt.subplot(2, n, n+i+1)
        # Convert from CHW to HWC for display
        recon_display = recons[i].permute(1, 2, 0).numpy()
        plt.imshow(recon_display)
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructions (32x32)', fontsize=12)
    plt.suptitle('Original vs Reconstructed 32x32 RGB Images', fontsize=16)
    plt.show()

def show_from_noise(model, latent_dim=LATENT_DIM, n=8, device='cpu'):
    model.eval()
    z = torch.randn(n, latent_dim).to(device)
    with torch.no_grad():
        imgs = model.decode(z).cpu()
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        # Convert from CHW to HWC for display
        img_display = imgs[i].permute(1, 2, 0).numpy()
        plt.imshow(img_display)
        plt.axis('off')
    plt.suptitle('Generated 32x32 RGB Images from Random Noise', fontsize=16)
    plt.show()

def load_model(filepath="ring_autoencoder_32x32.pth", latent_dim=LATENT_DIM):
    """Load trained model"""
    if os.path.exists(filepath):
        model = Autoencoder(latent_dim)
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
        return model
    else:
        print(f"No saved model found at {filepath}")
        return None

def main():
    print("=== 32x32 RGB RING AUTOENCODER TEST ===")
    
    # Load dataset and model
    dataset = RingsDataset()
    print(f"Dataset: {len(dataset)} images of size {dataset.img_size}x{dataset.img_size}")
    
    # Load trained model
    model = load_model()
    if model is None:
        print("No trained model found!")
        return
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Latent dimension: {LATENT_DIM}")
    
    device = torch.device('cpu')
    
    # Show sample images
    print("\n1. Sample dataset images:")
    show_examples(dataset, n=8)
    
    # Show reconstructions
    print("\n2. Reconstruction quality:")
    show_reconstructions(model, dataset, n=8, device=device)
    
    # Show generations from noise
    print("\n3. Generation from random noise:")
    show_from_noise(model, latent_dim=LATENT_DIM, n=8, device=device)
    
    # Test a specific example
    print("\n4. Single image analysis:")
    test_img = dataset[42].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        latent = model.encode(test_img)
        reconstructed = model.decode(latent)
        mse = torch.mean((test_img - reconstructed) ** 2).item()
        
    print(f"Test image MSE: {mse:.6f}")
    print(f"Latent code stats: min={latent.min().item():.3f}, max={latent.max().item():.3f}, mean={latent.mean().item():.3f}")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(test_img[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())
    plt.title('Reconstructed')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    diff = torch.abs(test_img[0] - reconstructed[0]).mean(dim=0).cpu().numpy()
    plt.imshow(diff, cmap='hot')
    plt.title('Absolute Difference')
    plt.axis('off')
    plt.colorbar()
    
    plt.suptitle(f'32x32 Single Image Test (MSE: {mse:.6f})')
    plt.tight_layout()
    plt.show()
    
    print("\n=== SUCCESS: 32x32 RGB Autoencoder Working! ===")

if __name__ == "__main__":
    main()
