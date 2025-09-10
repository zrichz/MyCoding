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
IMG_SIZE = 64
N_IMAGES = 600
MIN_RINGS = 5
MAX_RINGS = 15
DATASET_DIR = "ring_dataset"

# --- 2. MODEL CONFIGURATION ---
LATENT_DIM = 64  # Change this value to adjust the latent space dimension

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
        """Generate and save ring images to disk"""
        for i in range(n_images):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            n_rings = random.randint(min_rings, max_rings)
            
            for _ in range(n_rings):
                # Random center (can be outside image)
                cx = random.uniform(-img_size, 2*img_size)
                cy = random.uniform(-img_size, 2*img_size)
                r = random.uniform(img_size*0.2, img_size*0.8)
                thickness = random.uniform(1.0, 7.0)
                y, x = np.ogrid[:img_size, :img_size]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                ring = ((dist >= r - thickness/2) & (dist <= r + thickness/2)).astype(np.float32)
                img = np.maximum(img, ring)  # allow overlap
            
            # Save as PNG
            img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
            img_pil.save(os.path.join(self.dataset_dir, f"ring_{i:04d}.png"))
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_images} images...")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image from disk
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        return torch.tensor(img_array[None, :, :])  # shape: (1, H, W)

def show_examples(dataset, n=8):
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(dataset[i][0].numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Sample Images', fontsize=16)
    plt.show()

# --- 2. AUTOENCODER MODEL ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128*4*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),  # 64x64
            nn.Sigmoid()
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
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

# --- 3. TRAINING ---
def train_autoencoder(model, dataloader, n_epochs=80, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
    return losses

# --- 4. VISUALIZATION & STATS ---
def show_reconstructions(model, dataset, n=8, device='cpu'):
    model.eval()
    idxs = np.random.choice(len(dataset), n, replace=False)
    imgs = torch.stack([dataset[i] for i in idxs]).to(device)
    with torch.no_grad():
        recons = model(imgs).cpu()
    plt.figure(figsize=(16, 8))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(imgs[i,0].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.subplot(2, n, n+i+1)
        plt.imshow(recons[i,0].numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Originals (top) and Reconstructions (bottom)', fontsize=16)
    plt.show()

def show_from_noise(model, latent_dim=LATENT_DIM, n=8, device='cpu'):
    model.eval()
    z = torch.randn(n, latent_dim).to(device)
    with torch.no_grad():
        imgs = model.decode(z).cpu()
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(imgs[i,0].numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Generated from Noise (Decoder Only)', fontsize=16)
    plt.show()

def plot_loss(losses):
    plt.figure(figsize=(12, 8))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('BCE Loss', fontsize=14)
    plt.title('Training Loss', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def save_model(model, filepath="ring_autoencoder.pth"):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'Autoencoder',
        'latent_dim': LATENT_DIM
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath="ring_autoencoder.pth", latent_dim=LATENT_DIM):
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

def dataset_stats(dataset):
    """Print dataset statistics"""
    print(f"\n--- DATASET STATISTICS ---")
    print(f"Total images: {len(dataset)}")
    print(f"Image size: {dataset.img_size}x{dataset.img_size}")
    print(f"Dataset directory: {dataset.dataset_dir}")
    
    # Sample a few images to check statistics
    sample_imgs = [dataset[i][0].numpy() for i in range(min(10, len(dataset)))]
    sample_means = [img.mean() for img in sample_imgs]
    sample_stds = [img.std() for img in sample_imgs]
    
    print(f"Sample mean intensity: {np.mean(sample_means):.4f} ± {np.std(sample_means):.4f}")
    print(f"Sample std intensity: {np.mean(sample_stds):.4f} ± {np.std(sample_stds):.4f}")
    print("-------------------------\n")

# --- 4. NEURAL NETWORK VISUALIZATION ---
def get_latent_reshape(latent_dim):
    """Get appropriate reshape dimensions for latent vector visualization"""
    if latent_dim == 1:
        return (1, 1)
    elif latent_dim == 2:
        return (1, 2)
    elif latent_dim == 4:
        return (2, 2)
    elif latent_dim == 8:
        return (2, 4)
    elif latent_dim == 16:
        return (4, 4)
    elif latent_dim == 32:
        return (4, 8)
    elif latent_dim == 64:
        return (8, 8)
    else:
        # For other dimensions, try to find a reasonable rectangular shape
        import math
        sqrt_dim = int(math.sqrt(latent_dim))
        if sqrt_dim * sqrt_dim == latent_dim:
            return (sqrt_dim, sqrt_dim)
        else:
            # Find factors closest to square
            for i in range(sqrt_dim, 0, -1):
                if latent_dim % i == 0:
                    return (i, latent_dim // i)
            return (1, latent_dim)  # Fallback to 1D

def visualize_encoder_layers(model, dataset, device='cpu', img_idx=0):
    """Visualize feature maps through encoder layers"""
    model.eval()
    
    # Get a sample image
    sample_img = dataset[img_idx].unsqueeze(0).to(device)  # Add batch dimension
    
    # Hook to capture intermediate outputs
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for each layer
    hooks = []
    layer_names = []
    for i, layer in enumerate(model.encoder.net):
        if isinstance(layer, (nn.Conv2d, nn.ReLU)):
            name = f"layer_{i}_{layer.__class__.__name__}"
            layer_names.append(name)
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model.encoder(sample_img)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Show original image
    plt.subplot(4, 8, 1)
    plt.imshow(sample_img[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Original\n64x64', fontsize=10)
    plt.axis('off')
    
    # Plot feature maps for each layer
    plot_idx = 2
    for layer_name in layer_names:
        if layer_name in activations:
            feature_maps = activations[layer_name][0]  # Remove batch dimension
            
            if len(feature_maps.shape) == 3:  # Conv layer output
                n_channels = min(8, feature_maps.shape[0])  # Show max 8 channels
                h, w = feature_maps.shape[1], feature_maps.shape[2]
                
                for ch in range(n_channels):
                    if plot_idx <= 32:  # 4x8 grid
                        plt.subplot(4, 8, plot_idx)
                        plt.imshow(feature_maps[ch].cpu().numpy(), cmap='viridis')
                        if ch == 0:
                            plt.title(f'{layer_name}\nCh {ch+1}/{feature_maps.shape[0]}\n{h}x{w}', fontsize=8)
                        else:
                            plt.title(f'Ch {ch+1}', fontsize=8)
                        plt.axis('off')
                        plot_idx += 1
    
    plt.suptitle('Encoder Feature Maps Progression', fontsize=8)
    # plt.tight_layout()
    plt.show()
    
    return activations

def visualize_decoder_layers(model, latent_vector, device='cpu'):
    """Visualize feature maps through decoder layers"""
    model.eval()
    
    if latent_vector is None:
        # Use random latent vector
        latent_vector = torch.randn(1, LATENT_DIM).to(device)
    
    # Hook to capture intermediate outputs
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for decoder layers
    hooks = []
    layer_names = []
    
    # Hook the fully connected layer
    hooks.append(model.decoder.fc.register_forward_hook(get_activation('fc_output')))
    layer_names.append('fc_output')
    
    # Hook the conv transpose layers
    for i, layer in enumerate(model.decoder.net):
        if isinstance(layer, (nn.ConvTranspose2d, nn.ReLU, nn.Sigmoid)):
            name = f"layer_{i}_{layer.__class__.__name__}"
            layer_names.append(name)
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model.decoder(latent_vector)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Show latent vector as heatmap
    plt.subplot(4, 8, 1)
    h, w = get_latent_reshape(LATENT_DIM)
    latent_2d = latent_vector[0].cpu().numpy().reshape(h, w)
    plt.imshow(latent_2d, cmap='RdBu', aspect='auto')
    plt.title(f'Latent Vector\n({LATENT_DIM}D→{h}x{w})', fontsize=10)
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    
    # Plot feature maps for each layer
    plot_idx = 2
    for layer_name in layer_names:
        if layer_name in activations:
            feature_maps = activations[layer_name][0]  # Remove batch dimension
            
            if len(feature_maps.shape) == 3:  # Conv layer output
                n_channels = min(6, feature_maps.shape[0])  # Show max 6 channels for decoder
                h, w = feature_maps.shape[1], feature_maps.shape[2]
                
                for ch in range(n_channels):
                    if plot_idx <= 32:  # 4x8 grid
                        plt.subplot(4, 8, plot_idx)
                        plt.imshow(feature_maps[ch].cpu().numpy(), cmap='viridis')
                        if ch == 0:
                            plt.title(f'{layer_name}\nCh {ch+1}/{feature_maps.shape[0]}\n{h}x{w}', fontsize=8)
                        else:
                            plt.title(f'Ch {ch+1}', fontsize=8)
                        plt.axis('off')
                        plot_idx += 1
            elif len(feature_maps.shape) == 1:  # FC layer output
                plt.subplot(4, 8, plot_idx)
                fc_2d = feature_maps.cpu().numpy().reshape(64, 32)  # 2048 = 64*32 for 64x64 images
                plt.imshow(fc_2d, cmap='viridis', aspect='auto')
                plt.title(f'{layer_name}\n2048D→64x32', fontsize=8)
                plt.axis('off')
                plot_idx += 1
    
    # Show final output
    if plot_idx <= 32:
        plt.subplot(4, 8, plot_idx)
        plt.imshow(output[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Final Output\n64x64', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Decoder Feature Maps Progression', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return activations, output

def visualize_complete_autoencoder(model, dataset, device='cpu', img_idx=0):
    """Complete visualization showing encoding → latent → decoding process"""
    model.eval()
    
    # Get sample image
    sample_img = dataset[img_idx].unsqueeze(0).to(device)
    
    # Get intermediate representations
    with torch.no_grad():
        latent_code = model.encoder(sample_img)
        reconstructed = model.decoder(latent_code)
    
    # Create mega-visualization
    fig = plt.figure(figsize=(24, 16))
    
    # Original image (large)
    plt.subplot(4, 6, 1)
    plt.imshow(sample_img[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Original Image\n64x64', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Encoder progression (simplified)
    encoder_outputs = {}
    def capture_encoder(name):
        def hook(model, input, output):
            encoder_outputs[name] = output.detach()
        return hook
    
    # Hook key encoder layers
    hooks = []
    hooks.append(model.encoder.net[0].register_forward_hook(capture_encoder('conv1')))  # 16 channels
    hooks.append(model.encoder.net[2].register_forward_hook(capture_encoder('conv2')))  # 32 channels
    hooks.append(model.encoder.net[4].register_forward_hook(capture_encoder('conv3')))  # 64 channels
    hooks.append(model.encoder.net[6].register_forward_hook(capture_encoder('conv4')))  # 128 channels
    
    with torch.no_grad():
        _ = model.encoder(sample_img)
    
    for hook in hooks:
        hook.remove()
    
    # Show encoder layers (2-3 channels each)
    conv_names = ['conv1', 'conv2', 'conv3', 'conv4']
    channels_to_show = [3, 3, 3, 3]
    
    for i, (conv_name, n_ch) in enumerate(zip(conv_names, channels_to_show)):
        feature_map = encoder_outputs[conv_name][0]
        for ch in range(min(n_ch, feature_map.shape[0])):
            subplot_idx = 7 + i * 3 + ch
            if subplot_idx <= 18:  # First 3 rows
                plt.subplot(4, 6, subplot_idx)
                plt.imshow(feature_map[ch].cpu().numpy(), cmap='viridis')
                if ch == 0:
                    plt.title(f'{conv_name.upper()}\n{feature_map.shape[1]}x{feature_map.shape[2]}', fontsize=10)
                plt.axis('off')
    
    # Latent space visualization
    plt.subplot(4, 6, 19)
    h, w = get_latent_reshape(LATENT_DIM)
    latent_2d = latent_code[0].cpu().numpy().reshape(h, w)
    im = plt.imshow(latent_2d, cmap='RdBu', aspect='auto')
    plt.title(f'Latent Code\n{LATENT_DIM}D ({h}x{w})', fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.colorbar(im, shrink=0.6)
    
    # Decoder progression
    decoder_outputs = {}
    def capture_decoder(name):
        def hook(model, input, output):
            decoder_outputs[name] = output.detach()
        return hook
    
    # Hook key decoder layers
    hooks = []
    hooks.append(model.decoder.net[0].register_forward_hook(capture_decoder('deconv1')))  # 64 channels
    hooks.append(model.decoder.net[2].register_forward_hook(capture_decoder('deconv2')))  # 32 channels
    hooks.append(model.decoder.net[4].register_forward_hook(capture_decoder('deconv3')))  # 16 channels
    
    with torch.no_grad():
        _ = model.decoder(latent_code)
    
    for hook in hooks:
        hook.remove()
    
    # Show decoder layers
    deconv_names = ['deconv1', 'deconv2', 'deconv3']
    for i, conv_name in enumerate(deconv_names):
        feature_map = decoder_outputs[conv_name][0]
        subplot_idx = 20 + i
        if subplot_idx <= 24:
            plt.subplot(4, 6, subplot_idx)
            # Show average across channels for cleaner visualization
            avg_feature = feature_map.mean(dim=0).cpu().numpy()
            plt.imshow(avg_feature, cmap='viridis')
            plt.title(f'{conv_name.upper()}\n{feature_map.shape[1]}x{feature_map.shape[2]}\n(avg)', fontsize=10)
            plt.axis('off')
    
    # Reconstructed image (large)
    plt.subplot(4, 6, 6)
    plt.imshow(reconstructed[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Reconstructed\n64x64', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Difference map
    plt.subplot(4, 6, 12)
    diff = torch.abs(sample_img[0, 0] - reconstructed[0, 0]).cpu().numpy()
    plt.imshow(diff, cmap='hot')
    plt.title('Difference\n(Error Map)', fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.colorbar(shrink=0.6)
    
    plt.suptitle(f'Complete Autoencoder Process - Image #{img_idx}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    mse = torch.mean((sample_img - reconstructed) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Latent code range: [{latent_code.min().item():.3f}, {latent_code.max().item():.3f}]")
    print(f"Latent code mean: {latent_code.mean().item():.3f}, std: {latent_code.std().item():.3f}")

def main():
    # 1. Create dataset
    print("=== RING DATASET AUTOENCODER ===")
    dataset = RingsDataset()
    dataset_stats(dataset)
    show_examples(dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Build model
    latent_dim = LATENT_DIM
    model = Autoencoder(latent_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Latent dimension: {latent_dim}")
    
    # 3. Train
    device = torch.device('cpu')  # Force CPU usage due to old GPU
    print(f"Using device: {device}")
    losses = train_autoencoder(model, dataloader, n_epochs=30, lr=1e-3, device=device)
    plot_loss(losses)
    
    # 4. Visualize reconstructions
    show_reconstructions(model, dataset, n=8, device=device)
    
    # 5. Visualize decoder from noise
    show_from_noise(model, latent_dim=latent_dim, n=8, device=device)
    
    # 6. Visualize neural network internals
    print("\n=== NEURAL NETWORK ANALYSIS ===")
    # Show complete autoencoder process for a few different images
    for img_idx in [0, 1, 2]:
        visualize_complete_autoencoder(model, dataset, device=device, img_idx=img_idx)
    
    # Show detailed encoder analysis
    print("Analyzing encoder layers...")
    visualize_encoder_layers(model, dataset, device=device, img_idx=0)
    
    # Show detailed decoder analysis with different latent vectors
    print("Analyzing decoder layers...")
    # Use the encoded latent of image 0
    sample_img = dataset[0].unsqueeze(0).to(device)
    with torch.no_grad():
        real_latent = model.encoder(sample_img)
    visualize_decoder_layers(model, real_latent, device=device)
    
    # Also show decoder with random latent
    print("Decoder with random latent vector...")
    visualize_decoder_layers(model, None, device=device)  # None = random latent
    
    # 7. Save model
    save_model(model)
    
    # 8. Print final stats
    print(f"\n=== FINAL RESULTS ===")
    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"Best training loss: {min(losses):.4f}")
    print(f"Loss improvement: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    print(f"Latent space dim: {latent_dim}")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Model parameters: {total_params:,}")
    print("=====================")

if __name__ == "__main__":
    main()
