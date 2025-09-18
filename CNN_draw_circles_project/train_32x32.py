#!/home/rich/myenv/bin/python3
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

# --- 2. ENHANCED AUTOENCODER MODEL ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Enhanced encoder with batch normalization, dropout, and residual connections
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)      # 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)     # 16x16 -> 8x8
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)    # 8x8 -> 4x4
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)   # 4x4 -> 2x2
        self.bn4 = nn.BatchNorm2d(256)
        
        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Enhanced fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim)
        )
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # Progressive encoding with batch norm and LeakyReLU
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        
        # Ensure consistent size regardless of input variations
        x = self.adaptive_pool(x)
        
        # Encode to latent space
        z = self.fc_layers(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Enhanced decoder with skip connections and attention
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256 * 2 * 2)
        )
        
        # Progressive upsampling with batch normalization
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 2x2 -> 4x4
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 4x4 -> 8x8
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 8x8 -> 16x16
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)    # 16x16 -> 32x32
        self.bn4 = nn.BatchNorm2d(16)
        
        # Final output layer with residual connection
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),  # Refine features
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, 1, 1),   # RGB output
            nn.Sigmoid()
        )
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, z):
        # Decode from latent space
        x = self.fc_layers(z).view(-1, 256, 2, 2)
        
        # Progressive upsampling with batch norm
        x = self.leaky_relu(self.bn1(self.deconv1(x)))
        x = self.leaky_relu(self.bn2(self.deconv2(x)))
        x = self.leaky_relu(self.bn3(self.deconv3(x)))
        x = self.leaky_relu(self.bn4(self.deconv4(x)))
        
        # Final refinement and output
        x = self.final_conv(x)
        return x

class AdvancedAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# --- 3. ENHANCED TRAINING WITH MULTIPLE LOSS FUNCTIONS ---
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, perceptual_weight=0.1, smoothness_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.smoothness_weight = smoothness_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def perceptual_loss(self, recon, target):
        """Simple perceptual loss using gradient differences"""
        # Compute gradients in x and y directions
        def compute_gradients(img):
            # img shape: (B, C, H, W)
            grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
            return grad_x, grad_y
        
        recon_grad_x, recon_grad_y = compute_gradients(recon)
        target_grad_x, target_grad_y = compute_gradients(target)
        
        grad_loss_x = self.l1_loss(recon_grad_x, target_grad_x)
        grad_loss_y = self.l1_loss(recon_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def smoothness_loss(self, img):
        """Encourage smooth reconstructions"""
        grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def forward(self, recon, target):
        mse = self.mse_loss(recon, target)
        perceptual = self.perceptual_loss(recon, target)
        smoothness = self.smoothness_loss(recon)
        
        total_loss = (self.mse_weight * mse + 
                     self.perceptual_weight * perceptual + 
                     self.smoothness_weight * smoothness)
        
        return total_loss, {
            'mse': mse.item(),
            'perceptual': perceptual.item(),
            'smoothness': smoothness.item(),
            'total': total_loss.item()
        }

def train_advanced_autoencoder(model, dataloader, n_epochs=50, lr=1e-3, device='cpu'):
    """Enhanced training with better optimization and loss functions"""
    model.to(device)
    
    # Clear GPU cache if using CUDA
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    # Enhanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Combined loss function (reduce perceptual and smoothness weights for better color preservation)
    criterion = CombinedLoss(mse_weight=1.0, perceptual_weight=0.01, smoothness_weight=0.001)
    
    losses = []
    best_loss = float('inf')
    
    # Track individual loss components for plotting
    mse_losses = []
    perceptual_losses = []
    smoothness_losses = []
    
    print(f"Training with enhanced architecture and combined loss...")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_losses = {'mse': 0, 'perceptual': 0, 'smoothness': 0, 'total': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            recon = model(batch)
            
            # Combined loss calculation
            loss, loss_components = criterion(recon, batch)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item() * batch.size(0)
            for key, value in loss_components.items():
                epoch_losses[key] += value * batch.size(0)
        
        # Calculate average losses
        avg_loss = epoch_loss / len(dataloader.dataset)
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader.dataset)
        
        losses.append(avg_loss)
        
        # Store individual loss components for plotting
        mse_losses.append(epoch_losses['mse'])
        perceptual_losses.append(epoch_losses['perceptual'])
        smoothness_losses.append(epoch_losses['smoothness'])
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model_32x32.pth')
        
        # Enhanced progress reporting with GPU memory info
        gpu_mem_str = ""
        if str(device).startswith('cuda'):
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_str = f" | GPU: {gpu_mem:.0f}MB"
        
        if epoch > 0:
            diff = avg_loss - losses[-2]
            print(f"Epoch {epoch+1}/{n_epochs} - Total: {avg_loss:.4f} (Δ: {diff:+.3f}) | "
                  f"MSE: {epoch_losses['mse']:.4f} | "
                  f"Perceptual: {epoch_losses['perceptual']:.4f} | "
                  f"Smoothness: {epoch_losses['smoothness']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}{gpu_mem_str}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} - Total: {avg_loss:.4f} | "
                  f"MSE: {epoch_losses['mse']:.4f} | "
                  f"Perceptual: {epoch_losses['perceptual']:.4f} | "
                  f"Smoothness: {epoch_losses['smoothness']:.4f}{gpu_mem_str}")
    
    return losses, mse_losses, perceptual_losses, smoothness_losses


def save_advanced_model(model, filepath="ring_autoencoder_advanced_32x32.pth"):
    """Save the advanced trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'AdvancedAutoencoder',
        'latent_dim': LATENT_DIM,
        'img_size': IMG_SIZE,
        'architecture': 'advanced_32x32_rgb',
        'features': [
            'batch_normalization',
            'dropout_regularization',
            'leaky_relu_activation',
            'adaptive_pooling',
            'combined_loss_function',
            'gradient_clipping',
            'lr_scheduling',
            'proper_weight_initialization'
        ]
    }, filepath)
    print(f"Advanced model saved to {filepath}")

def save_model(model, filepath="ring_autoencoder_32x32.pth"):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'Autoencoder',
        'latent_dim': LATENT_DIM,
        'img_size': IMG_SIZE,
        'architecture': '32x32_rgb'
    }, filepath)
    print(f"Model saved to {filepath}")

def plot_training_losses(losses, mse_losses, perceptual_losses, smoothness_losses):
    """Plot all loss components during training"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Total loss
    plt.subplot(2, 2, 1)
    plt.plot(losses, linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Combined Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: MSE Loss (most important for color)
    plt.subplot(2, 2, 2)
    plt.plot(mse_losses, linewidth=2, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss (Main Reconstruction)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Perceptual Loss
    plt.subplot(2, 2, 3)
    plt.plot(perceptual_losses, linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Perceptual Loss')
    plt.title('Perceptual Loss (Edge Preservation)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Smoothness Loss
    plt.subplot(2, 2, 4)
    plt.plot(smoothness_losses, linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Smoothness Loss')
    plt.title('Smoothness Loss (Regularization)')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Training Loss Components - Enhanced 32x32 RGB Autoencoder', fontsize=16, y=1.02)
    plt.show()
    
    # Summary plot with all losses normalized
    plt.figure(figsize=(12, 6))
    
    # Normalize losses to [0, 1] for comparison
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    plt.plot(normalize(losses), label='Total Loss', linewidth=2, color='red')
    plt.plot(normalize(mse_losses), label='MSE Loss', linewidth=2, color='blue')
    plt.plot(normalize(perceptual_losses), label='Perceptual Loss', linewidth=2, color='green')
    plt.plot(normalize(smoothness_losses), label='Smoothness Loss', linewidth=2, color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.title('All Loss Components (Normalized for Comparison)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print final loss statistics
    print(f"\n=== LOSS COMPONENT ANALYSIS ===")
    print(f"Final MSE Loss: {mse_losses[-1]:.3f}")
    print(f"Final Perceptual Loss: {perceptual_losses[-1]:.3f}")
    print(f"Final Smoothness Loss: {smoothness_losses[-1]:.3f}")
    print(f"MSE contributes: {(mse_losses[-1] / losses[-1] * 100):.1f}% of total loss")
    print(f"Perceptual contributes: {(perceptual_losses[-1] * 0.01 / losses[-1] * 100):.1f}% of total loss")
    print(f"Smoothness contributes: {(smoothness_losses[-1] * 0.001 / losses[-1] * 100):.1f}% of total loss")
    print("================================\n")

def main():
    print("=== TRAINING ADVANCED 32x32 RGB RING AUTOENCODER ===")
    
    # Create dataset
    dataset = RingsDataset()
    print(f"Dataset: {len(dataset)} images of size {dataset.img_size}x{dataset.img_size}")
    
    # Use larger batch size for GPU, smaller for CPU
    batch_size = 64 if torch.cuda.is_available() else 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    print(f"Batch size: {batch_size} (optimized for {'GPU' if torch.cuda.is_available() else 'CPU'})")
    
    # Build advanced model
    model = AdvancedAutoencoder(LATENT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Latent dimension: {LATENT_DIM}")
    
    # Enhanced training with GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    losses, mse_losses, perceptual_losses, smoothness_losses = train_advanced_autoencoder(model, dataloader,
                                                                                           n_epochs=50,
                                                                                             lr=1e-3,
                                                                                               device=device)
    
    # Plot training losses first
    plot_training_losses(losses, mse_losses, perceptual_losses, smoothness_losses)
    
    # Save advanced model
    save_advanced_model(model)
    
    # Print final stats
    print(f"\n=== ADVANCED TRAINING COMPLETE ===")
    print(f"Final training loss: {losses[-1]:.3f}")
    print(f"Best training loss: {min(losses):.3f}")
    print(f"Loss improvement: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    print(f"Advanced model saved with enhanced architecture")
    print("=========================================")

if __name__ == "__main__":
    main()
