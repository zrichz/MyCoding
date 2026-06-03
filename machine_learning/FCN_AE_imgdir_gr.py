import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io
from PIL import Image

# Global variables
pix_per_side = 64
trained_net = None
trained_decoder = None
training_dataset = None


# Custom dataset that works with flat directories or subdirectories
class FlexibleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        
        # Support common image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # Recursively find all images
        for ext in image_extensions:
            self.image_paths.extend(self.root_dir.rglob(f'*{ext}'))
            self.image_paths.extend(self.root_dir.rglob(f'*{ext.upper()}'))
        
        self.image_paths = sorted(self.image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label for compatibility


# Variational Autoencoder (VAE) - properly structured latent space
class VAENet(nn.Module):
    def __init__(self, pix_size, latent_dim=16):
        super().__init__()
        self.pix_size = pix_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = nn.Linear(pix_size*pix_size, 600)
        self.enc2 = nn.Linear(600, 300)
        
        # Latent space - mean and log variance
        self.fc_mu = nn.Linear(300, latent_dim)
        self.fc_logvar = nn.Linear(300, latent_dim)
        
        # Decoder
        self.dec1 = nn.Linear(latent_dim, 300)
        self.dec2 = nn.Linear(300, 600)
        self.dec3 = nn.Linear(600, pix_size*pix_size)
    
    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Standalone decoder for generation
class VAEDecoder(nn.Module):
    def __init__(self, pix_size, latent_dim=16):
        super().__init__()
        self.pix_size = pix_size
        self.latent_dim = latent_dim
        
        self.dec1 = nn.Linear(latent_dim, 300)
        self.dec2 = nn.Linear(300, 600)
        self.dec3 = nn.Linear(600, pix_size*pix_size)
    
    def forward(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = Reconstruction loss + KL divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence


def train_model(image_dir, image_size, num_epochs, batch_size, learning_rate, beta_kl, progress=gr.Progress()):
    global trained_net, trained_decoder, pix_per_side, training_dataset
    
    pix_per_side = image_size
    
    # Check if directory exists
    if not Path(image_dir).exists():
        return None, "Error: Directory does not exist. Please provide a valid directory path."
    
    try:
        # Define transformation
        transform = transforms.Compose([
            transforms.Resize((pix_per_side, pix_per_side)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load dataset - works with flat or nested directories
        progress(0, desc="Loading images...")
        dataset = FlexibleImageDataset(image_dir, transform=transform)
        
        print(f"\nDataset loading: Found {len(dataset)} images in {image_dir}")
        
        if len(dataset) == 0:
            return None, "Error: No images found in the directory."
        
        # Adjust batch size if necessary
        actual_batch_size = min(batch_size, len(dataset))
        if actual_batch_size < batch_size:
            print(f"Warning: Batch size reduced from {batch_size} to {actual_batch_size} (only {len(dataset)} images available)")
        
        training_dataset = dataset  # Save for reconstruction
        dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)
        
        # Verify dataloader has data
        num_batches = len(dataloader)
        if num_batches == 0:
            return None, f"Error: No batches possible. Found {len(dataset)} images."
        
        print(f"Using batch size: {actual_batch_size}, Total batches per epoch: {num_batches}")
        
        # Create VAE model
        progress(0.1, desc=f"Creating VAE model... {num_batches} batches per epoch")
        net = VAENet(pix_per_side, latent_dim=16)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        
        # Training
        losses = []
        recon_losses = []
        kl_losses = []
        
        print(f"\nStarting training: {num_epochs} epochs, {num_batches} batches per epoch")
        
        for epoch in range(num_epochs):
            progress((epoch + 1) / num_epochs, desc=f"Training epoch {epoch+1}/{num_epochs}")
            
            epoch_losses = []
            epoch_recon = []
            epoch_kl = []
            
            batch_count = 0
            for images, _ in dataloader:
                images = images.view(images.size(0), -1)
                
                # Forward pass
                recon_batch, mu, logvar = net(images)
                
                # Calculate losses
                recon_loss = F.mse_loss(recon_batch, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta_kl * kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item() / len(images))
                epoch_recon.append(recon_loss.item() / len(images))
                epoch_kl.append(kl_loss.item() / len(images))
                batch_count += 1
            
            avg_loss = np.mean(epoch_losses)
            avg_recon = np.mean(epoch_recon)
            avg_kl = np.mean(epoch_kl)
            
            losses.append(avg_loss)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Batches={batch_count}")
        
        print(f"Training complete. Final loss: {losses[-1]:.4f}")
        
        # Create standalone decoder
        decoder = VAEDecoder(pix_per_side, latent_dim=16)
        decoder.dec1.weight.data = net.dec1.weight.data
        decoder.dec1.bias.data = net.dec1.bias.data
        decoder.dec2.weight.data = net.dec2.weight.data
        decoder.dec2.bias.data = net.dec2.bias.data
        decoder.dec3.weight.data = net.dec3.weight.data
        decoder.dec3.bias.data = net.dec3.bias.data
        
        # Save trained models
        trained_net = net
        trained_decoder = decoder
        
        # Create loss plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(losses, '.-', label='Total Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(recon_losses, '.-', label='Reconstruction Loss', linewidth=2)
        ax2.plot(kl_losses, '.-', label='KL Divergence', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        loss_plot = Image.open(buf)
        plt.close(fig)
        
        final_loss = losses[-1]
        message = (f"Training completed successfully.\n"
                  f"Total epochs trained: {num_epochs}\n"
                  f"Images found: {len(dataset)}\n"
                  f"Batch size used: {actual_batch_size}\n"
                  f"Batches per epoch: {num_batches}\n"
                  f"Total batches processed: {num_epochs * num_batches}\n"
                  f"Final loss: {final_loss:.4f}\n"
                  f"Final reconstruction loss: {recon_losses[-1]:.4f}\n"
                  f"Final KL divergence: {kl_losses[-1]:.4f}\n"
                  f"Image size: {pix_per_side}x{pix_per_side}")
        
        return loss_plot, message
        
    except Exception as e:
        import traceback
        error_msg = f"Error during training: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def generate_images(num_images, seed, sample_method):
    global trained_decoder, pix_per_side
    
    if trained_decoder is None:
        return None, "Error: Please train the model first before generating images."
    
    try:
        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
        
        # Generate images from latent space
        generated = []
        for _ in range(num_images):
            if sample_method == "Normal (0,1)":
                # Sample from standard normal distribution (proper for VAE)
                latent = torch.randn(16)
            else:
                # Sample from uniform [0,1] distribution
                latent = torch.rand(16)
            
            output = trained_decoder(latent)
            img = output.view(pix_per_side, pix_per_side).detach().numpy()
            generated.append(img)
        
        # Create visualization
        cols = min(6, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        
        if num_images == 1:
            axs = np.array([[axs]])
        elif rows == 1:
            axs = axs.reshape(1, -1)
        elif cols == 1:
            axs = axs.reshape(-1, 1)
        
        for idx in range(num_images):
            row = idx // cols
            col = idx % cols
            axs[row, col].imshow(generated[idx], cmap='gray')
            axs[row, col].axis('off')
        
        # Hide empty subplots
        for idx in range(num_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf)
        plt.close(fig)
        
        seed_info = f"Seed: {seed}" if seed >= 0 else "Seed: Random"
        message = (f"Generated {num_images} images from random latent vectors.\n"
                  f"{seed_info}\n"
                  f"Sampling: {sample_method}")
        
        return result_img, message
        
    except Exception as e:
        return None, f"Error during generation: {str(e)}"


def reconstruct_images(num_images, seed):
    global trained_net, training_dataset, pix_per_side
    
    if trained_net is None or training_dataset is None:
        return None, "Error: Please train the model first."
    
    try:
        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Randomly select images from training set
        indices = np.random.choice(len(training_dataset), min(num_images, len(training_dataset)), replace=False)
        
        originals = []
        reconstructions = []
        
        trained_net.eval()
        with torch.no_grad():
            for idx in indices:
                # Get original image
                img_tensor, _ = training_dataset[idx]
                img_flat = img_tensor.view(1, -1)
                
                # Reconstruct through VAE
                recon, mu, logvar = trained_net(img_flat)
                
                # Convert to numpy
                original = img_tensor.squeeze().numpy()
                reconstructed = recon.view(pix_per_side, pix_per_side).numpy()
                
                originals.append(original)
                reconstructions.append(reconstructed)
        
        # Create comparison visualization
        cols = min(6, num_images)
        rows = ((num_images * 2) + cols - 1) // cols
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        
        if rows == 1:
            axs = axs.reshape(1, -1)
        
        for idx in range(num_images):
            # Original image
            orig_row = (idx * 2) // cols
            orig_col = (idx * 2) % cols
            axs[orig_row, orig_col].imshow(originals[idx], cmap='gray')
            axs[orig_row, orig_col].set_title('Original', fontsize=8)
            axs[orig_row, orig_col].axis('off')
            
            # Reconstructed image
            recon_row = (idx * 2 + 1) // cols
            recon_col = (idx * 2 + 1) % cols
            axs[recon_row, recon_col].imshow(reconstructions[idx], cmap='gray')
            axs[recon_row, recon_col].set_title('Reconstructed', fontsize=8)
            axs[recon_row, recon_col].axis('off')
        
        # Hide empty subplots
        for idx in range(num_images * 2, rows * cols):
            row = idx // cols
            col = idx % cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf)
        plt.close(fig)
        
        seed_info = f"Seed: {seed}" if seed >= 0 else "Seed: Random"
        message = (f"Reconstructed {num_images} training images.\n"
                  f"{seed_info}\n"
                  f"Each pair shows: Original (top/left) vs Reconstructed (bottom/right)")
        
        return result_img, message
        
    except Exception as e:
        return None, f"Error during reconstruction: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="VAE Image Generator") as demo:
    gr.Markdown("# Variational Autoencoder (VAE) Image Generator")
    gr.Markdown("Train a VAE on your images, reconstruct training images, and generate novel images from latent space.")
    
    with gr.Tab("Training"):
        gr.Markdown("## Train the VAE")
        gr.Markdown("Images can be in a flat directory or organized in subdirectories. All common image formats supported.")
        
        with gr.Row():
            with gr.Column():
                train_dir = gr.Textbox(
                    label="Image Directory Path",
                    placeholder="/path/to/images",
                    value="/home/rich/MyCoding/images_general/512x512"
                )
                
                with gr.Row():
                    train_img_size = gr.Slider(
                        minimum=32,
                        maximum=256,
                        value=64,
                        step=16,
                        label="Image Size (pixels per side)"
                    )
                    train_epochs = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=50,
                        step=10,
                        label="Number of Epochs"
                    )
                
                with gr.Row():
                    train_batch_size = gr.Slider(
                        minimum=1,
                        maximum=64,
                        value=8,
                        step=1,
                        label="Batch Size"
                    )
                    train_lr = gr.Slider(
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.001,
                        step=0.0001,
                        label="Learning Rate"
                    )
                
                train_beta = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="KL Divergence Weight (beta)",
                    info="Higher values enforce more structured latent space"
                )
                
                train_btn = gr.Button("Train VAE Model", variant="primary")
            
            with gr.Column():
                train_plot = gr.Image(label="Training Loss")
                train_status = gr.Textbox(label="Status", lines=8)
        
        train_btn.click(
            fn=train_model,
            inputs=[train_dir, train_img_size, train_epochs, train_batch_size, train_lr, train_beta],
            outputs=[train_plot, train_status]
        )
    
    with gr.Tab("Reconstruct Images"):
        gr.Markdown("## Reconstruct Training Images")
        gr.Markdown("See how well the VAE reconstructs actual training images. This shows the quality of the learned encoding.")
        
        with gr.Row():
            with gr.Column():
                recon_num = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=1,
                    label="Number of Images to Reconstruct"
                )
                recon_seed = gr.Number(
                    value=-1,
                    label="Random Seed (use -1 for random)"
                )
                recon_btn = gr.Button("Reconstruct Images", variant="primary")
            
            with gr.Column():
                recon_output = gr.Image(label="Original vs Reconstructed")
                recon_status = gr.Textbox(label="Status", lines=3)
        
        recon_btn.click(
            fn=reconstruct_images,
            inputs=[recon_num, recon_seed],
            outputs=[recon_output, recon_status]
        )
    
    with gr.Tab("Generate Novel Images"):
        gr.Markdown("## Generate New Images from Latent Space")
        gr.Markdown("Generate completely new images by sampling from the learned latent distribution. VAE ensures structured latent space.")
        
        with gr.Row():
            with gr.Column():
                gen_num = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=1,
                    label="Number of Images to Generate"
                )
                gen_seed = gr.Number(
                    value=-1,
                    label="Random Seed (use -1 for random)"
                )
                gen_method = gr.Radio(
                    choices=["Normal (0,1)", "Uniform (0,1)"],
                    value="Normal (0,1)",
                    label="Latent Sampling Method",
                    info="Normal is proper for VAE, Uniform for exploration"
                )
                gen_btn = gr.Button("Generate Images", variant="primary")
            
            with gr.Column():
                gen_output = gr.Image(label="Generated Images")
                gen_status = gr.Textbox(label="Status", lines=3)
        
        gen_btn.click(
            fn=generate_images,
            inputs=[gen_num, gen_seed, gen_method],
            outputs=[gen_output, gen_status]
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## Variational Autoencoder (VAE) for Image Generation
        
        ### What Changed?
        
        **1. No Subdirectories Required**
        - Images can be in a flat directory or nested subdirectories
        - All common formats supported: jpg, png, bmp, gif, tiff
        
        **2. VAE Architecture (Proper Latent Space)**
        - Standard AE replaced with Variational Autoencoder
        - Latent space is now structured and continuous
        - Enables meaningful generation from random noise
        
        **3. Reconstruction Tab**
        - See how well the model reconstructs actual training images
        - Better indicator of model quality than random generation
        
        ### Architecture
        - Encoder: input -> 600 -> 300 -> 16D latent (mean and variance)
        - Reparameterization trick for backpropagation
        - Decoder: 16D -> 300 -> 600 -> output
        - Loss: Reconstruction (MSE) + KL Divergence
        
        ### Why VAE vs Standard AE?
        - Standard AE: Unstructured latent space, random noise produces poor results
        - VAE: Regularized latent space following normal distribution
        - VAE enables meaningful interpolation and generation
        
        ### KL Divergence Weight (beta)
        - Controls trade-off between reconstruction quality and latent space structure
        - Higher beta: More structured latent space, better generation, slightly worse reconstruction
        - Lower beta: Better reconstruction, less structured latent space
        
        ### Tips
        - Start with beta=1.0 for balanced results
        - Use Normal sampling for proper VAE generation
        - Check reconstructions first to verify model quality
        - Use the same seed for reproducible results
        """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
