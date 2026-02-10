# vanilla autoencoder with Gradio interface - MULTI-IMAGE version
# adapted from Mike X Cohen code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import io
from matplotlib.figure import Figure
import os
import glob
from torch.utils.data import Dataset, DataLoader

#-----------------------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Dataset for loading multiple images"""
    def __init__(self, image_paths, target_size):
        self.image_paths = image_paths
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.target_size, self.target_size))
        img_array = np.array(image).flatten()
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return torch.tensor(img_array, dtype=torch.float32)


def create_autoencoder(input_size, hidden1=None, hidden2=None):
    """
    Create an autoencoder model dynamically based on input size
    """
    # Use provided sizes or calculate defaults
    if hidden1 is None:
        hidden1 = max(128, input_size // 500)
    if hidden2 is None:
        hidden2 = max(8, hidden1 // 26)
    
    class aenet(nn.Module):
        def __init__(self, input_dim, h1, h2):
            super().__init__()
            self.input = nn.Linear(input_dim, h1)
            self.enc = nn.Linear(h1, h2)
            self.lat = nn.Linear(h2, h1)
            self.dec = nn.Linear(h1, input_dim)

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.enc(x))
            x = F.relu(self.lat(x))
            y = torch.sigmoid(self.dec(x))
            return y

    # Create the model instance
    net = aenet(input_size, hidden1, hidden2)
    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)

    return net, lossfun, optimizer


def add_noise_to_images(img_batch, noise_level=0.8):
    """
    Add Gaussian noise to images while retaining original information
    """
    # Add Gaussian noise to original image (retains some structure)
    noise = torch.randn_like(img_batch) * noise_level
    noisy = img_batch + noise
    noisy = torch.clamp(noisy, 0, 1)  # Keep values in valid range
    return noisy


def fig_to_pil(fig):
    """
    Convert matplotlib figure to PIL Image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    result = img.copy()
    buf.close()
    plt.close(fig)
    return result


def load_images_from_folder(folder_path, target_size):
    """
    Load all images from a folder
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_paths:
        return None, None, f"No images found in {folder_path}"
    
    # Sort for consistent ordering
    image_paths.sort()
    
    # Load all images
    images = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((target_size, target_size))
            img_array = np.array(image).flatten()
            img_array = img_array / 255.0
            images.append(torch.tensor(img_array, dtype=torch.float32))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not images:
        return None, None, "Failed to load any images"
    
    return torch.stack(images), image_paths


def process_images(folder_path, target_size, epochs, batch_size, hidden1, hidden2):
    """
    Main processing function for multiple images - no progress parameter
    """
    if not folder_path or not os.path.exists(folder_path):
        yield "Please provide a valid folder path.", None, None, None, None, "0%", gr.Slider(minimum=1, maximum=1, value=1, step=1)
        return
    
    # Load images
    images_tensor, image_paths = load_images_from_folder(folder_path, target_size)
    
    if images_tensor is None:
        yield f"Error: No valid images found in {folder_path}", None, None, None, None, "0%", gr.Slider(minimum=1, maximum=1, value=1, step=1)
        return
    
    num_images = len(images_tensor)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(images_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_size = images_tensor.shape[1]
    net, lossfun, optimizer = create_autoencoder(input_size, hidden1, hidden2)
    
    # Pre-generate noisy versions of all images
    noisy_images = add_noise_to_images(images_tensor)
    
    # Training loop with intermediate updates
    import time
    last_update_time = time.time()
    update_interval = 5  # seconds
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_data in dataloader:
            batch_imgs = batch_data[0]
            
            # Forward pass
            outputs = net(batch_imgs)
            loss = lossfun(outputs, batch_imgs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        current_time = time.time()
        
        # Yield intermediate results every 5 seconds
        if current_time - last_update_time >= update_interval or epoch == epochs - 1:
            # Generate denoised images for all data
            with torch.no_grad():
                denoised_images = net(noisy_images)
            
            # Create status message
            status = f"🔄 Training in progress...\n" if epoch < epochs - 1 else "✓ Training complete!\n"
            status += f"Epoch: {epoch + 1}/{epochs}\n"
            status += f"Loss: {avg_loss:.6f}\n"
            status += f"Images: {num_images}\n"
            status += f"Image size: {target_size}x{target_size}\n"
            status += f"Batch size: {batch_size}\n"
            status += f"Parameters: {sum(p.numel() for p in net.parameters()):,}"
            
            # Create loss plot
            loss_fig = Figure(figsize=(10, 5))
            ax = loss_fig.subplots(1, 1)
            ax.plot(losses, '.-', linewidth=1, markersize=3)
            ax.set_xlabel('Epochs', fontsize=12)
            ax.set_ylabel('Model Loss (MSE)', fontsize=12)
            ax.set_yscale('log')
            ax.set_title(f'Training Loss - Epoch {epoch + 1}/{epochs}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            loss_fig.tight_layout()
            loss_plot = fig_to_pil(loss_fig)
            
            # Show first image by default
            img_idx = 0
            original_img = create_single_image_display(images_tensor[img_idx], target_size, "Original")
            noisy_img = create_single_image_display(noisy_images[img_idx], target_size, "Noisy")
            denoised_img = create_single_image_display(denoised_images[img_idx], target_size, f"Denoised (Epoch {epoch + 1})")
            
            # Progress text
            progress_text = f"Epoch {epoch + 1}/{epochs} ({100*(epoch+1)//epochs}%)"
            
            # Store for later viewing
            global stored_results
            stored_results = {
                'original': images_tensor,
                'noisy': noisy_images,
                'denoised': denoised_images,
                'target_size': target_size,
                'num_images': num_images
            }
            
            # Update slider with new maximum
            slider_update = gr.Slider(minimum=1, maximum=num_images, value=1, step=1)
            
            yield status, original_img, noisy_img, denoised_img, loss_plot, progress_text, slider_update
            
            last_update_time = current_time
    
    # Final yield with complete status
    with torch.no_grad():
        denoised_images = net(noisy_images)
    
    status = f"✓ Training complete!\n"
    status += f"Final loss: {losses[-1]:.6f}\n"
    status += f"Total images: {num_images}\n"
    status += f"Image size: {target_size}x{target_size}\n"
    status += f"Epochs trained: {epochs}\n"
    status += f"Batch size: {batch_size}\n"
    status += f"Parameters: {sum(p.numel() for p in net.parameters()):,}"
    
    # Create final loss plot
    loss_fig = Figure(figsize=(10, 5))
    ax = loss_fig.subplots(1, 1)
    ax.plot(losses, '.-', linewidth=1, markersize=3)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Model Loss (MSE)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Training Loss - COMPLETE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    loss_fig.tight_layout()
    loss_plot = fig_to_pil(loss_fig)
    
    # Show first image
    img_idx = 0
    original_img = create_single_image_display(images_tensor[img_idx], target_size, "Original")
    noisy_img = create_single_image_display(noisy_images[img_idx], target_size, "Noisy")
    denoised_img = create_single_image_display(denoised_images[img_idx], target_size, "Denoised (Final)")
    
    # Store for later viewing
    stored_results = {
        'original': images_tensor,
        'noisy': noisy_images,
        'denoised': denoised_images,
        'target_size': target_size,
        'num_images': num_images
    }
    
    progress_text = "Complete!"
    
    # Update slider with final maximum
    slider_update = gr.Slider(minimum=1, maximum=num_images, value=1, step=1)
    
    yield status, original_img, noisy_img, denoised_img, loss_plot, progress_text, slider_update


def create_single_image_display(img_tensor, target_size, title):
    """Create a single image display"""
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots(1, 1)
    ax.imshow(img_tensor.view(target_size, target_size, 3).detach().numpy())
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    return fig_to_pil(fig)



def view_specific_image(image_index):
    """View a specific image from the stored results"""
    global stored_results
    
    if 'original' not in stored_results:
        return None, None, None
    
    idx = int(image_index) - 1  # Convert to 0-based index
    
    if idx < 0 or idx >= len(stored_results['original']):
        return None, None, None
    
    target_size = stored_results['target_size']
    
    original_img = create_single_image_display(stored_results['original'][idx], target_size, f"Original (Image {image_index})")
    noisy_img = create_single_image_display(stored_results['noisy'][idx], target_size, f"Noisy (Image {image_index})")
    denoised_img = create_single_image_display(stored_results['denoised'][idx], target_size, f"Denoised (Image {image_index})")
    
    return original_img, noisy_img, denoised_img


# Global storage for results
stored_results = {}


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Multi-Image Autoencoder Denoiser", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Vanilla Autoencoder - Multi-Image Denoising
        
        Train a neural network autoencoder on multiple images from a folder!
        The model learns to compress and reconstruct your images, then removes white Gaussian noise.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                folder_path = gr.Textbox(
                    label="Folder Path",
                    value="C:/MyCoding/images_general/512x512",
                    info="Path to folder containing images"
                )
                
                target_size = gr.Slider(
                    minimum=32, 
                    maximum=256, 
                    value=64, 
                    step=16,
                    label="Processing Size (pixels)",
                    info="All images will be resized to this"
                )
                
                epochs = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=400,
                    step=50,
                    label="Training Epochs",
                    info="More epochs = better quality (but slower)"
                )
                
                batch_size = gr.Slider(
                    minimum=4,
                    maximum=64,
                    value=16,
                    step=4,
                    label="Batch Size",
                    info="Number of images per training batch"
                )
                
                hidden1_size = gr.Slider(
                    minimum=16,
                    maximum=2048,
                    value=128,
                    step=16,
                    label="Hidden Layer 1 Size",
                    info="First encoding layer neuron count"
                )
                
                hidden2_size = gr.Slider(
                    minimum=1,
                    maximum=512,
                    value=8,
                    step=1,
                    label="Bottleneck Layer Size",
                    info="Compressed representation size"
                )
                
                process_btn = gr.Button("🚀 Train & Denoise", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                image_selector = gr.Slider(
                    minimum=1,
                    maximum=1,
                    value=1,
                    step=1,
                    label="View Image Number",
                    info="Select which image to display"
                )
                
                view_btn = gr.Button("👁️ View Selected Image", variant="secondary")
            
            with gr.Column(scale=2):
                progress_display = gr.Textbox(label="Training Progress", lines=2, interactive=False)
                status_text = gr.Textbox(label="Status", lines=6, interactive=False)
                
                with gr.Row():
                    original_display = gr.Image(label="Original", height=300)
                    noisy_display = gr.Image(label="Noisy", height=300)
                    denoised_display = gr.Image(label="Denoised", height=300)
                
                loss_plot = gr.Image(label="Training Loss Curve", height=300)
        
        # Wire up the training
        process_btn.click(
            fn=process_images,
            inputs=[folder_path, target_size, epochs, batch_size, hidden1_size, hidden2_size],
            outputs=[status_text, original_display, noisy_display, denoised_display, loss_plot, progress_display, image_selector]
        )
        
        # Wire up the image viewer
        view_btn.click(
            fn=view_specific_image,
            inputs=[image_selector],
            outputs=[original_display, noisy_display, denoised_display]
        )
        
        # Also update on slider change
        image_selector.change(
            fn=view_specific_image,
            inputs=[image_selector],
            outputs=[original_display, noisy_display, denoised_display]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)
