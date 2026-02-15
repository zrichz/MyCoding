# vanilla autoencoder with Gradio interface - works with any image size
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

#-----------------------------------------------------------------------------------------

def create_autoencoder(input_size, hidden1=None, hidden2=None):
    """
    Create an autoencoder model dynamically based on input size
    """
    # Use provided sizes or calculate defaults
    if hidden1 is None:
        hidden1 = max(26, input_size // 500)
    if hidden2 is None:
        hidden2 = max(1, hidden1 // 26)
    
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


def train_model(dataT, numepochs, progress=None, update_callback=None):
    """
    Train the autoencoder model with periodic updates
    """
    import time
    input_size = dataT.shape[1]
    net, lossfun, optimizer = create_autoencoder(input_size)
    losses = []
    
    # Calculate update frequency (every ~5 seconds worth of epochs or every 10 epochs, whichever is smaller)
    last_update_time = time.time()
    update_interval = 5  # seconds

    for epochi in range(numepochs):
        yHat = net(dataT)
        loss = lossfun(yHat, dataT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        # Update progress bar
        if progress:
            progress_value = 0.1 + (0.85 * (epochi + 1) / numepochs)  # Scale from 10% to 95%
            progress(progress_value, 
                    desc=f"Epoch {epochi + 1}/{numepochs} | Loss: {loss.item():.6f}")
        
        # Call update callback periodically
        current_time = time.time()
        if update_callback and (current_time - last_update_time >= update_interval or epochi == numepochs - 1):
            update_callback(epochi + 1, losses, net)
            last_update_time = current_time
    
    return losses, net


def add_noise_to_image(img_tensor, noise_level=0.8):
    """
    Add Gaussian noise to image while retaining original information
    """
    # Add Gaussian noise to original image (retains some structure)
    noise = torch.randn_like(img_tensor) * noise_level
    noisy = img_tensor + noise
    noisy = torch.clamp(noisy, 0, 1)  # Keep values in valid range
    return noisy


def fig_to_pil(fig):
    """
    Convert matplotlib figure to PIL Image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return Image.open(buf)


def process_image(input_image, target_size, epochs, noise_level, hidden1, hidden2, progress=gr.Progress()):
    """
    Main processing function for Gradio interface - yields intermediate results
    """
    if input_image is None:
        yield None, None, "Please upload an image first."
        return
    
    progress(0, desc="Loading image...")
    
    # Load and resize image
    if isinstance(input_image, np.ndarray):
        image = Image.fromarray(input_image.astype('uint8'), 'RGB')
    else:
        image = input_image
    
    # Resize image to target size
    image_resized = image.resize((target_size, target_size))
    
    progress(0.05, desc="Preparing data...")
    
    # Prepare data
    image_flattened = np.array(image_resized).flatten()
    image_flattened = image_flattened / np.max(image_flattened) if np.max(image_flattened) > 0 else image_flattened
    dataT = torch.tensor(image_flattened).float().unsqueeze(0)
    
    progress(0.1, desc="Starting training...")
    
    # Pre-generate noisy version for visualization
    X = add_noise_to_image(dataT, noise_level)
    
    # Create update callback for intermediate results
    def update_callback(current_epoch, losses, net):
        # Generate intermediate denoised result
        with torch.no_grad():
            deOccluded = net(X)
        
        # Create comparison figure
        fig = Figure(figsize=(15, 5))
        axs = fig.subplots(1, 3)
        
        axs[0].imshow(dataT.view(target_size, target_size, 3).detach().numpy())
        axs[0].set_title('Original', fontsize=14, fontweight='bold')
        axs[0].axis('off')
        
        axs[1].imshow(X.view(target_size, target_size, 3).detach().numpy())
        axs[1].set_title('Noisy', fontsize=14, fontweight='bold')
        axs[1].axis('off')
        
        axs[2].imshow(deOccluded.view(target_size, target_size, 3).detach().numpy())
        axs[2].set_title(f'De-noised (Epoch {current_epoch})', fontsize=14, fontweight='bold')
        axs[2].axis('off')
        
        fig.tight_layout()
        result_image = fig_to_pil(fig)
        plt.close(fig)
        
        # Create loss curve
        loss_fig = Figure(figsize=(10, 5))
        ax = loss_fig.subplots(1, 1)
        ax.plot(losses, '.-', linewidth=1, markersize=3)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Model Loss (MSE)', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'Training Loss - Epoch {current_epoch}/{epochs}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        loss_fig.tight_layout()
        loss_image = fig_to_pil(loss_fig)
        plt.close(loss_fig)
        
        # Status message
        status = f"🔄 Training in progress...\n"
        status += f"Current epoch: {current_epoch}/{epochs}\n"
        status += f"Current loss: {losses[-1]:.6f}\n"
        status += f"Image size: {target_size}x{target_size}\n"
        status += f"Noise level: {noise_level:.2f}"
        
        return result_image, loss_image, status
    
    # Train model with callback
    input_size = dataT.shape[1]
    net, lossfun, optimizer = create_autoencoder(input_size, hidden1, hidden2)
    losses = []
    
    import time
    last_update_time = time.time()
    update_interval = 5  # seconds

    for epochi in range(epochs):
        yHat = net(dataT)
        loss = lossfun(yHat, dataT)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
        current_time = time.time()
        
        # Yield intermediate results every 5 seconds - update progress only when yielding
        if current_time - last_update_time >= update_interval:
            progress_value = 0.1 + (0.85 * (epochi + 1) / epochs)
            progress(progress_value, desc=f"Epoch {epochi + 1}/{epochs} | Loss: {loss.item():.6f}")
            result_img, loss_img, status = update_callback(epochi + 1, losses, net)
            yield result_img, loss_img, status
            last_update_time = current_time
    
    progress(0.99, desc="Training complete!")
    
    # Final update with complete status
    with torch.no_grad():
        deOccluded = net(X)
    
    # Create final comparison figure
    fig = Figure(figsize=(15, 5))
    axs = fig.subplots(1, 3)
    
    axs[0].imshow(dataT.view(target_size, target_size, 3).detach().numpy())
    axs[0].set_title('Original', fontsize=14, fontweight='bold')
    axs[0].axis('off')
    
    axs[1].imshow(X.view(target_size, target_size, 3).detach().numpy())
    axs[1].set_title('Noisy', fontsize=14, fontweight='bold')
    axs[1].axis('off')
    
    axs[2].imshow(deOccluded.view(target_size, target_size, 3).detach().numpy())
    axs[2].set_title('De-noised (Final)', fontsize=14, fontweight='bold')
    axs[2].axis('off')
    
    fig.tight_layout()
    result_image = fig_to_pil(fig)
    plt.close(fig)
    
    # Create final loss curve
    loss_fig = Figure(figsize=(10, 5))
    ax = loss_fig.subplots(1, 1)
    ax.plot(losses, '.-', linewidth=1, markersize=3)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Model Loss (MSE)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Training Loss Over Time - COMPLETE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    loss_fig.tight_layout()
    loss_image = fig_to_pil(loss_fig)
    plt.close(loss_fig)
    
    # Final status message
    status = f"✓ Training complete!\n"
    status += f"Final loss: {losses[-1]:.6f}\n"
    status += f"Image size: {target_size}x{target_size}\n"
    status += f"Total parameters: {sum(p.numel() for p in net.parameters()):,}\n"
    status += f"Epochs trained: {epochs}\n"
    status += f"Noise level: {noise_level:.2f}"
    
    # Store model globally for noise generation
    global stored_model, stored_target_size
    stored_model = net
    stored_target_size = target_size
    
    yield result_image, loss_image, status


# Global storage for trained model
stored_model = None
stored_target_size = None


def generate_and_denoise_pure_noise():
    """
    Generate pure RGB white noise and pass it through the trained model
    """
    global stored_model, stored_target_size
    
    if stored_model is None or stored_target_size is None:
        return None, "⚠️ Please train a model first!"
    
    # Generate pure white noise
    noise = torch.rand(1, stored_target_size * stored_target_size * 3)
    
    # Pass through model
    with torch.no_grad():
        reconstructed = stored_model(noise)
    
    # Create comparison figure
    fig = Figure(figsize=(10, 5))
    axs = fig.subplots(1, 2)
    
    axs[0].imshow(noise.view(stored_target_size, stored_target_size, 3).detach().numpy())
    axs[0].set_title('Pure White Noise Input', fontsize=14, fontweight='bold')
    axs[0].axis('off')
    
    axs[1].imshow(reconstructed.view(stored_target_size, stored_target_size, 3).detach().numpy())
    axs[1].set_title('Model Reconstruction', fontsize=14, fontweight='bold')
    axs[1].axis('off')
    
    fig.tight_layout()
    result_image = fig_to_pil(fig)
    plt.close(fig)
    
    status = f"✓ Generated pure white noise and reconstructed\n"
    status += f"Image size: {stored_target_size}x{stored_target_size}\n"
    status += f"Click button again for new random noise pattern"
    
    return result_image, status


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Image Autoencoder Denoiser") as demo:
        gr.Markdown("""
        # Vanilla Autoencoder - Image Denoising
        
        Upload an RGB image and train a NN autoencoder to denoise it.
        The model learns to compress and reconstruct the image, then removes Gaussian noise.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="pil", height=400)
                
                target_size = gr.Slider(
                    minimum=32, 
                    maximum=256, 
                    value=64, 
                    step=16,
                    label="Processing Size (pixels)",
                    info="Smaller = faster, larger = more detail"
                )
                
                epochs = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=400,
                    step=50,
                    label="Training Epochs",
                    info="More epochs = better quality (but slower)"
                )
                
                noise_level = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.8,
                    step=0.05,
                    label="Noise Level",
                    info="Gaussian noise intensity"
                )
                
                hidden1_size = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=128,
                    step=1,
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
                gr.Markdown("### Test with Pure Noise")
                noise_test_btn = gr.Button("🎲 Generate Pure White Noise", variant="secondary", size="lg")
            
            with gr.Column(scale=2):
                output_image = gr.Image(label="Results: Original | Noisy | De-noised", height=400)
                loss_plot = gr.Image(label="Training Loss Curve", height=300)
                status_text = gr.Textbox(label="Status", lines=6, interactive=False)
        
        gr.Markdown("---")
        
        with gr.Row():
            noise_output = gr.Image(label="Pure Noise Test: Input | Reconstruction", height=400)
            noise_status = gr.Textbox(label="Noise Test Status", lines=3, interactive=False)
        
        # Wire up the interface
        process_btn.click(
            fn=process_image,
            inputs=[input_image, target_size, epochs, noise_level, hidden1_size, hidden2_size],
            outputs=[output_image, loss_plot, status_text]
        )
        
        # Wire up noise test button
        noise_test_btn.click(
            fn=generate_and_denoise_pure_noise,
            inputs=[],
            outputs=[noise_output, noise_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
