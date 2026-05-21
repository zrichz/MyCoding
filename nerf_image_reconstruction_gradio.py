#!/usr/bin/env python3
"""
NeRF-style Image Reconstruction from Sparse Pixels
Compare 4 different model sizes: Small, Medium, Large, XLarge
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time


class PositionalEncoder(nn.Module):
    """Positional encoding for coordinate inputs (like NeRF)"""
    
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        
    def forward(self, x):
        """
        x: [batch, 2] normalized coordinates in [0, 1]
        returns: [batch, 2 + 2*num_frequencies*2] encoded coordinates
        """
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(2.0 * np.pi * freq * x))
            encoded.append(torch.cos(2.0 * np.pi * freq * x))
        return torch.cat(encoded, dim=-1)


class TinyMLP(nn.Module):
    """Small MLP for image reconstruction"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # Output RGB in [0, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ImageReconstructionModel:
    """Manages the training and inference of the MLP model"""
    
    def __init__(self, num_frequencies=10, hidden_dim=256, num_layers=4, lr=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = PositionalEncoder(num_frequencies)
        # Calculate input dimension after positional encoding
        input_dim = 2 + 2 * num_frequencies * 2
        self.mlp = TinyMLP(input_dim, hidden_dim, num_layers).to(self.device)
        
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.image = None
        self.sparse_mask = None
        self.sparse_coords = None
        self.sparse_colors = None
        self.total_steps = 0
        
    def load_image_and_create_sparse(self, image_array, sparsity_percent):
        """
        Load image and create sparse pixel sampling
        sparsity_percent: percentage of pixels to KEEP (0-100)
        """
        self.image = image_array.astype(np.float32) / 255.0
        h, w = self.image.shape[:2]
        
        # Create mask of which pixels to keep
        num_pixels = h * w
        num_keep = int(num_pixels * sparsity_percent / 100.0)
        
        # Random sampling
        all_indices = np.arange(num_pixels)
        keep_indices = np.random.choice(all_indices, num_keep, replace=False)
        
        self.sparse_mask = np.zeros((h, w), dtype=bool)
        row_indices = keep_indices // w
        col_indices = keep_indices % w
        self.sparse_mask[row_indices, col_indices] = True
        
        # Extract sparse coordinates and colors
        coords_y, coords_x = np.where(self.sparse_mask)
        
        # Normalize coordinates to [0, 1]
        self.sparse_coords = np.stack([
            coords_x / (w - 1),
            coords_y / (h - 1)
        ], axis=1).astype(np.float32)
        
        self.sparse_colors = self.image[coords_y, coords_x]
        
        return num_keep, h, w
    
    def train_steps(self, num_steps, batch_size=8192):
        """Train for specified number of steps"""
        self.mlp.train()
        
        num_samples = len(self.sparse_coords)
        losses = []
        
        print(f"\nStarting NeRF training for {num_steps} steps...")
        
        for step in range(num_steps):
            # Random batch sampling
            indices = np.random.choice(num_samples, min(batch_size, num_samples), replace=False)
            
            batch_coords = torch.from_numpy(self.sparse_coords[indices]).to(self.device)
            batch_colors = torch.from_numpy(self.sparse_colors[indices]).to(self.device)
            
            # Forward pass
            encoded_coords = self.encoder(batch_coords)
            pred_colors = self.mlp(encoded_coords)
            
            # Compute loss
            loss = self.criterion(pred_colors, batch_colors)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            self.total_steps += 1
            
            # Print progress every 10 iterations
            if (step + 1) % 10 == 0:
                current_loss = loss.item()
                print(f"  Step {self.total_steps} (batch {step + 1}/{num_steps}): Loss = {current_loss:.6f}")
        
        avg_loss = np.mean(losses)
        print(f"Training complete. Average loss: {avg_loss:.6f}\n")
        
        return avg_loss
    
    def reconstruct_image(self):
        """Reconstruct full image"""
        self.mlp.eval()
        
        h, w = self.image.shape[:2]
        
        # Create all pixel coordinates
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        all_coords = np.stack([
            x_coords.flatten() / (w - 1),
            y_coords.flatten() / (h - 1)
        ], axis=1).astype(np.float32)
        
        # Inference in batches to avoid memory issues
        batch_size = 16384
        reconstructed_colors = []
        
        with torch.no_grad():
            for i in range(0, len(all_coords), batch_size):
                batch = torch.from_numpy(all_coords[i:i+batch_size]).to(self.device)
                encoded = self.encoder(batch)
                colors = self.mlp(encoded)
                reconstructed_colors.append(colors.cpu().numpy())
        
        reconstructed_colors = np.concatenate(reconstructed_colors, axis=0)
        reconstructed_image = reconstructed_colors.reshape(h, w, 3)
        
        # Convert back to uint8
        reconstructed_image = (reconstructed_image * 255).clip(0, 255).astype(np.uint8)
        
        return reconstructed_image
    
    def get_sparse_visualization(self):
        """Create visualization of sparse pixels"""
        h, w = self.image.shape[:2]
        sparse_vis = np.zeros((h, w, 3), dtype=np.uint8)
        sparse_vis[self.sparse_mask] = (self.image[self.sparse_mask] * 255).astype(np.uint8)
        return sparse_vis


# Model configuration presets
MODEL_CONFIGS = {
    'Small': {'hidden_dim': 128, 'num_layers': 3, 'params': '~50k'},
    'Medium': {'hidden_dim': 256, 'num_layers': 4, 'params': '~200k'},
    'Large': {'hidden_dim': 512, 'num_layers': 6, 'params': '~1.5M'},
    'XLarge': {'hidden_dim': 1024, 'num_layers': 8, 'params': '~8M'}
}

# Global model instances
models = {
    'Small': None,
    'Medium': None,
    'Large': None,
    'XLarge': None
}

# Shared sparse data
sparse_data = {
    'image': None,
    'sparse_coords': None,
    'sparse_colors': None,
    'sparse_mask': None,
    'sparse_vis': None
}


def load_and_create_sparse(image, sparsity_percent):
    """Load image and create sparse sampling for all models"""
    global models, sparse_data
    
    if image is None:
        return None, None, None, None, None, "Please load an image first"
    
    # Create sparse sampling (shared across all models)
    image_data = image.astype(np.float32) / 255.0
    h, w = image_data.shape[:2]
    
    num_pixels = h * w
    num_keep = int(num_pixels * sparsity_percent / 100.0)
    
    all_indices = np.arange(num_pixels)
    keep_indices = np.random.choice(all_indices, num_keep, replace=False)
    
    sparse_mask = np.zeros((h, w), dtype=bool)
    row_indices = keep_indices // w
    col_indices = keep_indices % w
    sparse_mask[row_indices, col_indices] = True
    
    coords_y, coords_x = np.where(sparse_mask)
    sparse_coords = np.stack([
        coords_x / (w - 1),
        coords_y / (h - 1)
    ], axis=1).astype(np.float32)
    
    sparse_colors = image_data[coords_y, coords_x]
    
    # Create visualization
    sparse_vis = np.zeros((h, w, 3), dtype=np.uint8)
    sparse_vis[sparse_mask] = (image_data[sparse_mask] * 255).astype(np.uint8)
    
    # Store shared data
    sparse_data['image'] = image_data
    sparse_data['sparse_coords'] = sparse_coords
    sparse_data['sparse_colors'] = sparse_colors
    sparse_data['sparse_mask'] = sparse_mask
    sparse_data['sparse_vis'] = sparse_vis
    
    # Initialize all 4 models with shared sparse data
    for size_name, config in MODEL_CONFIGS.items():
        model = ImageReconstructionModel(
            num_frequencies=10,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            lr=1e-3
        )
        model.image = image_data
        model.sparse_coords = sparse_coords
        model.sparse_colors = sparse_colors
        model.sparse_mask = sparse_mask
        models[size_name] = model
    
    info = (f"Image loaded: {w} x {h} pixels\n"
            f"Kept {num_keep:,} pixels ({sparsity_percent:.1f}%)\n"
            f"Removed {h*w - num_keep:,} pixels\n\n"
            f"Initialized 4 NeRF models:\n"
            f"- Small: {MODEL_CONFIGS['Small']['params']} parameters\n"
            f"- Medium: {MODEL_CONFIGS['Medium']['params']} parameters\n"
            f"- Large: {MODEL_CONFIGS['Large']['params']} parameters\n"
            f"- XLarge: {MODEL_CONFIGS['XLarge']['params']} parameters\n\n"
            f"Ready to train all models")
    
    return sparse_vis, None, None, None, None, info


def train_all_models(num_steps):
    """Train all 4 models sequentially"""
    global models
    
    if models['Small'] is None:
        return None, None, None, None, "Please load and create sparse sample first"
    
    if num_steps <= 0:
        return None, None, None, None, "Number of steps must be positive"
    
    results = {}
    info_lines = []
    
    print("\n" + "="*70)
    print("Training all 4 NeRF models sequentially")
    print("="*70)
    
    for size_name in ['Small', 'Medium', 'Large', 'XLarge']:
        model = models[size_name]
        config = MODEL_CONFIGS[size_name]
        
        print(f"\n{'='*70}")
        print(f"Training {size_name} model ({config['params']} params)")
        print(f"  Architecture: {config['num_layers']} layers, {config['hidden_dim']} hidden units")
        print(f"{'='*70}")
        
        start_time = time.time()
        avg_loss = model.train_steps(num_steps)
        train_time = time.time() - start_time
        
        reconstruction = model.reconstruct_image()
        results[size_name] = reconstruction
        
        info_lines.append(
            f"{size_name}: {model.total_steps} total steps, "
            f"Loss={avg_loss:.6f}, Time={train_time:.1f}s"
        )
        
        print(f"  Completed in {train_time:.1f}s")
    
    print(f"\n{'='*70}")
    print("All models trained")
    print(f"{'='*70}\n")
    
    info = "Training Complete\n\n" + "\n".join(info_lines)
    
    return (results['Small'], results['Medium'], 
            results['Large'], results['XLarge'], info)


def save_reconstruction(img, size_name):
    """Save a reconstructed image"""
    if img is None:
        return f"No {size_name} image to save"
    
    model = models[size_name]
    if model is None:
        return f"No {size_name} model available"
    
    pil_img = Image.fromarray(img)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"nerf_{size_name.lower()}_{timestamp}_steps{model.total_steps}.png"
    filepath = f"/home/rich/MyCoding/fractal_outputs/{filename}"
    
    pil_img.save(filepath)
    
    return f"Saved {size_name} to {filename}"


def reset_all():
    """Reset all models"""
    global models, sparse_data
    
    for key in models:
        models[key] = None
    
    for key in sparse_data:
        sparse_data[key] = None
    
    return None, None, None, None, None, "All models reset. Load a new image to start."


# Create Gradio interface
with gr.Blocks(title="NeRF Multi-Model Comparison") as demo:
    gr.Markdown("# NeRF Image Reconstruction: Multi-Model Size Comparison")
    gr.Markdown("""
    Train 4 different NeRF model sizes **sequentially** and compare results side-by-side:
    - **Small**: 3 layers, 128 units (~50k params) - Fastest
    - **Medium**: 4 layers, 256 units (~200k params) - Balanced
    - **Large**: 6 layers, 512 units (~1.5M params) - High quality
    - **XLarge**: 8 layers, 1024 units (~8M params) - Maximum quality
    
    All models train on the same sparse data. Compare quality vs. training time tradeoffs.
    """)
    
    with gr.Row():
        # Left column - controls
        with gr.Column(scale=1):
            gr.Markdown("### Input & Controls")
            input_image = gr.Image(label="Input Image", type="numpy")
            
            sparsity_slider = gr.Slider(
                minimum=0.5, maximum=20, value=1, step=0.5,
                label="Sparsity % (pixels to keep)"
            )
            create_btn = gr.Button("Create Sparse Sample", variant="primary", size="lg")
            
            gr.Markdown("### Training")
            num_steps = gr.Slider(
                minimum=10, maximum=2000, value=100, step=10,
                label="Training Steps (per model)"
            )
            train_btn = gr.Button("Train All 4 Models", variant="primary", size="lg")
            reset_btn = gr.Button("Reset All", variant="secondary")
            
            gr.Markdown("### Sparse Visualization")
            sparse_output = gr.Image(label="Sparse Pixels (input)", height=300)
        
        # Right column - 4 model outputs
        with gr.Column(scale=3):
            gr.Markdown("### Reconstructions (Side-by-Side)")
            
            with gr.Row():
                small_output = gr.Image(label="Small (~50k params)", height=300)
                medium_output = gr.Image(label="Medium (~200k params)", height=300)
            
            with gr.Row():
                large_output = gr.Image(label="Large (~1.5M params)", height=300)
                xlarge_output = gr.Image(label="XLarge (~8M params)", height=300)
            
            gr.Markdown("### Save Options")
            with gr.Row():
                save_small_btn = gr.Button("Save Small")
                save_medium_btn = gr.Button("Save Medium")
                save_large_btn = gr.Button("Save Large")
                save_xlarge_btn = gr.Button("Save XLarge")
    
    info_text = gr.Textbox(label="Status & Training Info", lines=10, max_lines=15)
    save_status = gr.Textbox(label="Save Status", lines=2)
    
    # Wire up the interface
    create_btn.click(
        fn=load_and_create_sparse,
        inputs=[input_image, sparsity_slider],
        outputs=[sparse_output, small_output, medium_output, large_output, xlarge_output, info_text]
    )
    
    train_btn.click(
        fn=train_all_models,
        inputs=[num_steps],
        outputs=[small_output, medium_output, large_output, xlarge_output, info_text]
    )
    
    save_small_btn.click(
        fn=lambda img: save_reconstruction(img, 'Small'),
        inputs=[small_output],
        outputs=[save_status]
    )
    
    save_medium_btn.click(
        fn=lambda img: save_reconstruction(img, 'Medium'),
        inputs=[medium_output],
        outputs=[save_status]
    )
    
    save_large_btn.click(
        fn=lambda img: save_reconstruction(img, 'Large'),
        inputs=[large_output],
        outputs=[save_status]
    )
    
    save_xlarge_btn.click(
        fn=lambda img: save_reconstruction(img, 'XLarge'),
        inputs=[xlarge_output],
        outputs=[save_status]
    )
    
    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[sparse_output, small_output, medium_output, large_output, xlarge_output, info_text]
    )


if __name__ == "__main__":
    print("Starting NeRF Multi-Model Comparison app...")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("\nModel Sizes:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name}: {config['num_layers']} layers, {config['hidden_dim']} units ({config['params']} params)")
    print("\nAll 4 models train sequentially on the same sparse data.")
    demo.launch(inbrowser=True)
