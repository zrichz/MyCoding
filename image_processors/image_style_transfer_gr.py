#!/home/rich/MyCoding/venvMyCoding/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import time

# --- 1. Image loading ---
def load_image(image, max_size=512):
    """Convert PIL Image or numpy array to tensor."""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    
    size = max(img.size) if max(img.size) < max_size else max_size
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- 2. Helper: Gram matrix ---
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

# --- 3. Extract features ---
def extract_features(x, vgg, content_layer, style_layers):
    features = {}
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in style_layers or name == content_layer:
            features[name] = x
    return features

# --- 4. Main style transfer function ---
def style_transfer(content_img, style_img, iterations=300, style_weight=1e6, content_weight=1.0, max_size=512, progress=gr.Progress()):
    """
    Apply neural style transfer.
    
    Args:
        content_img: PIL Image or numpy array of content image
        style_img: PIL Image or numpy array of style image
        iterations: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        max_size: Maximum size for images
        progress: gr.Progress object for updating progress
    
    Returns:
        PIL Image of stylized result
    """
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Update initial progress
    progress(0, desc="Initializing...")
    
    # Load and prepare images
    print("Loading images...")
    progress(0.05, desc="Loading images...")
    content = load_image(content_img, max_size).to(device)
    style = load_image(style_img, max_size).to(device)
    print(f"Content shape: {content.shape}, Style shape: {style.shape}")
    
    # Load VGG19
    print("Loading VGG19 model...")
    progress(0.1, desc="Loading VGG19...")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    
    # Layers used for content and style
    content_layer = "21"   # conv4_2
    style_layers = ["0", "5", "10", "19", "28"]  # conv1_1 ... conv5_1
    
    # Extract features
    print("Extracting features...")
    progress(0.15, desc="Extracting features...")
    
    with torch.no_grad():
        content_feats = extract_features(content, vgg, content_layer, style_layers)
        style_feats = extract_features(style, vgg, content_layer, style_layers)
        # Detach features to prevent gradient issues
        content_target = content_feats[content_layer].detach()
        style_grams = {layer: gram_matrix(style_feats[layer]).detach() for layer in style_layers}
    
    # Initialize output image
    output = content.clone().requires_grad_(True)
    
    # Optimization - use Adam with higher learning rate for normalized space
    optimizer = optim.Adam([output], lr=0.01)
    
    # Run optimization
    print(f"Starting optimization for {iterations} iterations...")
    progress(0.2, desc="Starting optimization...")
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass
        feats = extract_features(output, vgg, content_layer, style_layers)
        
        # Content loss
        content_loss = torch.nn.functional.mse_loss(feats[content_layer], content_target)
        
        # Style loss
        style_loss = 0
        for layer in style_layers:
            gram_output = gram_matrix(feats[layer])
            style_loss += torch.nn.functional.mse_loss(gram_output, style_grams[layer])
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Clamp pixel values to valid range (normalized space)
        with torch.no_grad():
            output.clamp_(-2.5, 2.5)  # Reasonable bounds for normalized images
        
        # Update progress bar EVERY iteration
        progress_value = 0.2 + ((i + 1) / iterations) * 0.75
        desc = f"Step {i+1}/{iterations} - Loss: {total_loss.item():.0f}"
        progress(progress_value, desc=desc)
        
        # Print to console periodically
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Step {i+1}/{iterations} - Loss: {total_loss.item():.0f} "
                  f"(Content: {content_loss.item():.2f}, Style: {style_loss.item():.2f})")
    
    print("Optimization complete. Converting to PIL Image...")
    progress(0.95, desc="Converting to image...")
    
    # Convert output to PIL Image - denormalize first
    out_tensor = output.detach().cpu().squeeze()
    
    # Denormalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out_tensor = out_tensor * std + mean
    out_tensor = out_tensor.clamp(0, 1)
    
    result = transforms.ToPILImage()(out_tensor)
    print(f"Generated image size: {result.size}, mode: {result.mode}")
    
    progress(1.0, desc="Complete!")
    
    return result

# --- 5. Gradio Interface ---
def process_images(content_img, style_img, iterations, style_weight, content_weight, size_preset, progress=gr.Progress()):
    """Wrapper function for Gradio interface."""
    if content_img is None or style_img is None:
        print("Error: Missing content or style image")
        return None
    
    # Parse size preset
    size_map = {
        "Fast (256px)": 256,
        "Normal (512px)": 512,
        "High Quality (768px)": 768
    }
    max_size = size_map.get(size_preset, 256)
    
    print("\n" + "="*50)
    print("Starting Neural Style Transfer")
    print(f"Mode: {size_preset} (max size: {max_size}px)")
    print("="*50)
    
    try:
        result = style_transfer(
            content_img, 
            style_img, 
            iterations=int(iterations),
            style_weight=float(style_weight),
            content_weight=float(content_weight),
            max_size=int(max_size),
            progress=progress
        )
        print("="*50)
        print("Style transfer completed successfully")
        print(f"Returning image: {result.size if result else 'None'}, type: {type(result)}")
        print("="*50 + "\n")
        
        # Ensure we're returning a PIL Image
        if result and isinstance(result, Image.Image):
            return result
        else:
            print("Warning: Result is not a valid PIL Image")
            return None
            
    except Exception as e:
        print(f"Error during style transfer: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Create Gradio interface
with gr.Blocks(title="Neural Style Transfer") as demo:
    gr.Markdown("# Neural Style Transfer")
    gr.Markdown("Upload a content image and a style image to transfer the artistic style.")
    
    with gr.Row():
        with gr.Column():
            content_input = gr.Image(label="Content Image", type="pil")
            style_input = gr.Image(label="Style Image", type="pil")
        
        with gr.Column():
            output_image = gr.Image(label="Stylized Result", type="pil")
    
    with gr.Row():
        iterations_slider = gr.Slider(
            minimum=50, maximum=500, value=200, step=10,
            label="Iterations (lower is faster)"
        )
        style_weight_slider = gr.Slider(
            minimum=1e4, maximum=1e7, value=1e6, step=1e4,
            label="Style Weight"
        )
    
    with gr.Row():
        content_weight_slider = gr.Slider(
            minimum=0.1, maximum=10.0, value=1.0, step=0.1,
            label="Content Weight"
        )
        size_preset = gr.Radio(
            choices=["Fast (256px)", "Normal (512px)", "High Quality (768px)"],
            value="Fast (256px)",
            label="Processing Size (smaller is faster)"
        )
    
    process_btn = gr.Button("Apply Style Transfer", variant="primary")
    
    gr.Markdown("### Notes")
    gr.Markdown("- Fast mode recommended for CPU processing")
    gr.Markdown("- Higher iterations produce better quality but take longer")
    gr.Markdown("- Higher style weight emphasizes the artistic style more")
    gr.Markdown("- Higher content weight preserves the original content better")
    gr.Markdown("- Progress bar shows real-time updates during processing")
    
    # Connect button to processing function
    process_btn.click(
        fn=process_images,
        inputs=[
            content_input, 
            style_input, 
            iterations_slider, 
            style_weight_slider, 
            content_weight_slider,
            size_preset
        ],
        outputs=output_image,
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
