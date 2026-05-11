import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np

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
        transforms.Lambda(lambda x: x.mul(255))
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
def style_transfer(content_img, style_img, iterations=300, style_weight=1e6, content_weight=1.0, max_size=512):
    """
    Apply neural style transfer.
    
    Args:
        content_img: PIL Image or numpy array of content image
        style_img: PIL Image or numpy array of style image
        iterations: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        max_size: Maximum size for images
    
    Returns:
        PIL Image of stylized result
    """
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare images
    content = load_image(content_img, max_size).to(device)
    style = load_image(style_img, max_size).to(device)
    
    # Load VGG19
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    
    # Layers used for content and style
    content_layer = "21"   # conv4_2
    style_layers = ["0", "5", "10", "19", "28"]  # conv1_1 ... conv5_1
    
    # Extract features
    content_feats = extract_features(content, vgg, content_layer, style_layers)
    style_feats = extract_features(style, vgg, content_layer, style_layers)
    style_grams = {layer: gram_matrix(style_feats[layer]) for layer in style_layers}
    
    # Initialize output image
    output = content.clone().requires_grad_(True)
    
    # Optimization
    optimizer = optim.LBFGS([output])
    
    def closure():
        optimizer.zero_grad()
        feats = extract_features(output, vgg, content_layer, style_layers)
        
        # Content loss
        content_loss = torch.nn.functional.mse_loss(
            feats[content_layer], content_feats[content_layer]
        )
        
        # Style loss
        style_loss = 0
        for layer in style_layers:
            gram = gram_matrix(feats[layer])
            style_loss += torch.nn.functional.mse_loss(gram, style_grams[layer])
        
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()
        return loss
    
    # Run optimization
    for i in range(iterations):
        optimizer.step(closure)
    
    # Convert output to PIL Image
    out_img = output.detach().cpu().squeeze().clamp(0, 255) / 255
    result = transforms.ToPILImage()(out_img)
    
    return result

# --- 5. Gradio Interface ---
def process_images(content_img, style_img, iterations, style_weight, content_weight, max_size):
    """Wrapper function for Gradio interface."""
    if content_img is None or style_img is None:
        return None
    
    try:
        result = style_transfer(
            content_img, 
            style_img, 
            iterations=int(iterations),
            style_weight=float(style_weight),
            content_weight=float(content_weight),
            max_size=int(max_size)
        )
        return result
    except Exception as e:
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
            minimum=50, maximum=500, value=300, step=10,
            label="Iterations"
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
        max_size_slider = gr.Slider(
            minimum=256, maximum=1024, value=512, step=64,
            label="Max Image Size"
        )
    
    process_btn = gr.Button("Apply Style Transfer", variant="primary")
    
    gr.Markdown("### Notes")
    gr.Markdown("- Higher iterations produce better quality but take longer")
    gr.Markdown("- Higher style weight emphasizes the artistic style more")
    gr.Markdown("- Higher content weight preserves the original content better")
    gr.Markdown("- Smaller image sizes process faster")
    
    # Connect button to processing function
    process_btn.click(
        fn=process_images,
        inputs=[
            content_input, 
            style_input, 
            iterations_slider, 
            style_weight_slider, 
            content_weight_slider,
            max_size_slider
        ],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
