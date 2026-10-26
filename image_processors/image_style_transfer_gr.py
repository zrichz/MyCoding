#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Neural style transfer between two images.
Each image is centre-cropped to a 512x512 square, then the style (colours, textures, brushwork) of one is painted onto the content of the other, and vice versa.
Results are shown in a 1024x1024 montage.
Built with PyTorch and Gradio.

How it works:
The script optimises the pixels of a copy of the content image (500 L-BFGS steps) so that its VGG19 activations match two targets at once.
VGG19 is a 19-layer convolutional neural network from Oxford's Visual Geometry Group, and only its convolutional "features" part is used here, with frozen pretrained weights.
Early layers respond to colours and edges, middle layers to textures and patterns, and deeper layers to object parts and overall layout.
  - Content: layer 21 (conv4_2), a deep layer. The output is penalised for differing from the content image's activations there, which preserves its layout and objects.
  - Style: layers 0, 5, 10, 19 and 28 (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1), one from each block. At each, the Gram matrix (how strongly feature channels fire together, ignoring where) is compared with that of the style image. Fine layers capture colour and brushstrokes, coarse layers capture larger motifs.
  - Weights: style 100000, content 0.1, so style dominates.

About ImageNet:
VGG19's weights come from training on ImageNet, a dataset of over 14 million web-scraped photos labelled with about 20,000 categories.
Models like this one are normally trained on the 1.28 million image, 1000 class subset used in the ILSVRC 2012 challenge.
The images were collected from internet search engines and photo sites, and ImageNet does not own them: the copyright stays with the original photographers and publishers, and consent was not sought from them or from the people pictured.
ImageNet is distributed for non-commercial research and educational use under its terms of access,
and the dataset's own people categories have been criticised for biased and offensive labels, with some removed in 2019 to 2021.
Whether weights trained on such data may be used commercially,
and whether training counts as fair use, is legally unsettled and varies by jurisdiction.
"""
import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np

# Fixed settings
ITERATIONS = 500
STYLE_WEIGHT = 100000.0
CONTENT_WEIGHT = 0.1
IMAGE_SIZE = 512

_vgg_cache = {}


def get_vgg(device):
    """Load VGG19 once per device and reuse it."""
    if device not in _vgg_cache:
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        vgg.requires_grad_(False)
        _vgg_cache[device] = vgg
    return _vgg_cache[device]


def to_square(image, size=IMAGE_SIZE):
    """Center crop to 1:1 and resize to size x size RGB."""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    img = img.convert("RGB")
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


# --- 1. Image loading ---
def load_image(img):
    """Convert a PIL image to a normalized tensor."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- 2. Helper: Gram matrix ---
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
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
def style_transfer(content_img, style_img, iterations=ITERATIONS, style_weight=STYLE_WEIGHT,
                   content_weight=CONTENT_WEIGHT, progress=None, progress_range=(0.0, 1.0),
                   label=""):
    """Apply neural style transfer to two square PIL images; returns a PIL image."""
    p0, p1 = progress_range

    def report(fraction, desc):
        if progress is not None:
            progress(p0 + (p1 - p0) * fraction, desc=desc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    report(0.0, f"{label}Preparing")

    content = load_image(content_img).to(device)
    style = load_image(style_img).to(device)
    vgg = get_vgg(device)
    
    # Layers used for content and style
    content_layer = "21"   # conv4_2
    style_layers = ["0", "5", "10", "19", "28"]  # conv1_1 ... conv5_1
    
    # Extract features
    with torch.no_grad():
        content_feats = extract_features(content, vgg, content_layer, style_layers)
        style_feats = extract_features(style, vgg, content_layer, style_layers)
        # Detach features to prevent gradient issues
        content_target = content_feats[content_layer].detach()
        style_grams = {layer: gram_matrix(style_feats[layer]).detach() for layer in style_layers}
    
    # Initialize output image
    output = content.clone().requires_grad_(True)
    
    optimizer = optim.LBFGS(
        [output],
        lr=1.0,
        max_iter=1,
        max_eval=20,
        history_size=10,
        line_search_fn="strong_wolfe"
    )

    # Run quasi-Newton optimization. Each step may evaluate the closure several times
    # while the line search finds a suitable update.
    print(f"Starting optimization for {iterations} iterations...")
    
    for i in range(iterations):
        latest_loss = None

        def closure():
            nonlocal latest_loss
            optimizer.zero_grad()
            feats = extract_features(output, vgg, content_layer, style_layers)
            content_loss = torch.nn.functional.mse_loss(
                feats[content_layer], content_target
            )
            style_loss = sum(
                torch.nn.functional.mse_loss(
                    gram_matrix(feats[layer]), style_grams[layer]
                )
                for layer in style_layers
            )
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            latest_loss = total_loss.detach()
            return total_loss

        optimizer.step(closure)

        with torch.no_grad():
            output.clamp_(-2.5, 2.5)

        report((i + 1) / iterations,
               f"{label}Step {i+1}/{iterations} - Loss: {latest_loss.item():.0f}")
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Step {i+1}/{iterations} - Loss: {latest_loss.item():.0f}")
    
    # Convert output to PIL Image - denormalize first
    out_tensor = output.detach().cpu().squeeze()
    
    # Denormalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out_tensor = out_tensor * std + mean
    out_tensor = out_tensor.clamp(0, 1)
    
    return transforms.ToPILImage()(out_tensor)

# --- 5. Gradio Interface ---
def process_images(image_1, image_2, progress=gr.Progress()):
    """Style each image with the other and return a 2x2 montage."""
    if image_1 is None or image_2 is None:
        return None

    try:
        img1 = to_square(image_1)
        img2 = to_square(image_2)

        styled_1 = style_transfer(img1, img2, progress=progress,
                                  progress_range=(0.0, 0.5), label="Image 1: ")
        styled_2 = style_transfer(img2, img1, progress=progress,
                                  progress_range=(0.5, 1.0), label="Image 2: ")

        # Left column: image 1 over its restyled version.
        # Right column: restyled image 2 over the original image 2.
        montage = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE * 2))
        montage.paste(img1, (0, 0))
        montage.paste(styled_1, (0, IMAGE_SIZE))
        montage.paste(styled_2, (IMAGE_SIZE, 0))
        montage.paste(img2, (IMAGE_SIZE, IMAGE_SIZE))
        return montage

    except Exception as e:
        print(f"Error during style transfer: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Create Gradio interface
with gr.Blocks(title="Neural Style Transfer") as demo:
    gr.Markdown("# Neural Style Transfer")
    gr.Markdown("Upload two images. Each is cropped to a square, then the style of each is applied to the other.")

    with gr.Row():
        with gr.Column():
            content_input = gr.Image(label="Image 1", type="pil", height=300)
            style_input = gr.Image(label="Image 2", type="pil", height=300)

        with gr.Column():
            output_image = gr.Image(label="Montage (1024x1024)", type="pil")

    process_btn = gr.Button("Apply Style Transfer", variant="primary")

    process_btn.click(
        fn=process_images,
        inputs=[content_input, style_input],
        outputs=output_image,
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
