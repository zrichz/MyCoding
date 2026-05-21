import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# --- 1. Image loading ---
def load_image(path, max_size=512):
    img = Image.open(path).convert("RGB")
    size = max(img.size) if max(img.size) < max_size else max_size
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return transform(img).unsqueeze(0)

content = load_image("content.jpg").cuda()
style = load_image("style.jpg").cuda()

# --- 2. VGG19 feature extractor ---
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.cuda().eval()

# Layers used for content and style
content_layer = "21"   # conv4_2
style_layers = ["0", "5", "10", "19", "28"]  # conv1_1 ... conv5_1

# --- 3. Helper: Gram matrix ---
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

# --- 4. Extract features ---
def extract_features(x):
    features = {}
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in style_layers or name == content_layer:
            features[name] = x
    return features

content_feats = extract_features(content)
style_feats = extract_features(style)
style_grams = {layer: gram_matrix(style_feats[layer]) for layer in style_layers}

# --- 5. Initialize output image ---
output = content.clone().requires_grad_(True)

# --- 6. Optimization ---
optimizer = optim.LBFGS([output])
style_weight = 1e6
content_weight = 1e0

def closure():
    optimizer.zero_grad()
    feats = extract_features(output)

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
for i in range(300):
    optimizer.step(closure)

# --- 7. Save result ---
out_img = output.detach().cpu().squeeze().clamp(0, 255) / 255
transforms.ToPILImage()(out_img).save("output.jpg")
