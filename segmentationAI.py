import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
model.eval()

# Quantize the model
model_half = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).to(device).half()
model_half.eval()

# Load the LRASPP models
model_lraspp_mobilenet_v3_large = models.segmentation.lraspp_mobilenet_v3_large(weights=models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT).to(device)
model_lraspp_mobilenet_v3_large.eval()

model_lraspp_mobilenet_v3_large_half = models.segmentation.lraspp_mobilenet_v3_large(weights=models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT).to(device).half()
model_lraspp_mobilenet_v3_large_half.eval()

# Load the FCN models
model_fcn_resnet50 = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT).to(device)
model_fcn_resnet50.eval()

model_fcn_resnet101 = models.segmentation.fcn_resnet101(weights=models.segmentation.FCN_ResNet101_Weights.DEFAULT).to(device)
model_fcn_resnet101.eval()

# Function to get model size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# Preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(1024),  # Use a higher resolution for more detail
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch

# Perform segmentation
def segment_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_batch = preprocess_image(input_image)
    with torch.no_grad():
        start_time = time.time()
        output = model(input_batch)['out'][0]
        end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time: {inference_time:.2f} ms")
    output_predictions = output.argmax(0)
    return output_predictions.cpu().numpy()

# Perform segmentation with quantized model
def segment_image_half(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_batch = preprocess_image(input_image).half()  # Convert input to half precision
    with torch.no_grad():
        start_time = time.time()
        output = model_half(input_batch)['out'][0]
        end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time (half precision): {inference_time:.2f} ms")
    output_predictions = output.argmax(0)
    return output_predictions.cpu().numpy()

# Perform segmentation with LRASPP model
def segment_image_lraspp(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_batch = preprocess_image(input_image)
    with torch.no_grad():
        start_time = time.time()
        output = model_lraspp_mobilenet_v3_large(input_batch)['out'][0]
        end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time (LRASPP): {inference_time:.2f} ms")
    output_predictions = output.argmax(0)
    return output_predictions.cpu().numpy()

# Perform segmentation with quantized LRASPP model
def segment_image_lraspp_half(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_batch = preprocess_image(input_image).half()  # Convert input to half precision
    with torch.no_grad():
        start_time = time.time()
        output = model_lraspp_mobilenet_v3_large_half(input_batch)['out'][0]
        end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time (LRASPP half precision): {inference_time:.2f} ms")
    output_predictions = output.argmax(0)
    return output_predictions.cpu().numpy()

# Perform segmentation with FCN ResNet50 model
def segment_image_fcn_resnet50(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_batch = preprocess_image(input_image)
    with torch.no_grad():
        start_time = time.time()
        output = model_fcn_resnet50(input_batch)['out'][0]
        end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time (FCN ResNet50): {inference_time:.2f} ms")
    output_predictions = output.argmax(0)
    return output_predictions.cpu().numpy()

# Perform segmentation with FCN ResNet101 model
def segment_image_fcn_resnet101(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_batch = preprocess_image(input_image)
    with torch.no_grad():
        start_time = time.time()
        output = model_fcn_resnet101(input_batch)['out'][0]
        end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time (FCN ResNet101): {inference_time:.2f} ms")
    output_predictions = output.argmax(0)
    return output_predictions.cpu().numpy()

# Visualize the segmentation
def visualize_segmentation(original_image, segmentation, segmentation_half, segmentation_lraspp, segmentation_lraspp_half, segmentation_fcn_resnet50, segmentation_fcn_resnet101):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 7, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.subplot(1, 7, 2)
    plt.imshow(segmentation, cmap='tab20')  # Use a high contrast colormap
    plt.axis('off')
    plt.subplot(1, 7, 3)
    plt.imshow(segmentation_half, cmap='tab20')  # Use a high contrast colormap
    plt.axis('off')
    plt.subplot(1, 7, 4)
    plt.imshow(segmentation_lraspp, cmap='tab20')  # Use a high contrast colormap
    plt.axis('off')
    plt.subplot(1, 7, 5)
    plt.imshow(segmentation_lraspp_half, cmap='tab20')  # Use a high contrast colormap
    plt.axis('off')
    plt.subplot(1, 7, 6)
    plt.imshow(segmentation_fcn_resnet50, cmap='tab20')  # Use a high contrast colormap
    plt.axis('off')
    plt.subplot(1, 7, 7)
    plt.imshow(segmentation_fcn_resnet101, cmap='tab20')  # Use a high contrast colormap
    plt.axis('off')
    plt.show()

# Function to select an image file
def select_image_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpeg *.jpg *.png *.bmp *.tiff")]
    )
    return file_path

# Print model sizes
print(f"Original model size: {get_model_size(model):.2f} MB")
print(f"Quantized model size: {get_model_size(model_half):.2f} MB")
print(f"LRASPP model size: {get_model_size(model_lraspp_mobilenet_v3_large):.2f} MB")
print(f"LRASPP quantized model size: {get_model_size(model_lraspp_mobilenet_v3_large_half):.2f} MB")
print(f"FCN ResNet50 model size: {get_model_size(model_fcn_resnet50):.2f} MB")
print(f"FCN ResNet101 model size: {get_model_size(model_fcn_resnet101):.2f} MB")

# Select image file
image_path = select_image_file()
if not image_path:
    print("No image file selected.")
else:
    # Segment the image "as is"
    original_image = Image.open(image_path).convert("RGB")
    segmentation = segment_image(image_path)
    
    # Segment the image with quantized model
    segmentation_half = segment_image_half(image_path)
    
    # Segment the image with LRASPP model
    segmentation_lraspp = segment_image_lraspp(image_path)
    
    # Segment the image with quantized LRASPP model
    segmentation_lraspp_half = segment_image_lraspp_half(image_path)
    
    # Segment the image with FCN ResNet50 model
    segmentation_fcn_resnet50 = segment_image_fcn_resnet50(image_path)
    
    # Segment the image with FCN ResNet101 model
    segmentation_fcn_resnet101 = segment_image_fcn_resnet101(image_path)
    
    # Visualize the results
    visualize_segmentation(original_image, segmentation, segmentation_half, segmentation_lraspp, segmentation_lraspp_half, segmentation_fcn_resnet50, segmentation_fcn_resnet101)