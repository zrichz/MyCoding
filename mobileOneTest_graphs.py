import os
import torch
from torchvision import transforms
from PIL import Image
from mobileone_pytorch import mobileone_s4
import json

# Function to load and preprocess the image
def load_image(image_path):
    # Define the transformations: resize, center crop, and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])
    # Load the image
    image = Image.open(image_path).convert('RGB')
    print(f"Loaded image size: {image.size}")
    image = transform(image)
    print(f"Transformed image shape: {image.shape}")
    image = image.unsqueeze(0)  # Add a batch dimension
    print(f"Image tensor shape after unsqueeze: {image.shape}")
    return image

# Function to predict the subject using MobileOne
def predict_subject(image, model):
    with torch.no_grad():
        model.eval()
        output = model(image)
        print(f"Model output shape: {output.shape}")
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        return top5_catid[0].tolist(), top5_prob[0].tolist()

# Function to display the selected image and prediction
def display_image_and_prediction(image_path, model, labels):
    # Load and preprocess the image
    image = load_image(image_path)

    # Predict the subject
    top5_catid, top5_prob = predict_subject(image, model)
    top5_labels = [labels[catid] for catid in top5_catid]

    # Print the predictions
    print('Predictions:')
    for catid, label, prob in zip(top5_catid, top5_labels, top5_prob):
        print(f'  Category {catid}: {label} ({prob*100:.2f}%)')

# Load the MobileOne model

model = mobileone_s4()
checkpoint = torch.load('/home/rich/myenv/mobileone_s4.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# Load ImageNet class labels from the JSON file
with open('imagenet_labels.json') as f:
    labels = json.load(f)

# Hardcoded image path
image_path = "/home/rich/Downloads/oranges.jpg"

# Display the image and prediction
display_image_and_prediction(image_path, model, labels)

