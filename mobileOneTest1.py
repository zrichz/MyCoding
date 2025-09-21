import os
import torch
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Toplevel, Listbox, Scrollbar
from tkinter import ttk
from mobileone_pytorch import mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4
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
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

# Function to predict the subject using MobileOne
def predict_subject(image, model):
    with torch.no_grad():
        model.eval()
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        return top3_catid[0].tolist(), top3_prob[0].tolist()

# Function to display the selected image and prediction
def display_image_and_prediction(image_path, models, labels):
    # Load and preprocess the image
    image = load_image(image_path)

    predictions = []
    for model_name, model in models.items():
        # Predict the subject
        top3_catid, top3_prob = predict_subject(image, model)
        top3_labels = [labels[catid] for catid in top3_catid]
        predictions.append((model_name, top3_labels, top3_prob))

    # Display the image thumbnail
    img = Image.open(image_path).resize((100, 100))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img

    # Clear the table
    for row in tree.get_children():
        tree.delete(row)

    # Insert the predictions into the table
    for model_name, top3_labels, top3_prob in predictions:
        for label, prob in zip(top3_labels, top3_prob):
            tree.insert("", "end", values=(model_name, label, f"{prob*100:.2f}%"))

# Create a Tkinter root window
root = tk.Tk()
root.geometry("800x600")  # Set the size of the root window

# Load the MobileOne models
models = {
    'MobileOne_s0': mobileone_s0(),
    'MobileOne_s1': mobileone_s1(),
    'MobileOne_s2': mobileone_s2(),
    'MobileOne_s3': mobileone_s3(),
    'MobileOne_s4': mobileone_s4()
}

# Load ImageNet class labels from the JSON file
with open('imagenet_labels.json') as f:
    labels = json.load(f)

# Create a label to display the image thumbnail
img_label = Label(root)
img_label.pack()

# Create a Treeview widget to display the predicted categories and confidence
tree = ttk.Treeview(root, columns=("Model", "Category", "Confidence"), show="headings")
tree.heading("Model", text="Model")
tree.heading("Category", text="Category")
tree.heading("Confidence", text="Confidence")
tree.pack(fill=tk.BOTH, expand=True)

# Hardcoded image path
image_path = "/home/rich/Downloads/oranges.jpg"

# Display the image and prediction
display_image_and_prediction(image_path, models, labels)

root.mainloop()
