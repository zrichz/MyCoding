import torch
from pathlib import Path
from sys import argv
import open_clip
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the CLIP model
print("Loading CLIP model...")
model_name = "ViT-B-32"
pretrained = "openai"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=pretrained,
    device=device,
)
print("CLIP model loaded successfully!")
print(f"Model: {model_name}")
print("model loaded onto device: ", device)

# Load the BLIP model and processor
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
print("BLIP model loaded successfully!")


# Function to generate captions using BLIP
def generate_caption_blip(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Hardcoded image path
image_path = "/home/rich/MyCoding/images_general/1024x1024RGB_image_sample.png"

# Load the image and generate a caption using CLIP
print(f"Loading image from {image_path}")
im = Image.open(image_path).convert("RGB")
print("Image loaded successfully!")

# Load the image and generate a caption using BLIP
print("Generating caption using BLIP...")
caption_blip = generate_caption_blip(im)
print(f"Generated Caption using BLIP: {caption_blip}")

# Load the image and generate a caption
print(f"Loading image from {image_path}")
im = Image.open(image_path).convert("RGB")
print("Image loaded successfully!")
print("Generating caption...")
caption = generate_caption_blip(im)
print(f"Generated Caption: {caption}")
