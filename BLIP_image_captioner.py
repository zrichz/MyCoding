import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP model and processor
print("Loading BLIP model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Device: {device}")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
print("BLIP model loaded successfully!")

# Function to generate captions using BLIP
def generate_caption_blip(image, max_length, min_length, num_beams, length_penalty):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Hardcoded image path
# image_path = "/home/rich/MyCoding/images_general/1024x1024RGB_image_sample.png"
# image_path = "/home/rich/MyCoding/images_general/64GANfaces.jpg"
image_path = "/home/rich/MyCoding/images_general/MonaLisa-662199825.jpg"

# Load the image
print(f"Loading image from {image_path}")
im = Image.open(image_path).convert("RGB")
print("Image loaded successfully!")

# Generate captions with increasing lengths
for i in range(6):
    max_length = (i + 1) * 8  # Increase max_length
    min_length = (i + 1) * 4  # Increase min_length
    num_beams = 4  # Use beam search with 5 beams
    length_penalty = 1.25  # increase length penalty to encourage longer captions
    #print(f"Generating caption with max_length={max_length}, min_length={min_length}, num_beams={num_beams}, length_penalty={length_penalty}...")
    caption = generate_caption_blip(im, max_length=max_length, min_length=min_length, num_beams=num_beams, length_penalty=length_penalty)
    print(f"Generated Caption {i + 1}: {caption}")