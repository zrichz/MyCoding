import os
from PIL import Image, ImageDraw, ImageFont
import emoji

# Get the list of all emojis
all_emojis = emoji.EMOJI_DATA.keys()

# Directory to save emoji images
emoji_dir = "emojis"
os.makedirs(emoji_dir, exist_ok=True)

# Path to the font file
font_path = "/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf"  # Update this path to a valid TTF font file on your system

# Create and save emoji images
for i, emoji_char in enumerate(all_emojis):
    # Create a blank image with white background
    img = Image.new("RGB", (64, 64), color="white")
    d = ImageDraw.Draw(img)
    
    try:
        # Load a font
        font = ImageFont.truetype(font_path, 48)
    except IOError:
        print(f"Font file not found: {font_path}")
        continue
    
    # Draw the emoji on the image
    d.text((10, 5), emoji_char, font=font, fill="black")
    
    # Save the image
    img.save(os.path.join(emoji_dir, f"emoji_{i}.png"))

print("Emoji images created and saved successfully!")