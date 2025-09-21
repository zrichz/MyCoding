import os
import platform
from PIL import Image, ImageDraw, ImageFont
import emoji

# Configuration
MAX_EMOJIS = 1000  # Limit number of emojis to prevent creating too many files
IMAGE_SIZE = 256   # Size of emoji images (64x64 pixels)
FONT_SIZE = 32    # Font size for emoji rendering

# Get the list of all emojis (limited to prevent too many files)
all_emojis = list(emoji.EMOJI_DATA.keys())[:MAX_EMOJIS]

# Directory to save emoji images
emoji_dir = "emojis"
os.makedirs(emoji_dir, exist_ok=True)

print(f"Will create up to {len(all_emojis)} emoji images ({IMAGE_SIZE}x{IMAGE_SIZE} pixels)")

# Function to find appropriate emoji font based on OS
def find_emoji_font():
    system = platform.system()
    
    if system == "Windows":
        # Windows emoji fonts
        possible_fonts = [
            "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
            "C:/Windows/Fonts/NotoColorEmoji.ttf",  # Noto Color Emoji
            "C:/Windows/Fonts/arial.ttf",  # Fallback
        ]
    elif system == "Darwin":  # macOS
        possible_fonts = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    else:  # Linux
        possible_fonts = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
        ]
    
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return font_path
    
    return None  # Use default font

# Find and load appropriate font
font_path = find_emoji_font()

# Create font object
try:
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, FONT_SIZE)
        print(f"Using font: {font_path}")
    else:
        print("No suitable emoji font found, using default font")
        font = ImageFont.load_default()
except Exception as e:
    print(f"Error loading font: {e}")
    font = ImageFont.load_default()

# Create and save emoji images
created_count = 0
for i, emoji_char in enumerate(all_emojis):
    # Skip if emoji is empty or None
    if not emoji_char or emoji_char.strip() == "":
        continue
        
    # Create a blank image with white background (grayscale to save memory)
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)  # "L" mode for grayscale, 255 = white
    d = ImageDraw.Draw(img)
    
    try:
        # Draw the emoji on the image with proper vertical centering
        bbox = d.textbbox((0, 0), emoji_char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate horizontal centering
        x = (IMAGE_SIZE - text_width) // 2
        
        # Calculate vertical centering - account for bbox offset from (0,0)
        # bbox gives us (left, top, right, bottom) relative to the anchor point
        # We need to center the actual visual content, not just the bounding box
        y = (IMAGE_SIZE - text_height) // 2 - bbox[1]
        
        d.text((x, y), emoji_char, font=font, fill=0)  # 0 = black in grayscale
        
        # Save the image as grayscale PNG
        img.save(os.path.join(emoji_dir, f"e_{created_count}.png"))
        created_count += 1
        
        # Print progress every 100 emojis
        if created_count % 100 == 0:
            print(f"Created {created_count} emoji images...")
            
    except Exception as e:
        print(f"Error creating emoji {emoji_char}: {e}")
        continue

print(f"Successfully created {created_count} emoji images in the '{emoji_dir}' directory!")