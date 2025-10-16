import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os

def process_image():
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename(
		title="Select an image",
		filetypes=[
			("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
			("PNG files", "*.png"),
			("JPEG files", "*.jpg *.jpeg"),
			("All files", "*.*")
		]
	)
	if not file_path:
		print("No file selected.")
		return

	img = Image.open(file_path).convert("RGB")
	width, height = img.size
	
	# Get pixel data and process each pixel
	pixels = list(img.getdata())
	new_pixels = []
	
	for r, g, b in pixels:
		# Calculate average of red and blue channels for new green value
		avg_rb = (r + b) // 2
		new_pixels.append((r, avg_rb, b))
	
	# Create new image with modified pixels
	new_img = Image.new("RGB", (width, height))
	new_img.putdata(new_pixels)
	base, ext = os.path.splitext(file_path)
	new_path = f"{base}_green_average{ext}"
	new_img.save(new_path)
	print(f"Saved: {new_path}")

if __name__ == "__main__":
	process_image()
