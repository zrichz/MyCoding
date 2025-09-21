import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import os

def process_image():
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename(
		title="Select an image",
		filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
	)
	if not file_path:
		print("No file selected.")
		return

	img = Image.open(file_path).convert("RGB")
	arr = np.array(img)
	# Calculate average of red and blue channels
	avg_rb = ((arr[..., 0].astype(np.uint16) + arr[..., 2].astype(np.uint16)) // 2).astype(np.uint8)
	arr[..., 1] = avg_rb  # Replace green channel

	new_img = Image.fromarray(arr)
	base, ext = os.path.splitext(file_path)
	new_path = f"{base}_green_average{ext}"
	new_img.save(new_path)
	print(f"Saved: {new_path}")

if __name__ == "__main__":
	process_image()
