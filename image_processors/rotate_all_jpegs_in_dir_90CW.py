from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def rotate_jpegs():
    # Ask user to select input directory
    input_dir = filedialog.askdirectory(title="Select directory containing JPEG files")
    
    if not input_dir:
        return  # User cancelled
    
    # Create output directory as subdirectory of input
    output_dir = os.path.join(input_dir, 'rotated_90CW')
    os.makedirs(output_dir, exist_ok=True)
    
    # Count files to process
    jpeg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not jpeg_files:
        messagebox.showinfo("No Files", "No JPEG files found in the selected directory.")
        return
    
    # Process each JPEG file
    processed = 0
    for filename in jpeg_files:
        try:
            filepath = os.path.join(input_dir, filename)
            img = Image.open(filepath)
            rotated = img.transpose(Image.Transpose.ROTATE_270)  # 90° CW
            rotated.save(os.path.join(output_dir, filename))
            processed += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    messagebox.showinfo("Complete", f"Successfully rotated {processed} JPEG files.\nSaved to: {output_dir}")

# Create GUI
root = tk.Tk()
root.title("JPEG Rotator - 90° Clockwise")
root.geometry("300x150")

# Main frame
frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True, fill='both')

# Title label
title_label = tk.Label(frame, text="Rotate JPEG Files 90° Clockwise", font=("Arial", 12, "bold"))
title_label.pack(pady=(0, 20))

# Button to select directory and rotate
rotate_button = tk.Button(frame, text="Select Directory & Rotate JPEGs", 
                         command=rotate_jpegs, bg="#4CAF50", fg="white", 
                         font=("Arial", 10), padx=20, pady=10)
rotate_button.pack()

# Instructions
instructions = tk.Label(frame, text="Click to choose a folder with JPEG files.\nRotated images will be saved in 'rotated_90CW' subfolder.", 
                       font=("Arial", 9), fg="gray")
instructions.pack(pady=(20, 0))

root.mainloop()
