import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tkinter as tk
from tkinter import filedialog, ttk
import threading

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

# GUI Application
class BLIPCaptionerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BLIP Image Captioner")
        self.root.geometry("800x600")
        
        # Default image path
        self.image_path = "/home/rich/MyCoding/images_general/MonaLisa-662199825.jpg"
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.grid(row=0, column=0, sticky="ew", columnspan=2)
        
        ttk.Label(file_frame, text="Image File:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.file_var = tk.StringVar(value=self.image_path)
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=50, state="readonly")
        file_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=2)
        
        file_frame.columnconfigure(1, weight=1)
        
        # Generate button
        generate_btn = ttk.Button(self.root, text="Generate Captions", command=self.generate_captions_threaded)
        generate_btn.grid(row=1, column=0, columnspan=2, pady=20)
        
        # Results area
        results_frame = ttk.Frame(self.root, padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        
        ttk.Label(results_frame, text="Generated Captions:").grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=1, column=0, sticky="nsew")
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, width=70, height=20)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Configure root grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_var.set(file_path)
            self.image_path = file_path
            
    def generate_captions_threaded(self):
        # Run caption generation in a separate thread to avoid blocking the GUI
        threading.Thread(target=self.generate_captions, daemon=True).start()
        
    def generate_captions(self):
        try:
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Generating captions...\n\n")
            self.results_text.update()
            
            # Load the image
            im = Image.open(self.image_path).convert("RGB")
            
            # Generate captions with increasing lengths
            for i in range(6):
                max_length = (i + 1) * 8  # Increase max_length
                min_length = (i + 1) * 4  # Increase min_length
                num_beams = 8  # Use beam search with 8 beams
                length_penalty = 1.25  # increase length penalty to encourage longer captions
                
                caption = generate_caption_blip(im, max_length=max_length, min_length=min_length, 
                                              num_beams=num_beams, length_penalty=length_penalty)
                
                # Update results in real-time
                result_text = f"Caption {i + 1} (max_len={max_length}): {caption}\n\n"
                self.results_text.insert(tk.END, result_text)
                self.results_text.see(tk.END)
                self.results_text.update()
                
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")

# Create and run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = BLIPCaptionerGUI(root)
    root.mainloop()