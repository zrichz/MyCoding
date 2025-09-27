#!/usr/bin/env python3
"""
Image Pair Combiner - Combines images horizontally in pairs, crops to square, and scales down
Takes pairs of 720x1600 images, combines them to 1440x1600, crops to 1440x1440 from top, then scales to 512x512
PIL-only version - no external dependencies required.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from pathlib import Path

class ImagePairCombiner:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Pair Combiner - Horizontal Combine & Scale")
        self.root.geometry("800x600")
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Store directory path and files
        self.input_dir = None
        self.image_files = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="Image Pair Combiner", 
                              font=("Arial", 20, "bold"))
        title_label.pack(pady=20)
        
        # Input directory selection
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=15, padx=40, fill='x')
        
        input_label = tk.Label(input_frame, text="Input Directory:", font=("Arial", 12, "bold"))
        input_label.pack(anchor='w')
        
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill='x', pady=5)
        
        self.input_path_label = tk.Label(input_path_frame, text="No directory selected", 
                                        bg="lightgray", relief="sunken", font=("Arial", 10))
        self.input_path_label.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 10))
        
        input_btn = ttk.Button(input_path_frame, text="Browse", 
                              command=self.select_input_directory, width=12)
        input_btn.pack(side=tk.RIGHT)
        
        # Output info
        output_frame = ttk.Frame(self.root)
        output_frame.pack(pady=15, padx=40, fill='x')
        
        output_label = tk.Label(output_frame, text="Output Directory:", font=("Arial", 12, "bold"))
        output_label.pack(anchor='w')
        
        self.output_path_label = tk.Label(output_frame, text="Will be created as 'input_dir/512x512pairs'", 
                                         bg="lightblue", relief="sunken", font=("Arial", 10))
        self.output_path_label.pack(fill='x', pady=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.root, text="Process Info", padding=20)
        info_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        info_text = tk.Text(info_frame, height=8, font=("Arial", 10), wrap=tk.WORD)
        info_text.pack(fill='both', expand=True)
        
        info_content = """Process Description:
• Combines pairs of images horizontally (720x1600 → 1440x1600)
• Crops combined image to 1440x1440 from the top
• Scales result down to 512x512
• Saves as PNG files with "_combined" suffix
• Output saved to input_directory/512x512pairs/

Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF
Images are processed in alphabetical order."""
        
        info_text.insert(tk.END, info_content)
        info_text.configure(state='disabled')
        
        # File count label
        self.file_count_label = tk.Label(self.root, text="No images found", 
                                        font=("Arial", 11))
        self.file_count_label.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.process_btn = ttk.Button(button_frame, text="Process Images", 
                                     command=self.process_images, state="disabled", width=15)
        self.process_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = ttk.Button(button_frame, text="Clear", 
                              command=self.clear_selections, width=10)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Select input and output directories to begin", 
                                    font=("Arial", 10))
        self.status_label.pack(pady=10)
        
    def select_input_directory(self):
        """Select input directory containing images"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_path_label.configure(text=directory)
            
            # Update output path display
            output_path = os.path.join(directory, "512x512pairs")
            self.output_path_label.configure(text=output_path)
            
            self.scan_images()
            self.update_ui_state()
    
    def scan_images(self):
        """Scan input directory for image files"""
        if not self.input_dir:
            return
            
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        
        self.image_files = []
        input_path = Path(self.input_dir)
        
        for file_path in sorted(input_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                self.image_files.append(file_path)
        
        # Update file count display
        count = len(self.image_files)
        pairs = count // 2
        remaining = count % 2
        
        if count == 0:
            self.file_count_label.configure(text="No supported images found")
        else:
            text = f"Found {count} images → {pairs} pairs"
            if remaining > 0:
                text += f" ({remaining} image will be skipped)"
            self.file_count_label.configure(text=text)
    
    def update_ui_state(self):
        """Update UI state based on selections"""
        ready = (self.input_dir and len(self.image_files) >= 2)
        self.process_btn.configure(state="normal" if ready else "disabled")
        
        if ready:
            pairs = len(self.image_files) // 2
            self.status_label.configure(text=f"Ready to process {pairs} image pairs")
        elif not self.input_dir:
            self.status_label.configure(text="Select input directory")
        elif len(self.image_files) < 2:
            self.status_label.configure(text="Need at least 2 images in input directory")
    
    def process_images(self):
        """Process all image pairs"""
        if not self.input_dir or len(self.image_files) < 2:
            messagebox.showerror("Error", "Please select input directory with at least 2 images")
            return
        
        try:
            pairs = len(self.image_files) // 2
            processed = 0
            errors = 0
            
            # Create output directory
            output_dir = Path(self.input_dir) / "512x512pairs"
            output_dir.mkdir(exist_ok=True)
            
            self.status_label.configure(text="Processing images...")
            self.root.update()
            
            # Process pairs
            for i in range(0, len(self.image_files), 2):
                if i + 1 >= len(self.image_files):
                    break  # Skip last image if odd number
                
                try:
                    img1_path = self.image_files[i]
                    img2_path = self.image_files[i + 1]
                    
                    # Load images
                    img1 = Image.open(img1_path)
                    img2 = Image.open(img2_path)
                    
                    # Convert to RGB if needed
                    if img1.mode != 'RGB':
                        img1 = img1.convert('RGB')
                    if img2.mode != 'RGB':
                        img2 = img2.convert('RGB')
                    
                    # Combine horizontally
                    combined_width = img1.width + img2.width
                    combined_height = max(img1.height, img2.height)
                    combined = Image.new('RGB', (combined_width, combined_height))
                    
                    # Paste images side by side
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))
                    
                    # Crop to square from top (1440x1440 assuming 720+720 width)
                    crop_size = min(combined.width, combined.height)
                    cropped = combined.crop((0, 0, crop_size, crop_size))
                    
                    # Scale to 512x512
                    final = cropped.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Generate output filename
                    base_name = img1_path.stem
                    output_path = output_dir / f"{base_name}_combined.png"
                    
                    # Ensure unique filename
                    counter = 1
                    while output_path.exists():
                        output_path = output_dir / f"{base_name}_combined_{counter}.png"
                        counter += 1
                    
                    # Save
                    final.save(output_path, "PNG")
                    processed += 1
                    
                    # Update progress
                    self.status_label.configure(text=f"Processing... {processed}/{pairs} pairs complete")
                    self.root.update()
                    
                except Exception as e:
                    print(f"Error processing pair {i//2 + 1}: {str(e)}")
                    errors += 1
            
            # Show results
            if errors == 0:
                self.status_label.configure(text=f"Complete! Processed {processed} image pairs")
                messagebox.showinfo("Success", f"Successfully processed {processed} image pairs!\nSaved to: {output_dir}")
            else:
                self.status_label.configure(text=f"Complete with {errors} errors. {processed} pairs processed")
                messagebox.showwarning("Partial Success", f"Processed {processed} pairs with {errors} errors.\nSaved to: {output_dir}")
                
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error during processing: {str(e)}")
            self.status_label.configure(text="Error occurred during processing")
    
    def clear_selections(self):
        """Clear all selections"""
        self.input_dir = None
        self.image_files = []
        
        self.input_path_label.configure(text="No directory selected")
        self.output_path_label.configure(text="Will be created as 'input_dir/512x512pairs'")
        self.file_count_label.configure(text="No images found")
        self.status_label.configure(text="Select input directory to begin")
        
        self.update_ui_state()

def main():
    root = tk.Tk()
    app = ImagePairCombiner(root)
    root.mainloop()

if __name__ == "__main__":
    main()
