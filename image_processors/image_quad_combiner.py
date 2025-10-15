#!/usr/bin/env python3
"""
Image Quad Combiner - Creates horizontal montages of 4 images and scales to fit 2560x1440
Takes groups of four 720x1600 images, combines them horizontally to 2880x1600, 
then scales with Lanczos to fit 2560x1440 while maintaining aspect ratio.
PIL-only version - no external dependencies required.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from pathlib import Path

class ImageQuadCombiner:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Quad Combiner - 4-Image Horizontal Combiner")
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
        title_label = tk.Label(self.root, text="Image Quad Combiner", 
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
        
        self.output_path_label = tk.Label(output_frame, text="Will be created as 'input_dir/quad_combos'", 
                                         bg="lightblue", relief="sunken", font=("Arial", 10))
        self.output_path_label.pack(fill='x', pady=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.root, text="Process Info", padding=20)
        info_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        info_text = tk.Text(info_frame, height=8, font=("Arial", 10), wrap=tk.WORD)
        info_text.pack(fill='both', expand=True)
        
        info_content = """Process Description:
• Combines 4 images horizontally (720x1600 each → 2880x1600 combo)
• Scales with Lanczos to fit 2560x1440 while maintaining aspect ratio
• Final size: 2560x1422 (preserves 2880:1600 ratio)
• Saves as PNG files with "_combo" suffix
• Output saved to input_directory/quad_combos/

Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF
Images are processed in groups of 4 in alphabetical order."""
        
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
        
        # Status label with more prominent styling for better visibility
        self.status_label = tk.Label(self.root, text="Select input directory to begin", 
                                    font=("Arial", 11, "bold"), fg="blue", 
                                    relief="sunken", bg="lightgray", padx=10, pady=5)
        self.status_label.pack(pady=15)
        
    def select_input_directory(self):
        """Select input directory containing images"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir = directory
            self.input_path_label.configure(text=directory)
            
            # Update output path display
            output_path = os.path.join(directory, "quad_combos")
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
        quads = count // 4
        remaining = count % 4
        
        if count == 0:
            self.file_count_label.configure(text="No supported images found")
        else:
            text = f"Found {count} images → {quads} quad combos"
            if remaining > 0:
                text += f" ({remaining} images will be skipped)"
            self.file_count_label.configure(text=text)
    
    def update_ui_state(self):
        """Update UI state based on selections"""
        ready = (self.input_dir and len(self.image_files) >= 4)
        self.process_btn.configure(state="normal" if ready else "disabled")
        
        if ready:
            quads = len(self.image_files) // 4
            self.status_label.configure(text=f"Ready to process {quads} quad combos")
        elif not self.input_dir:
            self.status_label.configure(text="Select input directory")
        elif len(self.image_files) < 4:
            self.status_label.configure(text="Need at least 4 images in input directory")
    
    def process_images(self):
        """Process all image quads"""
        if not self.input_dir or len(self.image_files) < 4:
            messagebox.showerror("Error", "Please select input directory with at least 4 images")
            return
        
        try:
            quads = len(self.image_files) // 4
            processed = 0
            errors = 0
            
            # Create output directory
            output_dir = Path(self.input_dir) / "quad_combos"
            output_dir.mkdir(exist_ok=True)
            
            self.status_label.configure(text=f"Starting to process {quads} quad combos...")
            self.root.update()
            
            # Process quads (groups of 4)
            for i in range(0, len(self.image_files), 4):
                if i + 3 >= len(self.image_files):
                    break  # Skip if less than 4 images remaining
                
                current_quad = (i // 4) + 1
                
                try:
                    # Update status to show current quad being processed
                    self.status_label.configure(text=f"Processing quad {current_quad} of {quads} - Loading images...")
                    self.root.update()
                    
                    # Load 4 images with individual progress feedback
                    img_paths = self.image_files[i:i+4]
                    images = []
                    
                    for idx, img_path in enumerate(img_paths, 1):
                        self.status_label.configure(text=f"Processing quad {current_quad} of {quads} - Loading image {idx}/4...")
                        self.root.update()
                        
                        img = Image.open(img_path)
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img)
                    
                    # Update status for combining step
                    self.status_label.configure(text=f"Processing quad {current_quad} of {quads} - Combining images...")
                    self.root.update()
                    
                    # Create horizontal montage (2880x1600 from four 720x1600 images)
                    montage_width = sum(img.width for img in images)  # Should be 2880
                    montage_height = max(img.height for img in images)  # Should be 1600
                    montage = Image.new('RGB', (montage_width, montage_height))
                    
                    # Paste images side by side
                    x_offset = 0
                    for img in images:
                        montage.paste(img, (x_offset, 0))
                        x_offset += img.width
                    
                    # Update status for scaling step
                    self.status_label.configure(text=f"Processing quad {current_quad} of {quads} - Scaling to fit 2560x1440...")
                    self.root.update()
                    
                    # Scale to fit 2560x1440 while maintaining 2880:1600 aspect ratio
                    # Original ratio is 2880:1600 = 1.8:1
                    # Calculate scaling to fit within 2560x1440
                    scale_x = 2560 / montage_width
                    scale_y = 1440 / montage_height
                    scale = min(scale_x, scale_y)
                    
                    new_width = int(montage_width * scale)
                    new_height = int(montage_height * scale)
                    
                    # Scale with Lanczos resampling
                    final = montage.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Update status for saving step
                    self.status_label.configure(text=f"Processing quad {current_quad} of {quads} - Saving combo image...")
                    self.root.update()
                    
                    # Generate output filename using first image's name
                    base_name = img_paths[0].stem
                    output_path = output_dir / f"{base_name}_combo.png"
                    
                    # Ensure unique filename
                    counter = 1
                    while output_path.exists():
                        output_path = output_dir / f"{base_name}_combo_{counter}.png"
                        counter += 1
                    
                    # Save
                    final.save(output_path, "PNG")
                    processed += 1
                    
                    # Update progress with completion status
                    self.status_label.configure(text=f"✓ Completed quad {current_quad} of {quads} - {processed} combos finished")
                    self.root.update()
                    
                except Exception as e:
                    # Update status to show error for this quad
                    self.status_label.configure(text=f"❌ Error processing quad {current_quad} of {quads}: {str(e)[:50]}...")
                    self.root.update()
                    print(f"Error processing quad {current_quad}: {str(e)}")
                    errors += 1
            
            # Show final results with detailed feedback
            if errors == 0:
                self.status_label.configure(text=f"🎉 All done! Successfully processed {processed} quad combos")
                messagebox.showinfo("Success", 
                                  f"✅ Processing Complete!\n\n"
                                  f"Successfully processed: {processed} quad combos\n"
                                  f"Total images combined: {processed * 4}\n"
                                  f"Saved to: {output_dir}")
            else:
                self.status_label.configure(text=f"⚠️ Completed with {errors} errors - {processed} combos processed successfully")
                messagebox.showwarning("Partial Success", 
                                     f"⚠️ Processing completed with some issues\n\n"
                                     f"Successfully processed: {processed} quad combos\n"
                                     f"Failed: {errors} quad combos\n"
                                     f"Total images processed: {processed * 4}\n"
                                     f"Saved to: {output_dir}")
                
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error during processing: {str(e)}")
            self.status_label.configure(text="Error occurred during processing")
    
    def clear_selections(self):
        """Clear all selections"""
        self.input_dir = None
        self.image_files = []
        
        self.input_path_label.configure(text="No directory selected")
        self.output_path_label.configure(text="Will be created as 'input_dir/quad_combos'")
        self.file_count_label.configure(text="No images found")
        self.status_label.configure(text="Select input directory to begin")
        
        self.update_ui_state()

def main():
    root = tk.Tk()
    app = ImageQuadCombiner(root)
    root.mainloop()

if __name__ == "__main__":
    main()
