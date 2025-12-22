"""
Generalized Image Montage Creator with GUI
Automatically detects image dimensions and creates 2x2 montages.
Works with images of any size - all images in a directory should have the same dimensions.

Examples:
- 1024x1024 images → 2048x2048 montages
- 3000x1000 images → 6000x2000 montages  
- 512x768 images → 1024x1536 montages
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
from pathlib import Path
import threading


def get_image_files(directory):
    """Get all image files from the specified directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def detect_image_dimensions(image_files):
    """Detect the dimensions of images in the directory"""
    if not image_files:
        return None, None
    
    # Check first few images to determine consistent dimensions
    dimensions_found = {}
    
    for i, image_path in enumerate(image_files[:5]):  # Check first 5 images
        try:
            with Image.open(image_path) as img:
                dims = (img.width, img.height)
                dimensions_found[dims] = dimensions_found.get(dims, 0) + 1
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue
    
    if not dimensions_found:
        return None, None
    
    # Return the most common dimensions
    most_common_dims = max(dimensions_found.items(), key=lambda x: x[1])[0]
    return most_common_dims


def load_and_validate_image(image_path, expected_dims):
    """Load an image and validate it matches expected dimensions"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check if dimensions match expected
            if (img.width, img.height) != expected_dims:
                print(f"Warning: {image_path.name} has dimensions {img.width}x{img.height}, expected {expected_dims[0]}x{expected_dims[1]}")
                # Resize to expected dimensions if they don't match
                img = img.resize(expected_dims, Image.Resampling.LANCZOS)
            
            # Return a copy since we're using context manager
            return img.copy()
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def create_montage(images, individual_size):
    """Create a 2x2 montage from 4 images"""
    if len(images) != 4:
        raise ValueError("Exactly 4 images are required for a 2x2 montage")
    
    # Calculate montage dimensions (2x2 grid)
    montage_width = individual_size[0] * 2
    montage_height = individual_size[1] * 2
    
    # Create the montage canvas
    montage = Image.new('RGB', (montage_width, montage_height), (255, 255, 255))
    
    # Position each image in the 2x2 grid
    positions = [
        (0, 0),                                    # Top-left
        (individual_size[0], 0),                   # Top-right
        (0, individual_size[1]),                   # Bottom-left
        (individual_size[0], individual_size[1])   # Bottom-right
    ]
    
    for i, img in enumerate(images):
        if img is not None:
            montage.paste(img, positions[i])
    
    return montage


def process_images_with_callback(input_directory, output_directory, progress_callback=None, completion_callback=None):
    """Process images with progress updates for GUI"""
    
    try:
        # Set output directory
        if output_directory is None:
            output_directory = Path(input_directory) / "montages"
        else:
            output_directory = Path(output_directory)
        
        # Create output directory if it doesn't exist
        output_directory.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = get_image_files(input_directory)
        
        if not image_files:
            if completion_callback:
                completion_callback(f"No image files found in {input_directory}", False)
            return
        
        # Detect image dimensions
        individual_dims = detect_image_dimensions(image_files)
        
        if individual_dims is None or individual_dims[0] is None or individual_dims[1] is None:
            if completion_callback:
                completion_callback("Could not determine image dimensions", False)
            return
        
        # Calculate montage dimensions
        montage_dims = (individual_dims[0] * 2, individual_dims[1] * 2)
        
        if progress_callback:
            progress_callback(f"Detected image size: {individual_dims[0]}x{individual_dims[1]}", 5)
            progress_callback(f"Montage size will be: {montage_dims[0]}x{montage_dims[1]}", 10)
        
        # Process images in groups of 4
        montage_count = 0
        loaded_images = []
        total_images = len(image_files)
        
        for i, image_path in enumerate(image_files):
            if progress_callback:
                progress_callback(f"Loading: {image_path.name}", (i / total_images) * 80 + 10)
            
            # Load and validate the image
            loaded_img = load_and_validate_image(image_path, individual_dims)
            
            if loaded_img is not None:
                loaded_images.append(loaded_img)
            else:
                # If image failed to process, add a white placeholder
                placeholder = Image.new('RGB', individual_dims, (255, 255, 255))
                loaded_images.append(placeholder)
            
            # Create montage when we have 4 images
            if len(loaded_images) == 4:
                montage_count += 1
                montage = create_montage(loaded_images, individual_dims)
                
                # Save the montage
                montage_filename = f"montage_{montage_count:03d}_{montage_dims[0]}x{montage_dims[1]}.jpg"
                montage_path = output_directory / montage_filename
                montage.save(montage_path, quality=90, optimize=True)
                
                if progress_callback:
                    progress_callback(f"Created: {montage_filename}", (i / total_images) * 80 + 10)
                
                # Reset for next group
                loaded_images = []
        
        # Handle remaining images (less than 4)
        if loaded_images:
            # Fill remaining slots with white images
            while len(loaded_images) < 4:
                placeholder = Image.new('RGB', individual_dims, (255, 255, 255))
                loaded_images.append(placeholder)
            
            montage_count += 1
            montage = create_montage(loaded_images, individual_dims)
            
            montage_filename = f"montage_{montage_count:03d}_{montage_dims[0]}x{montage_dims[1]}_partial.jpg"
            montage_path = output_directory / montage_filename
            montage.save(montage_path, quality=90, optimize=True)
        
        # Final completion
        result_message = f"""Processing Complete!
        
📊 Summary:
• Input image size: {individual_dims[0]}x{individual_dims[1]}
• Montage size: {montage_dims[0]}x{montage_dims[1]}
• Total images processed: {total_images}
• Montages created: {montage_count}
• Output location: {output_directory}

✅ All montages saved successfully!"""
        
        if completion_callback:
            completion_callback(result_message, True)
            
    except Exception as e:
        error_message = f"Error during processing: {str(e)}"
        if completion_callback:
            completion_callback(error_message, False)


class GeneralizedMontageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generalized Image Montage Creator")
        self.root.geometry("600x450")
        self.root.resizable(True, True)
        
        self.input_directory = None
        self.output_directory = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the user interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🖼️ Generalized Image Montage Creator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Description
        desc_text = ("Auto-detects image dimensions and creates 2x2 montages\n"
                    "Works with any image size (e.g., 3000x1000 → 6000x2000)")
        desc_label = ttk.Label(main_frame, text=desc_text, 
                              justify=tk.CENTER, foreground="gray")
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Input directory selection
        ttk.Label(main_frame, text="Input Directory:").grid(row=2, column=0, sticky="w", pady=5)
        self.input_var = tk.StringVar(value="No directory selected")
        self.input_label = ttk.Label(main_frame, textvariable=self.input_var, 
                                    foreground="blue", relief="sunken", padding=5)
        self.input_label.grid(row=2, column=1, sticky="ew", padx=(10, 5), pady=5)
        
        self.input_button = ttk.Button(main_frame, text="Browse...", 
                                      command=self.select_input_directory)
        self.input_button.grid(row=2, column=2, padx=(5, 0), pady=5)
        
        # Output directory selection (optional)
        ttk.Label(main_frame, text="Output Directory:").grid(row=3, column=0, sticky="w", pady=5)
        self.output_var = tk.StringVar(value="Auto (input_dir/montages)")
        self.output_label = ttk.Label(main_frame, textvariable=self.output_var, 
                                     foreground="gray", relief="sunken", padding=5)
        self.output_label.grid(row=3, column=1, sticky="ew", padx=(10, 5), pady=5)
        
        self.output_button = ttk.Button(main_frame, text="Browse...", 
                                       command=self.select_output_directory)
        self.output_button.grid(row=3, column=2, padx=(5, 0), pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="🚀 Create Montages", 
                                        command=self.start_processing, state="disabled")
        self.process_button.grid(row=4, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to process images...")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Results text area
        self.result_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        self.result_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=10)
        self.result_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        self.result_text = tk.Text(self.result_frame, height=8, wrap=tk.WORD, 
                                  font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_frame.rowconfigure(0, weight=1)
        
        # Initial result text
        self.result_text.insert(tk.END, "Welcome to Generalized Image Montage Creator!\n\n")
        self.result_text.insert(tk.END, "Features:\n")
        self.result_text.insert(tk.END, "• Auto-detects image dimensions in directory\n")
        self.result_text.insert(tk.END, "• Creates 2x2 montages (4 images per montage)\n")
        self.result_text.insert(tk.END, "• Works with any image size\n")
        self.result_text.insert(tk.END, "• Examples:\n")
        self.result_text.insert(tk.END, "  - 1024x1024 → 2048x2048 montages\n")
        self.result_text.insert(tk.END, "  - 3000x1000 → 6000x2000 montages\n")
        self.result_text.insert(tk.END, "  - 512x768 → 1024x1536 montages\n\n")
        self.result_text.insert(tk.END, "Instructions:\n")
        self.result_text.insert(tk.END, "1. Select a directory containing images\n")
        self.result_text.insert(tk.END, "2. Optionally choose output directory\n")
        self.result_text.insert(tk.END, "3. Click 'Create Montages' to start\n\n")
        self.result_text.insert(tk.END, "Supported formats: JPG, PNG, BMP, GIF, TIFF, WebP\n")
        self.result_text.config(state=tk.DISABLED)
    
    def select_input_directory(self):
        """Open directory selection dialog for input"""
        directory = filedialog.askdirectory(title="Select Input Directory with Images")
        if directory:
            self.input_directory = directory
            # Truncate long paths for display
            display_path = directory
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            self.input_var.set(display_path)
            self.input_label.config(foreground="black")
            
            # Check for images and detect dimensions
            image_files = get_image_files(directory)
            self.update_result_text(f"\n📁 Selected: {directory}")
            self.update_result_text(f"🖼️  Found {len(image_files)} image files")
            
            if len(image_files) > 0:
                # Detect dimensions
                dims = detect_image_dimensions(image_files)
                if dims and dims[0] is not None and dims[1] is not None:
                    montage_dims = (dims[0] * 2, dims[1] * 2)
                    self.update_result_text(f"📏 Detected image size: {dims[0]}x{dims[1]}")
                    self.update_result_text(f"🔧 Montage size will be: {montage_dims[0]}x{montage_dims[1]}")
                    self.process_button.config(state="normal")
                    self.update_result_text("✅ Ready to create montages!")
                else:
                    self.process_button.config(state="disabled")
                    self.update_result_text("❌ Could not detect image dimensions")
            else:
                self.process_button.config(state="disabled")
                self.update_result_text("❌ No supported image files found")
    
    def select_output_directory(self):
        """Open directory selection dialog for output"""
        directory = filedialog.askdirectory(title="Select Output Directory for Montages")
        if directory:
            self.output_directory = directory
            # Truncate long paths for display
            display_path = directory
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            self.output_var.set(display_path)
            self.output_label.config(foreground="black")
            self.update_result_text(f"\n📤 Output: {directory}")
    
    def update_result_text(self, message):
        """Add message to result text area"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.root.update()
    
    def update_progress(self, message, percentage):
        """Update progress bar and label"""
        self.progress_var.set(message)
        self.progress_bar['value'] = percentage
        self.root.update()
    
    def on_completion(self, message, success):
        """Handle processing completion"""
        self.update_result_text(f"\n{message}")
        
        if success:
            self.progress_var.set("✅ Processing completed successfully!")
            self.progress_bar['value'] = 100
        else:
            self.progress_var.set("❌ Processing failed")
            self.progress_bar['value'] = 0
        
        self.process_button.config(state="normal", text="🚀 Create Montages")
        
        if success:
            # Ask if user wants to open output folder
            if messagebox.askyesno("Success!", "Montages created successfully!\n\nWould you like to open the output folder?"):
                if self.output_directory:
                    output_dir = self.output_directory
                elif self.input_directory:
                    output_dir = str(Path(self.input_directory) / "montages")
                else:
                    return
                os.startfile(output_dir)  # Windows-specific
    
    def start_processing(self):
        """Start the image processing in a separate thread"""
        if not self.input_directory:
            messagebox.showerror("Error", "Please select an input directory first!")
            return
        
        # Disable button and update UI
        self.process_button.config(state="disabled", text="Processing...")
        self.progress_bar['value'] = 0
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        
        # Start processing in separate thread
        thread = threading.Thread(target=process_images_with_callback,
                                 args=(self.input_directory, self.output_directory, 
                                      self.update_progress, self.on_completion))
        thread.daemon = True
        thread.start()


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = GeneralizedMontageGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
