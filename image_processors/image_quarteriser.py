#!/home/rich/MyCoding/image_processors/.venv/bin/python3
"""
Image Quarteriser - Split images into four equal quarters
Uses tkinter for directory selection and shows progress to user
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import threading
import time

class ImageQuarteriser:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Quarteriser")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Variables
        self.selected_directory = tk.StringVar()
        self.is_processing = False
        self.total_images = 0
        self.processed_images = 0
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Image Quarteriser", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Directory selection
        ttk.Label(main_frame, text="Select Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.dir_entry = ttk.Entry(main_frame, textvariable=self.selected_directory, width=50)
        self.dir_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        
        self.browse_button = ttk.Button(main_frame, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=1, column=2, padx=(5, 0), pady=5)
        
        # Instructions
        instructions = """Instructions:
1. Select a directory containing images
2. Click 'Start Processing' to split all images into quarters
3. Each image will be split into 4 parts: _TL (top-left), _TR (top-right), _BL (bottom-left), _BR (bottom-right)
4. Quarter images will be saved in a 'quarters' subdirectory
5. Image dimensions will be adjusted to be divisible by 2 if needed

Supported formats: JPG, PNG, BMP, TIFF, GIF, WebP"""
        
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instructions_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        instructions_frame.columnconfigure(0, weight=1)
        
        instructions_label = ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT)
        instructions_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready to process images...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.detail_var = tk.StringVar(value="")
        self.detail_label = ttk.Label(progress_frame, textvariable=self.detail_var, font=("Arial", 9))
        self.detail_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Results text area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        self.results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def browse_directory(self):
        """Open directory selection dialog"""
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if directory:
            self.selected_directory.set(directory)
            self.scan_directory()
    
    def scan_directory(self):
        """Scan selected directory for images"""
        directory = self.selected_directory.get()
        if not directory or not os.path.exists(directory):
            return
            
        image_files = []
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                image_files.append(filename)
        
        self.total_images = len(image_files)
        if self.total_images > 0:
            self.progress_var.set(f"Found {self.total_images} image(s) ready to process")
            self.log_message(f"Scanned directory: {directory}")
            self.log_message(f"Found {self.total_images} image files")
        else:
            self.progress_var.set("No supported images found in selected directory")
            self.log_message("No supported image files found")
    
    def log_message(self, message):
        """Add message to results text area"""
        timestamp = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_processing(self):
        """Start processing images in a separate thread"""
        directory = self.selected_directory.get()
        if not directory or not os.path.exists(directory):
            messagebox.showerror("Error", "Please select a valid directory first")
            return
        
        if self.total_images == 0:
            messagebox.showwarning("Warning", "No supported images found in the selected directory")
            return
        
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.DISABLED)
        
        # Start processing in separate thread to keep GUI responsive
        self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the processing"""
        self.is_processing = False
        self.progress_var.set("Stopping processing...")
        self.log_message("Processing stopped by user")
    
    def process_images(self):
        """Process all images in the selected directory"""
        directory = self.selected_directory.get()
        quarters_dir = os.path.join(directory, "quarters")
        
        try:
            # Create quarters subdirectory
            os.makedirs(quarters_dir, exist_ok=True)
            self.log_message(f"Created quarters directory: {quarters_dir}")
            
            # Get list of image files
            image_files = []
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                    image_files.append(filename)
            
            self.total_images = len(image_files)
            self.processed_images = 0
            self.progress_bar['maximum'] = self.total_images
            
            # Process each image
            for i, filename in enumerate(image_files):
                if not self.is_processing:
                    break
                
                try:
                    # Update progress
                    self.progress_var.set(f"Processing {i+1}/{self.total_images}: {filename}")
                    self.detail_var.set(f"Loading and analyzing image...")
                    
                    # Load image
                    input_path = os.path.join(directory, filename)
                    with Image.open(input_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        original_width, original_height = img.size
                        
                        # Ensure dimensions are divisible by 2
                        width = original_width if original_width % 2 == 0 else original_width - 1
                        height = original_height if original_height % 2 == 0 else original_height - 1
                        
                        if width != original_width or height != original_height:
                            img = img.resize((width, height), Image.Resampling.LANCZOS)
                            self.log_message(f"  Resized {filename} from {original_width}x{original_height} to {width}x{height}")
                        
                        # Calculate quarter dimensions
                        quarter_width = width // 2
                        quarter_height = height // 2
                        
                        self.detail_var.set(f"Splitting into quarters ({quarter_width}x{quarter_height} each)...")
                        
                        # Extract quarters
                        quarters = {
                            '_TL': img.crop((0, 0, quarter_width, quarter_height)),  # Top-left
                            '_TR': img.crop((quarter_width, 0, width, quarter_height)),  # Top-right
                            '_BL': img.crop((0, quarter_height, quarter_width, height)),  # Bottom-left
                            '_BR': img.crop((quarter_width, quarter_height, width, height))  # Bottom-right
                        }
                        
                        # Save quarters
                        base_name = os.path.splitext(filename)[0]
                        extension = os.path.splitext(filename)[1]
                        
                        saved_count = 0
                        for suffix, quarter_img in quarters.items():
                            if not self.is_processing:
                                break
                                
                            quarter_filename = f"{base_name}{suffix}{extension}"
                            quarter_path = os.path.join(quarters_dir, quarter_filename)
                            quarter_img.save(quarter_path, quality=95 if extension.lower() in ['.jpg', '.jpeg'] else None)
                            saved_count += 1
                        
                        if saved_count == 4:
                            self.log_message(f"  ✓ Successfully split {filename} into 4 quarters")
                        else:
                            self.log_message(f"  ⚠ Partially processed {filename} ({saved_count}/4 quarters saved)")
                
                except Exception as e:
                    self.log_message(f"  ✗ Error processing {filename}: {str(e)}")
                
                # Update progress
                self.processed_images += 1
                self.progress_bar['value'] = self.processed_images
                self.root.update_idletasks()
            
            # Final status
            if self.is_processing:
                self.progress_var.set(f"✓ Completed! Processed {self.processed_images}/{self.total_images} images")
                self.detail_var.set(f"All quarter images saved in: {quarters_dir}")
                self.log_message(f"✓ Processing completed successfully!")
                self.log_message(f"  Total images processed: {self.processed_images}")
                self.log_message(f"  Total quarters created: {self.processed_images * 4}")
                self.log_message(f"  Output directory: {quarters_dir}")
                
                # Show completion message
                messagebox.showinfo("Complete", 
                    f"Successfully processed {self.processed_images} images!\n"
                    f"Created {self.processed_images * 4} quarter images.\n"
                    f"Saved in: {quarters_dir}")
            else:
                self.progress_var.set(f"Stopped - Processed {self.processed_images}/{self.total_images} images")
                self.detail_var.set("Processing was stopped by user")
        
        except Exception as e:
            self.log_message(f"✗ Fatal error: {str(e)}")
            self.progress_var.set("Error occurred during processing")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        finally:
            # Re-enable controls
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.browse_button.config(state=tk.NORMAL)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting Image Quarteriser...")
    app = ImageQuarteriser()
    app.run()

if __name__ == "__main__":
    main()
