#!/usr/bin/env python3
"""
Canny Batch Processor - Apply Canny edge detection to all images in a directory
Processes all images in a selected directory and saves Canny edge detection results
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageFilter
import cv2
import numpy as np
import os
import threading

class CannyBatchProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Canny Batch Processor")
        self.root.geometry("800x700")  # Larger window for 1920x1080 screen
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Variables
        self.input_directory = None
        self.output_directory = None
        self.processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main title - Larger font for 1920x1080
        title_label = tk.Label(self.root, text="Canny Edge Detection Batch Processor", 
                              font=("Arial", 20, "bold"), fg="navy")
        title_label.pack(pady=25)
        
        # Description - Larger font for better visibility
        desc_label = tk.Label(self.root, 
                             text="Select a directory with images to apply Canny edge detection\nResults will be saved in a 'canny' subdirectory",
                             font=("Arial", 13), justify=tk.CENTER)
        desc_label.pack(pady=15)
        
        # Directory selection frame
        dir_frame = ttk.LabelFrame(self.root, text="Directory Selection", padding=20)
        dir_frame.pack(pady=20, padx=30, fill='x')
        
        # Input directory
        input_frame = ttk.Frame(dir_frame)
        input_frame.pack(fill='x', pady=10)
        
        tk.Label(input_frame, text="Input Directory:", font=("Arial", 10, "bold")).pack(anchor='w')
        
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill='x', pady=5)
        
        self.input_path_var = tk.StringVar(value="No directory selected")
        input_label = tk.Label(input_path_frame, textvariable=self.input_path_var, 
                              bg="white", relief="sunken", anchor="w")
        input_label.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 10))
        
        select_btn = ttk.Button(input_path_frame, text="Browse", 
                               command=self.select_input_directory, width=10)
        select_btn.pack(side=tk.RIGHT)
        
        # Canny parameters frame - Larger fonts for better visibility
        params_frame = ttk.LabelFrame(self.root, text="Canny Parameters", padding=25)
        params_frame.pack(pady=15, padx=40, fill='x')
        
        # Low threshold
        low_frame = ttk.Frame(params_frame)
        low_frame.pack(fill='x', pady=8)
        
        tk.Label(low_frame, text="Low Threshold:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.low_threshold = tk.IntVar(value=50)
        low_spinbox = tk.Spinbox(low_frame, from_=1, to=255, textvariable=self.low_threshold, 
                                width=10, font=("Arial", 12))
        low_spinbox.pack(side=tk.RIGHT)
        
        # High threshold
        high_frame = ttk.Frame(params_frame)
        high_frame.pack(fill='x', pady=8)
        
        tk.Label(high_frame, text="High Threshold:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.high_threshold = tk.IntVar(value=150)
        high_spinbox = tk.Spinbox(high_frame, from_=1, to=255, textvariable=self.high_threshold, 
                                 width=10, font=("Arial", 12))
        high_spinbox.pack(side=tk.RIGHT)
        
        # Aperture size
        aperture_frame = ttk.Frame(params_frame)
        aperture_frame.pack(fill='x', pady=8)
        
        tk.Label(aperture_frame, text="Aperture Size:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.aperture_size = tk.IntVar(value=3)
        aperture_spinbox = tk.Spinbox(aperture_frame, from_=3, to=7, increment=2, 
                                     textvariable=self.aperture_size, width=10, font=("Arial", 12))
        aperture_spinbox.pack(side=tk.RIGHT)
        
        # Processing controls - Make them more prominent
        controls_frame = ttk.LabelFrame(self.root, text="Processing Controls", padding=20)
        controls_frame.pack(pady=20, padx=30, fill='x')
        
        button_container = ttk.Frame(controls_frame)
        button_container.pack()
        
        self.process_btn = ttk.Button(button_container, text="🚀 PROCESS IMAGES", 
                                     command=self.start_processing, width=20,
                                     style='Accent.TButton')
        self.process_btn.pack(side=tk.LEFT, padx=15)
        
        self.stop_btn = ttk.Button(button_container, text="⏹ STOP", 
                                  command=self.stop_processing, width=12, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=15)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=15)
        progress_frame.pack(pady=10, padx=30, fill='x')
        
        self.progress_var = tk.StringVar(value="Ready to process images")
        self.progress_label = tk.Label(progress_frame, textvariable=self.progress_var, 
                                      font=("Arial", 10))
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=10, fill='x')
        
        # Status label
        self.status_label = tk.Label(self.root, text="Select input directory to begin", 
                                    font=("Arial", 10))
        self.status_label.pack(pady=10)
        
    def select_input_directory(self):
        """Select input directory containing images"""
        directory = filedialog.askdirectory(title="Select Directory with Images")
        
        if directory:
            self.input_directory = directory
            # Truncate path if too long for display
            display_path = directory
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]
            
            self.input_path_var.set(display_path)
            
            # Set output directory (input_dir/canny)
            self.output_directory = os.path.join(directory, "canny")
            
            # Count images in directory
            image_count = self.count_images(directory)
            self.status_label.configure(text=f"Found {image_count} image files in directory")
            
            # Enable process button if images found
            if image_count > 0:
                self.process_btn.configure(state='normal')
                self.status_label.configure(text=f"Ready to process {image_count} image files", 
                                          fg="green", font=("Arial", 11, "bold"))
            else:
                self.process_btn.configure(state='disabled')
                self.status_label.configure(text="No image files found in directory", 
                                          fg="orange", font=("Arial", 11))
        
    def count_images(self, directory):
        """Count image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
        count = 0
        
        try:
            for filename in os.listdir(directory):
                if os.path.splitext(filename.lower())[1] in image_extensions:
                    count += 1
        except Exception:
            pass
            
        return count
    
    def get_image_files(self, directory):
        """Get list of image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
        image_files = []
        
        try:
            for filename in os.listdir(directory):
                if os.path.splitext(filename.lower())[1] in image_extensions:
                    image_files.append(filename)
        except Exception as e:
            print(f"Error reading directory: {e}")
            
        return sorted(image_files)
    
    def apply_canny_filter(self, image_path, output_path):
        """Apply Canny edge detection to a single image"""
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False, "Could not load image"
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 
                             self.low_threshold.get(), 
                             self.high_threshold.get(), 
                             apertureSize=self.aperture_size.get())
            
            # Save the result
            cv2.imwrite(output_path, edges)
            
            return True, "Success"
            
        except Exception as e:
            return False, str(e)
    
    def start_processing(self):
        """Start processing images in a separate thread"""
        if not self.input_directory:
            messagebox.showwarning("No Directory", "Please select an input directory first")
            return
        
        # Start processing in background thread
        self.processing = True
        self.process_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        
        # Start processing thread
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
    
    def stop_processing(self):
        """Stop processing"""
        self.processing = False
        self.process_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.progress_var.set("Processing stopped by user")
        
    def process_images(self):
        """Process all images with Canny edge detection"""
        try:
            # Create output directory
            os.makedirs(self.output_directory, exist_ok=True)
            
            # Get list of image files
            image_files = self.get_image_files(self.input_directory)
            total_files = len(image_files)
            
            if total_files == 0:
                self.root.after(0, lambda: self.progress_var.set("No image files found"))
                self.root.after(0, self.processing_complete)
                return
            
            processed = 0
            errors = 0
            
            for i, filename in enumerate(image_files):
                if not self.processing:  # Check if stopped
                    break
                
                # Update progress
                self.root.after(0, lambda f=filename: self.progress_var.set(f"Processing: {f}"))
                self.root.after(0, lambda v=int((i/total_files)*100): self.progress_bar.configure(value=v))
                
                # Process image
                input_path = os.path.join(self.input_directory, filename)
                # Change extension to .png for Canny output
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_canny.png"
                output_path = os.path.join(self.output_directory, output_filename)
                
                success, message = self.apply_canny_filter(input_path, output_path)
                
                if success:
                    processed += 1
                else:
                    errors += 1
                    print(f"Error processing {filename}: {message}")
            
            # Update final progress
            self.root.after(0, lambda: self.progress_bar.configure(value=100))
            
            if self.processing:  # Only show success if not stopped
                self.root.after(0, lambda: self.progress_var.set(
                    f"Complete! Processed {processed} images, {errors} errors"))
                
                if errors == 0:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Processing Complete", 
                        f"Successfully processed {processed} images!\nResults saved in: {self.output_directory}"))
                else:
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Processing Complete with Errors", 
                        f"Processed {processed} images with {errors} errors.\nResults saved in: {self.output_directory}"))
            
            self.root.after(0, self.processing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
            self.root.after(0, self.processing_complete)
    
    def processing_complete(self):
        """Reset UI after processing is complete"""
        self.processing = False
        self.process_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')

def main():
    # Check if required packages are available
    missing_packages = []
    
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
    
    try:
        import numpy as np
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing_packages.append("pillow")
    
    if missing_packages:
        error_msg = f"Missing required packages: {', '.join(missing_packages)}\n\n"
        error_msg += "Please install with:\n"
        error_msg += f"pip install {' '.join(missing_packages)}\n\n"
        error_msg += "Or use the launcher scripts which will install automatically."
        
        # Show error in GUI if possible, otherwise print
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Missing Dependencies", error_msg)
            root.destroy()
        except:
            print(error_msg)
        return
    
    root = tk.Tk()
    app = CannyBatchProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
