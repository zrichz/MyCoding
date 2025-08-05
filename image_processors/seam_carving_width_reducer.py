#!/usr/bin/env python3
"""
Seam Carving Width Reducer
A GUI application that reduces image width using seam carving algorithm
applied to the first and last 25% of the image width.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
from pathlib import Path


class SeamCarvingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seam Carving Width Reducer")
        self.root.geometry("800x600")
        
        self.original_image = None
        self.image_path = None
        self.processed_image = None
        self.current_photo = None  # Keep reference to displayed photo
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection
        ttk.Label(main_frame, text="Select Image:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Percentage input
        ttk.Label(main_frame, text="Width Reduction (50-99%):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.percentage_var = tk.StringVar(value="70")
        percentage_frame = ttk.Frame(main_frame)
        percentage_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        self.percentage_entry = ttk.Entry(percentage_frame, textvariable=self.percentage_var, width=10)
        self.percentage_entry.pack(side=tk.LEFT)
        ttk.Label(percentage_frame, text="%").pack(side=tk.LEFT, padx=(5, 0))
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Image", command=self.process_image, state=tk.DISABLED)
        self.process_button.grid(row=1, column=2, padx=(10, 0))
        
        # Image display area
        self.image_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.pack(expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)
    
    def browse_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image()
    
    def load_image(self):
        """Load and display the selected image"""
        if self.image_path is None:
            return
            
        try:
            self.original_image = Image.open(self.image_path)
            
            # Convert to RGB if necessary
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            
            self.display_image(self.original_image)
            self.process_button.config(state=tk.NORMAL)
            filename = os.path.basename(self.image_path) if self.image_path else "Unknown"
            self.update_status(f"Loaded: {filename} ({self.original_image.size[0]}x{self.original_image.size[1]})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image in the GUI"""
        # Resize image for display while maintaining aspect ratio
        display_size = (400, 300)
        image_copy = image.copy()
        image_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image_copy)
        self.image_label.config(image=photo, text="")
        self.current_photo = photo  # Keep a reference
    
    def update_status(self, message):
        """Update status label"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def validate_percentage(self):
        """Validate the percentage input"""
        try:
            percentage = float(self.percentage_var.get())
            if 50 <= percentage <= 99:
                return percentage
            else:
                messagebox.showerror("Invalid Input", "Percentage must be between 50 and 99")
                return None
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")
            return None
    
    def energy_function(self, image):
        """Calculate energy function using gradient magnitude"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Energy is the magnitude of gradients
        energy = np.sqrt(grad_x**2 + grad_y**2)
        return energy
    
    def find_vertical_seam(self, energy):
        """Find the minimum energy vertical seam using dynamic programming"""
        rows, cols = energy.shape
        dp = np.copy(energy)
        
        # Fill the DP table
        for i in range(1, rows):
            for j in range(cols):
                # Consider three possible paths from previous row
                candidates = []
                
                # From directly above
                candidates.append(dp[i-1, j])
                
                # From top-left (if exists)
                if j > 0:
                    candidates.append(dp[i-1, j-1])
                
                # From top-right (if exists)
                if j < cols - 1:
                    candidates.append(dp[i-1, j+1])
                
                dp[i, j] += min(candidates)
        
        # Backtrack to find the seam
        seam = []
        j = np.argmin(dp[-1])
        
        for i in range(rows - 1, -1, -1):
            seam.append(j)
            
            if i > 0:
                candidates = [dp[i-1, j]]
                indices = [j]
                
                if j > 0:
                    candidates.append(dp[i-1, j-1])
                    indices.append(j-1)
                
                if j < cols - 1:
                    candidates.append(dp[i-1, j+1])
                    indices.append(j+1)
                
                j = indices[np.argmin(candidates)]
        
        seam.reverse()
        return seam
    
    def remove_vertical_seam(self, image, seam):
        """Remove a vertical seam from the image"""
        rows, cols, channels = image.shape
        result = np.zeros((rows, cols - 1, channels), dtype=image.dtype)
        
        for i in range(rows):
            j = seam[i]
            result[i, :j] = image[i, :j]
            result[i, j:] = image[i, j+1:]
        
        return result
    
    def seam_carve_region(self, image, start_col, end_col, seams_to_remove):
        """Apply seam carving to a specific region of the image"""
        if seams_to_remove <= 0:
            return image
        
        # Extract the region
        region = image[:, start_col:end_col].copy()
        
        for seam_idx in range(seams_to_remove):
            # Calculate energy for current region
            energy = self.energy_function(region)
            
            # Find minimum energy seam
            seam = self.find_vertical_seam(energy)
            
            # Remove the seam
            region = self.remove_vertical_seam(region, seam)
            
            # Update progress
            progress = ((seam_idx + 1) / seams_to_remove) * 50  # 50% for each region
            if start_col > 0:  # Second region
                progress += 50
            self.progress_var.set(progress)
            self.root.update_idletasks()
        
        return region
    
    def process_image(self):
        """Process the image with seam carving"""
        percentage = self.validate_percentage()
        if percentage is None:
            return
        
        if self.original_image is None:
            messagebox.showerror("Error", "No image selected")
            return
        
        try:
            self.update_status("Processing image...")
            self.progress_var.set(0)
            
            # Convert PIL image to numpy array
            image_array = np.array(self.original_image)
            
            # Calculate dimensions
            height, width, channels = image_array.shape
            target_width = int(width * percentage / 100)
            seams_to_remove = width - target_width
            
            # Calculate region boundaries (first 25% and last 25%)
            first_quarter = width // 4
            last_quarter_start = width - first_quarter
            
            # Split seams between two regions
            seams_first_region = seams_to_remove // 2
            seams_last_region = seams_to_remove - seams_first_region
            
            self.update_status("Processing first 25% of image...")
            
            # Process first 25%
            first_region = self.seam_carve_region(
                image_array, 0, first_quarter, seams_first_region
            )
            
            self.update_status("Processing last 25% of image...")
            
            # Process last 25%
            last_region = self.seam_carve_region(
                image_array, last_quarter_start, width, seams_last_region
            )
            
            # Combine regions
            middle_region = image_array[:, first_quarter:last_quarter_start]
            
            # Combine all regions
            result = np.concatenate([first_region, middle_region, last_region], axis=1)
            
            # Convert back to PIL Image
            self.processed_image = Image.fromarray(result.astype(np.uint8))
            
            # Save the image
            self.save_processed_image()
            
            # Display the result
            self.display_image(self.processed_image)
            
            self.progress_var.set(100)
            self.update_status(f"Complete! Reduced from {width}px to {result.shape[1]}px width")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.progress_var.set(0)
            self.update_status("Error occurred during processing")
    
    def save_processed_image(self):
        """Save the processed image with _reduced_width suffix"""
        if self.processed_image is None or self.image_path is None:
            return
        
        # Create output filename
        path_obj = Path(self.image_path)
        output_filename = f"{path_obj.stem}_reduced_width{path_obj.suffix}"
        output_path = path_obj.parent / output_filename
        
        # Save the image
        self.processed_image.save(output_path)
        
        messagebox.showinfo("Success", f"Image saved as: {output_filename}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = SeamCarvingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
