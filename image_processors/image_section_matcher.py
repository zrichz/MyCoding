#!/usr/bin/env python3
"""
Image Section Matcher - Creates new images by matching 16x16 sections between two source images
Loads two 512x512 images, divides into sections, and finds best-fit matches using similarity matrices
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import random
from datetime import datetime

class ImageSectionMatcher:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Section Matcher")
        self.root.geometry("1600x900")  # Large window for full HD screen
        self.root.configure(bg='#808080')  # Mid-grey background
        
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Variables
        self.source_image = None
        self.target_image = None
        self.result_image = None
        self.section_size = 16  # Size of each section (16x16 pixels)
        
        # Similarity methods
        self.similarity_methods = {
            "Mean Squared Error": self.calculate_mse,
            "Structural Similarity": self.calculate_ssim,
            "Normalized Cross-Correlation": self.calculate_ncc,
            "Histogram Correlation": self.calculate_hist_corr
        }
        self.current_similarity_method = "Mean Squared Error"
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=15)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # File selection frame
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill='x', pady=5)
        
        # Source image selection
        source_frame = ttk.LabelFrame(file_frame, text="Source Image (sections to match)", padding=10)
        source_frame.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        self.source_path_var = tk.StringVar(value="No source image selected")
        source_label = ttk.Label(source_frame, textvariable=self.source_path_var, foreground="gray")
        source_label.pack(pady=5)
        
        ttk.Button(source_frame, text="Browse Source Image", 
                  command=self.select_source_image).pack()
        
        # Target image selection  
        target_frame = ttk.LabelFrame(file_frame, text="Target Image (sections to choose from)", padding=10)
        target_frame.pack(side=tk.RIGHT, fill='x', expand=True, padx=(5, 0))
        
        self.target_path_var = tk.StringVar(value="No target image selected")
        target_label = ttk.Label(target_frame, textvariable=self.target_path_var, foreground="gray")
        target_label.pack(pady=5)
        
        ttk.Button(target_frame, text="Browse Target Image", 
                  command=self.select_target_image).pack()
        
        # Settings frame
        settings_frame = ttk.LabelFrame(control_frame, text="Matching Settings", padding=15)
        settings_frame.pack(fill='x', pady=10)
        
        # Algorithm info
        info_frame = ttk.Frame(settings_frame)
        info_frame.pack(fill='x', pady=5)
        
        info_label = ttk.Label(info_frame, 
                              text="🔀 Sections processed in random order, each target section used only once",
                              font=("Arial", 9), foreground="blue")
        info_label.pack()
        
        # Section size selection
        size_frame = ttk.Frame(settings_frame)
        size_frame.pack(fill='x', pady=5)
        
        ttk.Label(size_frame, text="Section Size:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.section_size_var = tk.IntVar(value=16)
        size_options = [8, 16, 32, 64]
        
        for size in size_options:
            rb = ttk.Radiobutton(size_frame, text=f"{size}×{size} pixels", 
                               variable=self.section_size_var, value=size,
                               command=self.update_section_size)
            rb.pack(side=tk.LEFT, padx=20)
        
        # Similarity method selection
        similarity_frame = ttk.Frame(settings_frame)
        similarity_frame.pack(fill='x', pady=5)
        
        ttk.Label(similarity_frame, text="Similarity Method:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.similarity_var = tk.StringVar(value="Mean Squared Error")
        similarity_combo = ttk.Combobox(similarity_frame, textvariable=self.similarity_var,
                                      values=list(self.similarity_methods.keys()),
                                      state="readonly", width=25)
        similarity_combo.pack(side=tk.LEFT, padx=10)
        similarity_combo.bind('<<ComboboxSelected>>', self.update_similarity_method)
        
        # Optimization settings
        optimization_frame = ttk.Frame(settings_frame)
        optimization_frame.pack(fill='x', pady=5)
        
        self.enable_optimization_var = tk.BooleanVar(value=True)
        optimization_check = ttk.Checkbutton(optimization_frame, 
                                           text="Enable swap optimization (2000 random swaps)",
                                           variable=self.enable_optimization_var)
        optimization_check.pack(side=tk.LEFT)
        
        # Generate button
        self.generate_btn = ttk.Button(control_frame, text="🔄 Generate Matched Image",
                                     command=self.generate_matched_image, state='disabled')
        self.generate_btn.pack(pady=15)
        
        # Progress
        self.progress_var = tk.StringVar(value="Select source and target images to begin")
        progress_label = ttk.Label(control_frame, textvariable=self.progress_var, font=("Arial", 10))
        progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(control_frame, length=400, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        
        # Image display area (3 columns)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill='both', expand=True)
        
        # Source image panel
        source_display_frame = ttk.LabelFrame(image_frame, text="Source Image", padding=10)
        source_display_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=(0, 5))
        
        self.source_canvas = tk.Canvas(source_display_frame, width=256, height=256, bg='white')
        self.source_canvas.pack()
        
        # Target image panel
        target_display_frame = ttk.LabelFrame(image_frame, text="Target Image", padding=10)
        target_display_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5)
        
        self.target_canvas = tk.Canvas(target_display_frame, width=256, height=256, bg='white')
        self.target_canvas.pack()
        
        # Result image panel
        result_display_frame = ttk.LabelFrame(image_frame, text="Matched Result", padding=10)
        result_display_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=(5, 0))
        
        self.result_canvas = tk.Canvas(result_display_frame, width=256, height=256, bg='white')
        self.result_canvas.pack()
        
        # Save button
        save_btn = ttk.Button(result_display_frame, text="Save Result", 
                             command=self.save_result_image)
        save_btn.pack(pady=5)
        
    def update_section_size(self):
        """Update section size setting"""
        self.section_size = self.section_size_var.get()
        
    def update_similarity_method(self, event=None):
        """Update similarity calculation method"""
        self.current_similarity_method = self.similarity_var.get()
        
    def select_source_image(self):
        """Select source image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Load and validate image
                img = Image.open(filepath)
                
                # Resize to 512x512 for consistency
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                self.source_image = img
                self.display_image(img, self.source_canvas, "source")
                
                # Update UI
                filename = os.path.basename(filepath)
                self.source_path_var.set(filename)
                self.check_ready_to_generate()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load source image: {str(e)}")
                
    def select_target_image(self):
        """Select target image file"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Load and validate image
                img = Image.open(filepath)
                
                # Resize to 512x512 for consistency
                if img.size != (512, 512):
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                self.target_image = img
                self.display_image(img, self.target_canvas, "target")
                
                # Update UI
                filename = os.path.basename(filepath)
                self.target_path_var.set(filename)
                self.check_ready_to_generate()
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load target image: {str(e)}")
                
    def check_ready_to_generate(self):
        """Check if ready to generate and update button state"""
        if self.source_image and self.target_image:
            self.generate_btn.configure(state='normal')
            sections_per_side = 512 // self.section_size
            total_sections = sections_per_side * sections_per_side
            self.progress_var.set(f"Ready to match {total_sections} sections ({sections_per_side}×{sections_per_side} grid)")
        else:
            self.generate_btn.configure(state='disabled')
            missing = []
            if not self.source_image:
                missing.append("source image")
            if not self.target_image:
                missing.append("target image")
            self.progress_var.set(f"Please select: {', '.join(missing)}")
            
    def display_image(self, img, canvas, image_type):
        """Display image on canvas (scaled to fit)"""
        # Create 256x256 display version
        display_img = img.copy()
        display_img.thumbnail((256, 256), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(128, 128, image=photo)
        
        # Keep reference
        setattr(self, f"{image_type}_photo_ref", photo)
        
    def extract_sections(self, image, section_size):
        """Extract all sections from an image"""
        sections = []
        sections_per_side = image.size[0] // section_size
        img_array = np.array(image)
        
        for row in range(sections_per_side):
            for col in range(sections_per_side):
                y_start = row * section_size
                y_end = y_start + section_size
                x_start = col * section_size
                x_end = x_start + section_size
                
                section = img_array[y_start:y_end, x_start:x_end]
                sections.append(section)
                
        return sections, sections_per_side
        
    def calculate_mse(self, section1, section2):
        """Calculate Mean Squared Error between two sections (lower is better)"""
        return np.mean((section1.astype(np.float32) - section2.astype(np.float32)) ** 2)
        
    def calculate_ssim(self, section1, section2):
        """Calculate Structural Similarity Index (higher is better, convert to lower-is-better)"""
        # Simple SSIM approximation
        mu1 = np.mean(section1)
        mu2 = np.mean(section2)
        sigma1 = np.var(section1)
        sigma2 = np.var(section2)
        sigma12 = np.mean((section1 - mu1) * (section2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return 1.0 - ssim  # Convert to lower-is-better
        
    def calculate_ncc(self, section1, section2):
        """Calculate Normalized Cross-Correlation (higher is better, convert to lower-is-better)"""
        s1_norm = (section1 - np.mean(section1)) / (np.std(section1) + 1e-8)
        s2_norm = (section2 - np.mean(section2)) / (np.std(section2) + 1e-8)
        ncc = np.mean(s1_norm * s2_norm)
        return 1.0 - ncc  # Convert to lower-is-better
        
    def calculate_hist_corr(self, section1, section2):
        """Calculate histogram correlation (higher is better, convert to lower-is-better)"""
        # Convert to grayscale for histogram
        gray1 = np.mean(section1, axis=2).astype(np.uint8)
        gray2 = np.mean(section2, axis=2).astype(np.uint8)
        
        # Calculate histograms
        hist1 = np.histogram(gray1, bins=32, range=(0, 255))[0]
        hist2 = np.histogram(gray2, bins=32, range=(0, 255))[0]
        
        # Normalize histograms
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)
        
        # Calculate correlation coefficient
        corr = np.corrcoef(hist1, hist2)[0, 1]
        if np.isnan(corr):
            corr = 0
        return 1.0 - corr  # Convert to lower-is-better
        
    def find_best_matching_section(self, source_section, target_sections, similarity_func, used_sections):
        """Find the best matching section from available (unused) target sections"""
        best_match_idx = -1
        best_similarity = float('inf')
        
        for idx, target_section in enumerate(target_sections):
            # Skip if this section has already been used
            if idx in used_sections:
                continue
                
            similarity = similarity_func(source_section, target_section)
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_match_idx = idx
                
        return best_match_idx, best_similarity
        
    def calculate_section_similarity_score(self, result_section, source_section, similarity_func):
        """Calculate how well a result section matches its corresponding source section"""
        return similarity_func(result_section, source_section)
        
    def optimize_with_swaps(self, result_array, source_sections, sections_per_side, similarity_func, max_swaps=2000):
        """Optimize the result by checking random swaps between sections"""
        self.progress_var.set("Optimizing with random swaps...")
        self.root.update()
        
        total_sections = len(source_sections)
        improvements_made = 0
        swaps_checked = 0
        
        while swaps_checked < max_swaps:
            # Pick two random sections to potentially swap
            idx1 = random.randint(0, total_sections - 1)
            idx2 = random.randint(0, total_sections - 1)
            
            # Skip if same section
            if idx1 == idx2:
                continue
                
            swaps_checked += 1
            
            # Update progress every 100 swaps
            if swaps_checked % 100 == 0:
                progress_pct = int((swaps_checked / max_swaps) * 100)
                self.progress_var.set(f"Optimizing... {progress_pct}% ({improvements_made} improvements)")
                self.root.update()
            
            # Calculate positions in the grid
            row1, col1 = idx1 // sections_per_side, idx1 % sections_per_side
            row2, col2 = idx2 // sections_per_side, idx2 % sections_per_side
            
            # Extract current sections from result
            y1_start, y1_end = row1 * self.section_size, (row1 + 1) * self.section_size
            x1_start, x1_end = col1 * self.section_size, (col1 + 1) * self.section_size
            
            y2_start, y2_end = row2 * self.section_size, (row2 + 1) * self.section_size
            x2_start, x2_end = col2 * self.section_size, (col2 + 1) * self.section_size
            
            current_section1 = result_array[y1_start:y1_end, x1_start:x1_end]
            current_section2 = result_array[y2_start:y2_end, x2_start:x2_end]
            
            # Calculate current similarity scores
            current_score1 = self.calculate_section_similarity_score(
                current_section1, source_sections[idx1], similarity_func)
            current_score2 = self.calculate_section_similarity_score(
                current_section2, source_sections[idx2], similarity_func)
            current_total_score = current_score1 + current_score2
            
            # Calculate scores if we swap the sections
            swapped_score1 = self.calculate_section_similarity_score(
                current_section2, source_sections[idx1], similarity_func)  # section2 -> position1
            swapped_score2 = self.calculate_section_similarity_score(
                current_section1, source_sections[idx2], similarity_func)  # section1 -> position2
            swapped_total_score = swapped_score1 + swapped_score2
            
            # If swapping improves the total score, do the swap
            if swapped_total_score < current_total_score:  # Lower is better
                # Perform the swap
                temp_section = current_section1.copy()
                result_array[y1_start:y1_end, x1_start:x1_end] = current_section2
                result_array[y2_start:y2_end, x2_start:x2_end] = temp_section
                improvements_made += 1
                
        return improvements_made, swaps_checked
        
    def generate_matched_image(self):
        """Generate the matched image using section matching"""
        if not self.source_image or not self.target_image:
            return
            
        try:
            # Show progress
            self.progress_var.set("Extracting sections...")
            self.progress_bar.start()
            self.root.update()
            
            # Extract sections from both images
            source_sections, sections_per_side = self.extract_sections(self.source_image, self.section_size)
            target_sections, _ = self.extract_sections(self.target_image, self.section_size)
            
            self.progress_var.set(f"Matching {len(source_sections)} sections...")
            self.root.update()
            
            # Get similarity function
            similarity_func = self.similarity_methods[self.current_similarity_method]
            
            # Create result image
            result_array = np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Create random processing order for source sections
            total_sections = len(source_sections)
            section_indices = list(range(total_sections))
            random.shuffle(section_indices)  # Randomize processing order
            
            # Track which target sections have been used
            used_target_sections = set()
            matched_indices = [-1] * total_sections  # Track matches by original position
            
            # Process source sections in random order
            for processed_count, source_idx in enumerate(section_indices):
                # Update progress
                if processed_count % 10 == 0:
                    progress_pct = int((processed_count / total_sections) * 100)
                    self.progress_var.set(f"Matching sections... {progress_pct}%")
                    self.root.update()
                
                source_section = source_sections[source_idx]
                
                # Find best matching unused target section
                best_idx, similarity_score = self.find_best_matching_section(
                    source_section, target_sections, similarity_func, used_target_sections)
                
                # Handle case where no unused sections remain (shouldn't happen if same size)
                if best_idx == -1:
                    # Fallback: find best match from all sections (allow reuse for remaining)
                    best_idx, similarity_score = self.find_best_matching_section(
                        source_section, target_sections, similarity_func, set())
                
                # Mark this target section as used
                used_target_sections.add(best_idx)
                matched_indices[source_idx] = best_idx
                
                # Calculate position in result image (based on original source position)
                row = source_idx // sections_per_side
                col = source_idx % sections_per_side
                
                y_start = row * self.section_size
                y_end = y_start + self.section_size
                x_start = col * self.section_size
                x_end = x_start + self.section_size
                
                result_array[y_start:y_end, x_start:x_end] = target_sections[best_idx]
            
            # Perform swap optimization if enabled
            improvements_made = 0
            swaps_checked = 0
            
            if self.enable_optimization_var.get():
                self.progress_var.set("Starting swap optimization phase...")
                self.root.update()
                
                improvements_made, swaps_checked = self.optimize_with_swaps(
                    result_array, source_sections, sections_per_side, similarity_func, max_swaps=2000)
            
            # Create result image
            self.result_image = Image.fromarray(result_array)
            self.display_image(self.result_image, self.result_canvas, "result")
            
            # Calculate statistics
            valid_matches = [idx for idx in matched_indices if idx != -1]
            unique_matches = len(set(valid_matches))
            total_available = len(target_sections)
            reused_sections = len(valid_matches) - unique_matches
            
            # Update progress with results
            self.progress_bar.stop()
            base_stats = f"Used {unique_matches}/{total_available} unique sections"
            if reused_sections > 0:
                base_stats += f", {reused_sections} reused"
            
            if self.enable_optimization_var.get():
                optimization_stats = f"Optimization: {improvements_made} improvements from {swaps_checked} swaps"
                self.progress_var.set(f"Complete! {base_stats} | {optimization_stats}")
            else:
                self.progress_var.set(f"Complete! {base_stats} (optimization disabled)")
            
        except Exception as e:
            self.progress_bar.stop()
            self.progress_var.set("Error generating matched image")
            messagebox.showerror("Error", f"Could not generate matched image: {str(e)}")
            
    def save_result_image(self):
        """Save the result image"""
        if not self.result_image:
            messagebox.showwarning("No Result", "Please generate a matched image first")
            return
            
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"section_matched_{timestamp}.png"
        
        filepath = filedialog.asksaveasfilename(
            title="Save Matched Image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=filetypes
        )
        
        if filepath:
            try:
                self.result_image.save(filepath)
                messagebox.showinfo("Success", f"Matched image saved as: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")

def main():
    # Check if required packages are available
    missing_packages = []
    
    try:
        import numpy as np
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_packages.append("pillow")
    
    if missing_packages:
        error_msg = f"Missing required packages: {', '.join(missing_packages)}\n\n"
        error_msg += "Please install with:\n"
        error_msg += f"pip install {' '.join(missing_packages)}\n\n"
        error_msg += "Or use the launcher script which will install automatically."
        
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
    app = ImageSectionMatcher(root)
    root.mainloop()

if __name__ == "__main__":
    main()
