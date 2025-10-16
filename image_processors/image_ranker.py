"""
Image Ranker - Compare all image pairs to build comprehensive rankings
Uses tournament-style pairwise comparisons to rank all images in a directory
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
from pathlib import Path
from itertools import combinations
import json

class ImageRanker:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Ranker - Pairwise Comparison")
        self.root.geometry("2560x1440")
        self.root.configure(bg='#808080')
        
        # Variables
        self.input_directory = None
        self.image_files = []
        self.current_pair_index = 0
        self.all_pairs = []
        self.comparison_results = {}  # Store win/loss records
        self.rankings = {}  # Final rankings
        
        # Current comparison images
        self.left_image = None
        self.right_image = None
        self.left_photo_ref = None
        self.right_photo_ref = None
        
        # Current pair being displayed
        self.current_left_path = None
        self.current_right_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with left sidebar and right comparison area
        main_container = tk.Frame(self.root, bg='#808080')
        main_container.pack(fill='both', expand=True)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Ranking.TFrame', background='#808080')
        style.configure('Ranking.TLabel', background='#808080', foreground='black')
        style.configure('Ranking.TButton', background='#606060', foreground='white')
        
        # Left sidebar for controls (400px wide)
        left_sidebar = tk.Frame(main_container, bg='#808080', width=400)
        left_sidebar.pack(side=tk.LEFT, fill='y', padx=10, pady=10)
        left_sidebar.pack_propagate(False)
        
        # Directory selection
        dir_frame = ttk.LabelFrame(left_sidebar, text="Image Directory", padding=10)
        dir_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Button(dir_frame, text="📁 Select Directory", 
                  command=self.select_directory).pack(pady=5)
        
        self.dir_label = ttk.Label(dir_frame, text="No directory selected", 
                                  foreground="gray", wraplength=350)
        self.dir_label.pack(pady=5)
        
        # Progress and instructions
        progress_frame = ttk.LabelFrame(left_sidebar, text="Ranking Progress", padding=10)
        progress_frame.pack(fill='x', pady=(0, 15))
        
        self.progress_var = tk.StringVar(value="Select a directory to begin ranking")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var, 
                                  font=("Arial", 11), wraplength=350)
        progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=350, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Instructions
        instruction_frame = ttk.LabelFrame(left_sidebar, text="Instructions", padding=10)
        instruction_frame.pack(fill='x', pady=(0, 15))
        
        instruction_text = "Click on the BETTER image of each pair.\n\nAll images will be compared against all others to build complete rankings.\n\nFiles will be automatically renamed when complete."
        instruction_label = ttk.Label(instruction_frame, text=instruction_text, 
                                    font=("Arial", 10), foreground="blue", 
                                    wraplength=350, justify='left')
        instruction_label.pack(pady=5)
        
        # Current comparison info
        comparison_info_frame = ttk.LabelFrame(left_sidebar, text="Current Comparison", padding=10)
        comparison_info_frame.pack(fill='x', pady=(0, 15))
        
        self.comparison_info_var = tk.StringVar(value="No comparison active")
        comparison_info_label = ttk.Label(comparison_info_frame, textvariable=self.comparison_info_var,
                                        font=("Arial", 10), wraplength=350, justify='left')
        comparison_info_label.pack(pady=5)
        
        # Right area for image comparison (rest of screen width)
        comparison_frame = tk.Frame(main_container, bg='#808080')
        comparison_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=(0, 10), pady=10)
        
        # Title for comparison area
        comparison_title = tk.Label(comparison_frame, text="Image Comparison - Click the Better Image", 
                                   bg='#808080', fg='black', font=("Arial", 16, "bold"))
        comparison_title.pack(pady=(0, 20))
        
        # Container for both images
        images_container = tk.Frame(comparison_frame, bg='#808080')
        images_container.pack(expand=True, fill='both')
        
        # Left image frame - maximize space (using ~1000px width, 1300px height)
        left_frame = tk.Frame(images_container, bg='#808080')
        left_frame.pack(side=tk.LEFT, padx=20, expand=True, fill='both')
        
        self.left_label = tk.Label(left_frame, text="Image A", 
                                  bg='#808080', fg='black', font=("Arial", 14, "bold"))
        self.left_label.pack(pady=(0, 10))
        
        self.left_canvas = tk.Canvas(left_frame, width=1000, height=1300, 
                                    bg='#1a1a1a', highlightthickness=3, 
                                    highlightbackground='#00FF00', cursor='hand2')
        self.left_canvas.pack(expand=True)
        self.left_canvas.bind("<Button-1>", self.choose_left)
        
        # Right image frame - maximize space (using ~1000px width, 1300px height)
        right_frame = tk.Frame(images_container, bg='#808080')
        right_frame.pack(side=tk.LEFT, padx=20, expand=True, fill='both')
        
        self.right_label = tk.Label(right_frame, text="Image B", 
                                   bg='#808080', fg='black', font=("Arial", 14, "bold"))
        self.right_label.pack(pady=(0, 10))
        
        self.right_canvas = tk.Canvas(right_frame, width=1000, height=1300, 
                                     bg='#1a1a1a', highlightthickness=3, 
                                     highlightbackground='#00FF00', cursor='hand2')
        self.right_canvas.pack(expand=True)
        self.right_canvas.bind("<Button-1>", self.choose_right)
        
    def select_directory(self):
        """Select directory containing images"""
        directory = filedialog.askdirectory(title="Select Image Directory for Ranking")
        if directory:
            self.input_directory = directory
            self.scan_images()
            
    def scan_images(self):
        """Scan directory for valid image files"""
        if not self.input_directory:
            return
            
        # Supported image formats
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        self.image_files = []
        for file_path in Path(self.input_directory).iterdir():
            if file_path.suffix.lower() in extensions:
                self.image_files.append(file_path)
        
        self.image_files.sort()  # Sort alphabetically
        
        if len(self.image_files) < 2:
            messagebox.showwarning("Insufficient Images", 
                                 "Need at least 2 images for ranking comparison")
            return
            
        # Generate all possible pairs
        self.all_pairs = list(combinations(self.image_files, 2))
        
        # Initialize comparison results
        self.comparison_results = {str(img_path): {'wins': 0, 'losses': 0} 
                                 for img_path in self.image_files}
        
        self.dir_label.config(text=f"{len(self.image_files)} images found - "
                                  f"{len(self.all_pairs)} comparisons needed")
        
        self.progress_var.set(f"Ready to rank {len(self.image_files)} images "
                             f"({len(self.all_pairs)} comparisons)")
        
        self.current_pair_index = 0
        self.progress_bar.configure(maximum=len(self.all_pairs), value=0)
        
        # Start first comparison
        self.show_next_pair()
        
    def show_next_pair(self):
        """Display the next pair of images for comparison"""
        if self.current_pair_index >= len(self.all_pairs):
            self.complete_ranking()
            return
            
        # Get current pair
        left_path, right_path = self.all_pairs[self.current_pair_index]
        self.current_left_path = left_path
        self.current_right_path = right_path
        
        # Update progress
        self.progress_var.set(f"Comparison {self.current_pair_index + 1} of {len(self.all_pairs)}")
        self.progress_bar.configure(value=self.current_pair_index)
        
        # Update comparison info in sidebar
        self.comparison_info_var.set(f"Left: {left_path.name}\n\nRight: {right_path.name}\n\nClick the better image!")
        
        # Update labels with filenames
        self.left_label.config(text=f"A: {left_path.name}")
        self.right_label.config(text=f"B: {right_path.name}")
        
        try:
            # Load and display left image
            left_img = Image.open(left_path)
            left_display = self.resize_for_display(left_img, 1000, 1300)
            self.left_image = ImageTk.PhotoImage(left_display)
            
            self.left_canvas.delete("all")
            self.left_canvas.create_image(500, 650, image=self.left_image)
            self.left_photo_ref = self.left_image
            
            # Load and display right image
            right_img = Image.open(right_path)
            right_display = self.resize_for_display(right_img, 1000, 1300)
            self.right_image = ImageTk.PhotoImage(right_display)
            
            self.right_canvas.delete("all")
            self.right_canvas.create_image(500, 650, image=self.right_image)
            self.right_photo_ref = self.right_image
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load images: {str(e)}")
            self.next_pair()
            
    def resize_for_display(self, image, max_width, max_height):
        """Resize image to fit display area while maintaining aspect ratio"""
        width, height = image.size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def choose_left(self, event=None):
        """User chose left image as better"""
        self.record_choice(self.current_left_path, self.current_right_path)
        
    def choose_right(self, event=None):
        """User chose right image as better"""
        self.record_choice(self.current_right_path, self.current_left_path)
        
    def record_choice(self, winner_path, loser_path):
        """Record the comparison result and move to next pair"""
        # Update win/loss records
        self.comparison_results[str(winner_path)]['wins'] += 1
        self.comparison_results[str(loser_path)]['losses'] += 1
        
        # Move to next pair
        self.next_pair()
        
    def next_pair(self):
        """Move to the next comparison pair"""
        self.current_pair_index += 1
        self.show_next_pair()
        
    def complete_ranking(self):
        """Complete the ranking process and calculate final rankings"""
        self.progress_var.set("Calculating final rankings...")
        self.progress_bar.configure(value=len(self.all_pairs))
        
        # Calculate ranking scores (wins - losses)
        scores = {}
        for img_path_str, record in self.comparison_results.items():
            scores[img_path_str] = record['wins'] - record['losses']
        
        # Sort by score (highest first)
        sorted_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign rankings (handle ties)
        current_rank = 1
        self.rankings = {}
        
        for i, (img_path_str, score) in enumerate(sorted_images):
            if i > 0 and score != sorted_images[i-1][1]:
                # Score changed, update rank
                current_rank = i + 1
            
            # Count how many images have this same score
            same_score_count = sum(1 for _, s in sorted_images if s == score)
            
            if same_score_count > 1:
                # Tied ranking
                self.rankings[img_path_str] = {
                    'rank': current_rank,
                    'score': score,
                    'tied': True,
                    'tie_count': same_score_count
                }
            else:
                # Unique ranking
                self.rankings[img_path_str] = {
                    'rank': current_rank,
                    'score': score,
                    'tied': False,
                    'tie_count': 1
                }
        
        # Clear comparison display
        self.left_canvas.delete("all")
        self.right_canvas.delete("all")
        self.left_label.config(text="Ranking Complete!")
        self.right_label.config(text="Ready to apply prefixes")
        
        self.progress_var.set(f"Ranking complete! {len(self.image_files)} images ranked.")
        
        # Show add ranks dialog
        self.show_add_ranks_dialog()
        
    def show_add_ranks_dialog(self):
        """Show simple dialog to add rank prefixes to filenames"""
        result = messagebox.askyesno(
            "Add Ranking Prefixes", 
            f"Ready to add ranking prefixes to {len(self.image_files)} image filenames.\n\n"
            "This will rename files with rank prefixes like:\n"
            "• 001_rank_image.jpg (1st place)\n"
            "• 003e_rank_image.jpg (tied for 3rd)\n\n"
            "Do you want to proceed?"
        )
        
        if result:
            # Start renaming in background thread
            import threading
            rename_thread = threading.Thread(target=self.apply_ranking_prefixes)
            rename_thread.daemon = True
            rename_thread.start()
        else:
            # User cancelled - show completion message
            self.comparison_info_var.set("Ranking complete!\nFile renaming cancelled by user.")
        
    def restart_ranking(self):
        """Restart the ranking process"""
        if self.image_files:
            result = messagebox.askyesno("Restart Ranking", 
                                       "Are you sure you want to restart the ranking process?\n"
                                       "All current progress will be lost.")
            if result:
                self.current_pair_index = 0
                self.comparison_results = {str(img_path): {'wins': 0, 'losses': 0} 
                                         for img_path in self.image_files}
                self.rankings = {}
                self.show_next_pair()
                
    def save_rankings(self):
        """Save ranking results to JSON file"""
        if not self.rankings:
            messagebox.showwarning("No Rankings", "Complete the ranking process first")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Rankings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.input_directory
        )
        
        if save_path:
            try:
                save_data = {
                    'directory': str(self.input_directory),
                    'total_images': len(self.image_files),
                    'total_comparisons': len(self.all_pairs),
                    'rankings': self.rankings,
                    'comparison_results': self.comparison_results
                }
                
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                    
                messagebox.showinfo("Rankings Saved", f"Rankings saved to:\n{save_path}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save rankings: {str(e)}")
                
    def apply_ranking_prefixes(self):
        """Apply ranking prefixes to all filenames"""
        if not self.rankings:
            self.root.after(0, lambda: messagebox.showwarning("No Rankings", "Complete the ranking process first"))
            return
            
        # Start renaming in separate thread (confirmation already done in dialog)
        thread = threading.Thread(target=self._apply_prefixes_thread)
        thread.daemon = True
        thread.start()
        
    def _apply_prefixes_thread(self):
        """Apply prefixes in separate thread"""
        try:
            renamed_count = 0
            errors = []
            
            for img_path_str, ranking in self.rankings.items():
                img_path = None
                try:
                    img_path = Path(img_path_str)
                    
                    # Generate prefix
                    rank = ranking['rank']
                    if ranking['tied']:
                        prefix = f"{rank:03d}e_rank_"
                    else:
                        prefix = f"{rank:03d}_rank_"
                    
                    # Create new filename
                    new_name = f"{prefix}{img_path.name}"
                    new_path = img_path.parent / new_name
                    
                    # Rename file
                    img_path.rename(new_path)
                    renamed_count += 1
                    
                    # Update progress
                    self.root.after(0, lambda: self.progress_var.set(
                        f"Renaming files... {renamed_count}/{len(self.image_files)}"))
                    
                except Exception as e:
                    filename = img_path.name if img_path else Path(img_path_str).name
                    errors.append(f"{filename}: {str(e)}")
                    
            # Update UI on completion
            if errors:
                error_msg = f"Renamed {renamed_count} files with {len(errors)} errors:\n\n"
                error_msg += "\n".join(errors[:10])  # Show first 10 errors
                if len(errors) > 10:
                    error_msg += f"\n... and {len(errors) - 10} more errors"
                    
                self.root.after(0, lambda: messagebox.showwarning("Partial Success", error_msg))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                              f"Successfully renamed {renamed_count} files!"))
                
            self.root.after(0, lambda: self.progress_var.set(
                f"Renaming complete! {renamed_count} files processed."))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Renaming Error", 
                                                           f"An error occurred: {str(e)}"))

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
        error_msg += "Or activate the virtual environment first."
        
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
    app = ImageRanker(root)
    root.mainloop()

if __name__ == "__main__":
    main()