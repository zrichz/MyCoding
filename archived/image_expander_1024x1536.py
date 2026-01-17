"""
Image Auto-Expander - 1024x1024 -> 1536x1024

This script is adapted from image_expander_720x1600.py to expand
1024x1024 images horizontally to 1536x1024 using the same algorithmic
approach (edge-sourced blur + luminance fade). It provides a minimal GUI
for selecting a directory and batch processing images.

Usage:
        - Run interactively (GUI):
                python image_processors/image_expander_1024x1536.py

    - The GUI allows selecting input/output folders and will save expanded
        images into an `expanded_1536x1024` folder by default.

Notes:
    - Input images are expected to be 1024x1024 (if larger/smaller, they will be
        resized/center-cropped before expansion)
    - Target output size: 1536 (width) x 1024 (height)
    - Expansion is horizontal only (left/right padding) to reach 1536 width
    - Uses precomputed Gaussian kernels and vectorized NumPy operations for speed
"""

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy import ndimage
import os
from datetime import datetime


class ImageExpander1024:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Auto-Expander - Batch Processor (1024x1536)")
        self.root.geometry("900x520")

        self.input_directory = None

        # Target dimensions (width x height)
        self.target_width = 1536
        self.target_height = 1024

        # Assumed input size (will resize/crop as needed)
        self.input_size = 1024

        # Fixed settings
        self.blur_amount = 120  # smaller than original since expansion is only horizontal
        self.luminance_drop = 25

        # Kernel cache
        self.kernel_cache = {}
        self._initialize_kernel_cache()

        self.setup_gui()

    def _initialize_kernel_cache(self):
        max_blur = self.blur_amount
        blur_steps = np.arange(0, max_blur + 0.5, 0.5)
        for blur_amount in blur_steps:
            if blur_amount > 0:
                kernel_size = int(blur_amount * 2) * 2 + 1
                sigma = blur_amount / 3.0
                x = np.arange(kernel_size) - kernel_size // 2
                kernel = np.exp(-x**2 / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                self.kernel_cache[round(blur_amount, 2)] = {
                    'kernel': kernel.astype(np.float32),
                    'kernel_size': kernel_size,
                    'sigma': sigma
                }

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        title = ttk.Label(main_frame, text="Image Auto-Expander (1024->1536)", font=("Arial", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 12))

        ttk.Label(main_frame, text="Input Directory:").grid(row=1, column=0, sticky="w")
        self.input_var = tk.StringVar(value="No directory selected")
        self.input_label = ttk.Label(main_frame, textvariable=self.input_var, relief="sunken")
        self.input_label.grid(row=1, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(main_frame, text="Browse...", command=self.select_directory).grid(row=1, column=2)

        ttk.Label(main_frame, text="Output Directory (optional):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.output_var = tk.StringVar(value="Auto: <input_dir>/expanded_1536x1024")
        self.output_label = ttk.Label(main_frame, textvariable=self.output_var, relief="sunken")
        self.output_label.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(main_frame, text="Browse...", command=self.select_output_directory).grid(row=2, column=2)

        self.process_button = ttk.Button(main_frame, text="Create Expansions", command=self.process_all_images, state="disabled")
        self.process_button.grid(row=3, column=0, columnspan=3, pady=(12, 12))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky="ew")

        self.progress_label = ttk.Label(main_frame, text="No directory selected")
        self.progress_label.grid(row=5, column=0, columnspan=3, pady=(6, 0))

        self.status_text = tk.Text(main_frame, height=10, wrap=tk.WORD, font=("Consolas", 9))
        self.status_text.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        main_frame.rowconfigure(6, weight=1)

        self.log("Ready. Select a directory containing 1024x1024 images (or images that can be resized).")

    def log(self, msg):
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)
        self.root.update()

    def select_directory(self):
        d = filedialog.askdirectory(title="Select input directory containing images")
        if not d:
            return
        self.input_directory = d
        self.input_var.set(d)
        files = self.get_image_files(d)
        self.log(f"Selected {d} — {len(files)} images found")
        if len(files) > 0:
            self.process_button.config(state="normal")

    def select_output_directory(self):
        d = filedialog.askdirectory(title="Select output directory")
        if not d:
            return
        self.output_directory = d
        self.output_var.set(d)

    def get_image_files(self, directory):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in exts]
        return sorted(files)

    def process_all_images(self):
        if not self.input_directory:
            return
        files = self.get_image_files(self.input_directory)
        total = len(files)
        if total == 0:
            messagebox.showwarning("No images", "No supported images found")
            return

        out_dir_name = "expanded_1536x1024"
        if hasattr(self, 'output_directory') and self.output_directory:
            out_dir = self.output_directory
        else:
            out_dir = os.path.join(self.input_directory, out_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        success = 0
        fail = 0

        for i, fname in enumerate(files):
            try:
                self.progress_var.set((i/total)*100)
                self.progress_label.config(text=f"Processing {i+1}/{total}: {fname}")
                self.root.update()

                input_path = os.path.join(self.input_directory, fname)
                base, ext = os.path.splitext(fname)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                out_name = f"{timestamp}_{i:03d}_1536x1024{ext}"
                out_path = os.path.join(out_dir, out_name)

                if self.process_single_image(input_path, out_path):
                    success += 1
                    self.log(f"Processed: {fname} -> {out_name}")
                else:
                    fail += 1
                    self.log(f"Failed: {fname}")

            except Exception as e:
                fail += 1
                self.log(f"Error {fname}: {e}")

        self.progress_var.set(100)
        self.progress_label.config(text=f"Complete: {success} successful, {fail} failed")
        messagebox.showinfo("Done", f"Processed {success} images, {fail} failed. Output: {out_dir}")

    def process_single_image(self, input_path, output_path):
        try:
            img = Image.open(input_path)
            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Center-crop or resize to input_size (1024x1024)
            if img.width != self.input_size or img.height != self.input_size:
                # If larger, center-crop first
                if img.width > self.input_size and img.height >= self.input_size:
                    left = (img.width - self.input_size)//2
                    img = img.crop((left, 0, left + self.input_size, self.input_size))
                elif img.height > self.input_size and img.width >= self.input_size:
                    top = (img.height - self.input_size)//2
                    img = img.crop((0, top, self.input_size, top + self.input_size))
                else:
                    img = img.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)

            arr = np.array(img)
            h, w = arr.shape[:2]

            # Calculate horizontal expansion needed
            width_expansion = max(0, self.target_width - w)
            left_pad = width_expansion // 2
            right_pad = width_expansion - left_pad

            # Create expanded canvas
            expanded = np.zeros((self.target_height, self.target_width, 3), dtype=arr.dtype)

            # Vertical center placement
            y_start = (self.target_height - h)//2
            y_end = y_start + h
            x_start = left_pad
            x_end = x_start + w
            expanded[y_start:y_end, x_start:x_end] = arr

            # Process left expansion (if any)
            if left_pad > 0:
                self._process_left_expansion(expanded, arr, left_pad, x_start)

            # Process right expansion (if any)
            if right_pad > 0:
                self._process_right_expansion(expanded, arr, right_pad, x_end)

            out_img = Image.fromarray(expanded.astype('uint8'))
            out_img.save(output_path, optimize=True)
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    def _process_left_expansion(self, expanded, src_arr, left_pad, x_start):
        # Source column is first column of the src_arr
        src_col = src_arr[:, 0]
        # For each pad column, compute blur and luminance and assign
        for i in range(left_pad):
            progress = (left_pad - i) / left_pad
            blur_amt = progress * self.blur_amount
            lum = progress * self.luminance_drop

            # Apply vertical blur along column (treat column as height x channels)
            col = self.apply_vertical_blur(src_col, blur_amt)
            col = self.apply_luminance_reduction(col, lum)
            expanded[:, i] = col

    def _process_right_expansion(self, expanded, src_arr, right_pad, x_end):
        src_col = src_arr[:, -1]
        for i in range(right_pad):
            progress = (i + 1) / right_pad
            blur_amt = progress * self.blur_amount
            lum = progress * self.luminance_drop

            col = self.apply_vertical_blur(src_col, blur_amt)
            col = self.apply_luminance_reduction(col, lum)
            expanded[:, x_end + i] = col

    def apply_vertical_blur(self, column_array, blur_amount):
        if blur_amount <= 0:
            return column_array
        blur_key = round(blur_amount * 2) / 2.0
        if blur_key not in self.kernel_cache:
            keys = list(self.kernel_cache.keys())
            blur_key = min(keys, key=lambda x: abs(x - blur_amount))
        kernel = self.kernel_cache[blur_key]['kernel']
        # column_array can be (height,3) or (height,)
        if column_array.ndim == 2:
            out = np.zeros_like(column_array)
            for c in range(column_array.shape[1]):
                out[:, c] = ndimage.convolve1d(column_array[:, c].astype(np.float32), kernel, mode='nearest')
            return np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = ndimage.convolve1d(column_array.astype(np.float32), kernel, mode='nearest')
            return np.clip(out, 0, 255).astype(np.uint8)

    def apply_luminance_reduction(self, line_array, luminance_factor):
        if luminance_factor <= 0:
            return line_array
        multiplier = 1.0 - (luminance_factor / 100.0)
        out = (line_array.astype(np.float32) * multiplier)
        return np.clip(out, 0, 255).astype(np.uint8)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ImageExpander1024()
    app.run()
