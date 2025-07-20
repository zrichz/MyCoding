import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from gaussian_clahe import gaussian_clahe_color
import os
import tempfile  # Add this import for temporary file handling

class CLAHEGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaussian CLAHE Image Enhancer")
        self.image = None
        self.output_image = None
        self.image_path = None
        self.max_size = tk.IntVar(value=480)  # Default max size for processing

        # Layout
        tk.Button(root, text="Open Image", command=self.load_image).pack()
        self.image_label = tk.Label(root)
        self.image_label.pack()
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(root, textvariable=self.progress_var)
        self.progress_label.pack()

        # Controls
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        tk.Label(control_frame, text="Tile Size:").grid(row=0, column=0)
        self.tile_entry = tk.Entry(control_frame, width=5)
        self.tile_entry.insert(0, "64")
        self.tile_entry.grid(row=0, column=1)

        tk.Label(control_frame, text="Kernel Size:").grid(row=0, column=2)
        self.kernel_entry = tk.Entry(control_frame, width=5)
        self.kernel_entry.insert(0, "21")
        self.kernel_entry.grid(row=0, column=3)

        tk.Label(control_frame, text="Stride:").grid(row=0, column=4)
        self.stride_entry = tk.Entry(control_frame, width=5)
        self.stride_entry.insert(0, "16")
        self.stride_entry.grid(row=0, column=5)

        # Max size selection
        size_frame = tk.Frame(root)
        size_frame.pack(pady=5)
        tk.Label(size_frame, text="Max Image Size:").pack(side=tk.LEFT)
        for size in [360, 512, 1024, 2048]:
            tk.Radiobutton(size_frame, text=str(size), variable=self.max_size, value=size).pack(side=tk.LEFT)

        tk.Button(root, text="Apply Enhancement", command=self.apply_clahe).pack(pady=3)
        tk.Button(root, text="Save Result", command=self.save_image).pack()
        tk.Button(root, text="Batch Process Directory", command=self.batch_process_directory).pack(pady=3)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        self.image_path = path
        self.image = Image.open(path).convert('RGB')

        # Resize image for processing based on selected max size
        max_dim = self.max_size.get()
        self.image.thumbnail((max_dim, max_dim))

        self.display_image(self.image)

    def display_image(self, img):
        preview = img.copy()
        preview.thumbnail((480, 480))
        tk_image = ImageTk.PhotoImage(preview)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

    def apply_clahe(self):
        if not self.image:
            messagebox.showerror("Error", "No image loaded.")
            return

        try:
            tile_size = int(self.tile_entry.get())
            kernel_size = int(self.kernel_entry.get())
            stride = int(self.stride_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Tile, stride, and kernel size must be integers.")
            return

        def gui_progress(percent):
            self.progress_var.set(f"Processing... {percent}%")
            self.root.update_idletasks()

        # Save the resized image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
            self.image.save(temp_path)

        # Pass the temporary file path to the processing function
        result = gaussian_clahe_color(
            temp_path,
            tile_size,
            kernel_size,
            stride,
            progress_callback=gui_progress
        )
        self.progress_var.set("Done.")
        self.output_image = result
        self.display_image(result)

    def save_image(self):
        if not self.output_image:
            messagebox.showinfo("Nothing to save", "Please apply enhancement first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.output_image.save(path)
            messagebox.showinfo("Saved", f"Image saved to {path}")

    def batch_process_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return

        output_directory = filedialog.askdirectory(title="Select Output Directory")
        if not output_directory:
            return

        try:
            tile_size = int(self.tile_entry.get())
            kernel_size = int(self.kernel_entry.get())
            stride = int(self.stride_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Tile, stride, and kernel size must be integers.")
            return

        def gui_progress(percent):
            self.progress_var.set(f"Processing... {percent}%")
            self.root.update_idletasks()

        for fname in os.listdir(directory):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(directory, fname)
                output_path = os.path.join(output_directory, f"enhanced_{fname}")

                # Save the resized image to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                    img = Image.open(input_path).convert('RGB')

                    # Resize image for processing based on selected max size
                    max_dim = self.max_size.get()
                    img.thumbnail((max_dim, max_dim))
                    img.save(temp_path)

                # Pass the temporary file path to the processing function
                result = gaussian_clahe_color(
                    temp_path,
                    tile_size,
                    kernel_size,
                    stride,
                    progress_callback=gui_progress
                )
                result.save(output_path)
                self.progress_var.set(f"Saved: {output_path}")

        messagebox.showinfo("Batch Processing Complete", f"Processed files saved to {output_directory}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CLAHEGUI(root)
    root.mainloop()
