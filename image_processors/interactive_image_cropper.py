#!/usr/bin/env python3
"""
Interactive Image Cropper
A GUI application for batch cropping images with interactive selection.
User selects crop area on thumbnail, crops are applied to full resolution originals.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from PIL import Image, ImageTk
import math


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Image Cropper")
        self.root.geometry("1000x900")
        
        # Application state
        self.image_directory = None
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.current_thumbnail = None
        self.scale_factor = 1.0
        
        # Crop selection variables
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_end_x = None
        self.crop_end_y = None
        self.crop_rectangle = None
        self.is_selecting = False
        self.image_x_offset = 0
        self.image_y_offset = 0
        self.photo = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Directory selection frame
        dir_frame = ttk.Frame(main_frame)
        dir_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)
        
        ttk.Label(dir_frame, text="Image Directory:").grid(row=0, column=0, sticky="w")
        self.dir_label = ttk.Label(dir_frame, text="No directory selected", foreground="gray")
        self.dir_label.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Button(dir_frame, text="Browse", command=self.select_directory).grid(row=0, column=2, padx=(10, 0))
        
        # Image display frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview (Click and drag to select crop area)", padding="5")
        self.image_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        
        # Canvas for image display and crop selection
        self.canvas = tk.Canvas(self.image_frame, bg="white", width=800, height=600)
        self.canvas.pack(expand=True, fill="both")
        
        # Bind mouse events for crop selection
        self.canvas.bind("<Button-1>", self.start_crop_selection)
        self.canvas.bind("<B1-Motion>", self.update_crop_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop_selection)
        
        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        control_frame.columnconfigure(1, weight=1)
        
        # Progress info
        self.progress_label = ttk.Label(control_frame, text="No images loaded")
        self.progress_label.grid(row=0, column=0, sticky="w")
        
        # Current image info
        self.image_info_label = ttk.Label(control_frame, text="")
        self.image_info_label.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        self.prev_button = ttk.Button(button_frame, text="Previous", command=self.previous_image, state="disabled")
        self.prev_button.pack(side="left", padx=(0, 5))
        
        self.crop_button = ttk.Button(button_frame, text="Crop & Save", command=self.crop_and_save, state="disabled")
        self.crop_button.pack(side="left", padx=(0, 5))
        
        self.skip_button = ttk.Button(button_frame, text="Skip", command=self.next_image, state="disabled")
        self.skip_button.pack(side="left", padx=(0, 5))
        
        self.next_button = ttk.Button(button_frame, text="Next", command=self.next_image, state="disabled")
        self.next_button.pack(side="left", padx=(0, 5))
        
        # Clear selection button
        self.clear_button = ttk.Button(button_frame, text="Clear Selection", command=self.clear_crop_selection, state="disabled")
        self.clear_button.pack(side="left", padx=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Select a directory to begin")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))
    
    def select_directory(self):
        """Select directory containing images"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        
        if directory:
            self.image_directory = Path(directory)
            self.load_image_files()
            self.update_directory_display()
            
    def load_image_files(self):
        """Load list of image files from selected directory"""
        if not self.image_directory:
            return
            
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        
        self.image_files = []
        for file_path in self.image_directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                self.image_files.append(file_path)
        
        self.image_files.sort()  # Sort alphabetically
        self.current_index = 0
        
        if self.image_files:
            self.load_current_image()
            self.update_controls()
            self.status_var.set(f"Loaded {len(self.image_files)} images")
        else:
            messagebox.showwarning("No Images", "No supported image files found in the selected directory.")
            self.status_var.set("No images found in directory")
    
    def update_directory_display(self):
        """Update the directory label"""
        if self.image_directory:
            dir_name = self.image_directory.name
            if len(str(self.image_directory)) > 60:
                self.dir_label.config(text=f".../{dir_name}", foreground="black")
            else:
                self.dir_label.config(text=str(self.image_directory), foreground="black")
    
    def load_current_image(self):
        """Load and display the current image"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
            
        try:
            image_path = self.image_files[self.current_index]
            
            # Load original image
            self.current_image = Image.open(image_path)
            
            # Create thumbnail for display (max 800x800)
            self.current_thumbnail = self.current_image.copy()
            self.current_thumbnail.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            # Calculate scale factor for mapping thumbnail coordinates to original
            self.scale_factor = min(
                self.current_image.width / self.current_thumbnail.width,
                self.current_image.height / self.current_thumbnail.height
            )
            
            # Display thumbnail
            self.display_image()
            self.update_image_info()
            self.clear_crop_selection()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.next_image()
    
    def display_image(self):
        """Display the thumbnail image on canvas"""
        if not self.current_thumbnail:
            return
            
        # Clear canvas
        self.canvas.delete("all")
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(self.current_thumbnail)
        
        # Place image immediately
        self._place_image_on_canvas()
    
    def _place_image_on_canvas(self):
        """Helper method to place image on canvas"""
        if not self.current_thumbnail or not self.photo:
            return
            
        # Ensure canvas is updated
        self.canvas.update_idletasks()
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas size is not available, use default and schedule retry
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
            # Schedule retry after canvas is properly sized
            self.root.after(100, self._place_image_on_canvas)
            return
        
        # Center image on canvas
        x = max(0, (canvas_width - self.current_thumbnail.width) // 2)
        y = max(0, (canvas_height - self.current_thumbnail.height) // 2)
        
        self.image_x_offset = x
        self.image_y_offset = y
        
        # Create the image on canvas
        image_id = self.canvas.create_image(x, y, anchor="nw", image=self.photo, tags="image")
        
        # Force canvas update
        self.canvas.update_idletasks()
    
    def on_canvas_resize(self, event):
        """Handle canvas resize to reposition image"""
        if self.current_thumbnail and self.photo:
            # Only reposition if the canvas size actually changed significantly
            if abs(event.width - 800) > 10 or abs(event.height - 600) > 10:
                canvas_width = event.width
                canvas_height = event.height
                
                x = max(0, (canvas_width - self.current_thumbnail.width) // 2)
                y = max(0, (canvas_height - self.current_thumbnail.height) // 2)
                
                self.image_x_offset = x
                self.image_y_offset = y
                
                # Update image position
                image_items = self.canvas.find_withtag("image")
                if image_items:
                    self.canvas.coords(image_items[0], x, y)
    
    def update_image_info(self):
        """Update image information display"""
        if not self.current_image or not self.image_files:
            return
            
        current_file = self.image_files[self.current_index]
        filename = current_file.name
        dimensions = f"{self.current_image.width}×{self.current_image.height}"
        
        self.image_info_label.config(text=f"{filename} ({dimensions})")
        self.progress_label.config(text=f"Image {self.current_index + 1} of {len(self.image_files)}")
    
    def start_crop_selection(self, event):
        """Start crop selection"""
        if not self.current_thumbnail:
            return
            
        # Get click coordinates
        x, y = event.x, event.y
        
        # Check if click is within image bounds
        if (self.image_x_offset <= x <= self.image_x_offset + self.current_thumbnail.width and
            self.image_y_offset <= y <= self.image_y_offset + self.current_thumbnail.height):
            
            self.is_selecting = True
            self.crop_start_x = x
            self.crop_start_y = y
            
            # Clear any existing crop selection
            if self.crop_rectangle:
                self.canvas.delete(self.crop_rectangle)
                self.crop_rectangle = None
    
    def update_crop_selection(self, event):
        """Update crop selection rectangle with 9:20 minimum aspect ratio constraint"""
        if not self.is_selecting or not self.current_thumbnail:
            return
            
        # Get current mouse position
        x, y = event.x, event.y
        
        # Constrain to image bounds
        x = max(self.image_x_offset, min(x, self.image_x_offset + self.current_thumbnail.width))
        y = max(self.image_y_offset, min(y, self.image_y_offset + self.current_thumbnail.height))
        
        # Ensure we have valid start coordinates
        if self.crop_start_x is None or self.crop_start_y is None:
            return
        
        # Calculate raw dimensions
        raw_width = abs(x - self.crop_start_x)
        raw_height = abs(y - self.crop_start_y)
        
        # Apply 9:20 minimum aspect ratio constraint (width:height = 9:20, so min_width = height * 9/20)
        min_aspect_ratio = 9.0 / 20.0  # 0.45
        
        if raw_height > 0:
            min_width_for_height = raw_height * min_aspect_ratio
            if raw_width < min_width_for_height:
                # Adjust width to maintain minimum aspect ratio
                raw_width = min_width_for_height
        
        # Determine final coordinates based on drag direction
        if x >= self.crop_start_x:  # Dragging right
            final_x = self.crop_start_x + raw_width
        else:  # Dragging left
            final_x = self.crop_start_x - raw_width
            
        if y >= self.crop_start_y:  # Dragging down
            final_y = self.crop_start_y + raw_height
        else:  # Dragging up
            final_y = self.crop_start_y - raw_height
        
        # Ensure final coordinates stay within image bounds
        final_x = max(self.image_x_offset, min(final_x, self.image_x_offset + self.current_thumbnail.width))
        final_y = max(self.image_y_offset, min(final_y, self.image_y_offset + self.current_thumbnail.height))
        
        # Recalculate actual width/height after bounds constraint
        actual_width = abs(final_x - self.crop_start_x)
        actual_height = abs(final_y - self.crop_start_y)
        
        # If bounds constraint broke our aspect ratio, adjust the other dimension
        if actual_height > 0:
            required_width = actual_height * min_aspect_ratio
            if actual_width < required_width:
                # Width was constrained, so adjust height to maintain aspect ratio
                new_height = actual_width / min_aspect_ratio
                if final_y >= self.crop_start_y:  # Dragging down
                    final_y = self.crop_start_y + new_height
                else:  # Dragging up
                    final_y = self.crop_start_y - new_height
                # Ensure adjusted height stays in bounds
                final_y = max(self.image_y_offset, min(final_y, self.image_y_offset + self.current_thumbnail.height))
        
        self.crop_end_x = final_x
        self.crop_end_y = final_y
        
        # Remove previous rectangle
        if self.crop_rectangle:
            self.canvas.delete(self.crop_rectangle)
        
        # Draw new rectangle
        try:
            self.crop_rectangle = self.canvas.create_rectangle(
                self.crop_start_x, self.crop_start_y,
                self.crop_end_x, self.crop_end_y,
                outline="red", width=2, tags="crop"
            )
        except Exception:
            pass  # Silently handle invalid coordinates
    
    def end_crop_selection(self, event):
        """End crop selection with aspect ratio constraint"""
        if not self.is_selecting or not self.current_thumbnail:
            return
            
        self.is_selecting = False
        
        # Get final coordinates, constrained to image bounds
        x = max(self.image_x_offset, min(event.x, self.image_x_offset + self.current_thumbnail.width))
        y = max(self.image_y_offset, min(event.y, self.image_y_offset + self.current_thumbnail.height))
        
        # Apply the same aspect ratio constraint as in update_crop_selection
        if self.crop_start_x is not None and self.crop_start_y is not None:
            # Calculate raw dimensions
            raw_width = abs(x - self.crop_start_x)
            raw_height = abs(y - self.crop_start_y)
            
            # Apply 9:20 minimum aspect ratio constraint
            min_aspect_ratio = 9.0 / 20.0  # 0.45
            
            if raw_height > 0:
                min_width_for_height = raw_height * min_aspect_ratio
                if raw_width < min_width_for_height:
                    # Adjust width to maintain minimum aspect ratio
                    raw_width = min_width_for_height
            
            # Determine final coordinates based on drag direction
            if x >= self.crop_start_x:  # Dragging right
                final_x = self.crop_start_x + raw_width
            else:  # Dragging left
                final_x = self.crop_start_x - raw_width
                
            if y >= self.crop_start_y:  # Dragging down
                final_y = self.crop_start_y + raw_height
            else:  # Dragging up
                final_y = self.crop_start_y - raw_height
            
            # Ensure final coordinates stay within image bounds
            final_x = max(self.image_x_offset, min(final_x, self.image_x_offset + self.current_thumbnail.width))
            final_y = max(self.image_y_offset, min(final_y, self.image_y_offset + self.current_thumbnail.height))
            
            # Recalculate actual width/height after bounds constraint
            actual_width = abs(final_x - self.crop_start_x)
            actual_height = abs(final_y - self.crop_start_y)
            
            # If bounds constraint broke our aspect ratio, adjust the other dimension
            if actual_height > 0:
                required_width = actual_height * min_aspect_ratio
                if actual_width < required_width:
                    # Width was constrained, so adjust height to maintain aspect ratio
                    new_height = actual_width / min_aspect_ratio
                    if final_y >= self.crop_start_y:  # Dragging down
                        final_y = self.crop_start_y + new_height
                    else:  # Dragging up
                        final_y = self.crop_start_y - new_height
                    # Ensure adjusted height stays in bounds
                    final_y = max(self.image_y_offset, min(final_y, self.image_y_offset + self.current_thumbnail.height))
            
            self.crop_end_x = final_x
            self.crop_end_y = final_y
        else:
            self.crop_end_x = x
            self.crop_end_y = y
        
        # Ensure we have valid coordinates
        if (self.crop_start_x is not None and self.crop_start_y is not None and
            self.crop_end_x is not None and self.crop_end_y is not None):
            
            # Ensure valid selection size (at least 10x10 pixels)
            width = abs(self.crop_end_x - self.crop_start_x)
            height = abs(self.crop_end_y - self.crop_start_y)
            
            if width > 10 and height > 10:
                # Update the rectangle to show the final constrained selection
                if self.crop_rectangle:
                    self.canvas.delete(self.crop_rectangle)
                
                self.crop_rectangle = self.canvas.create_rectangle(
                    self.crop_start_x, self.crop_start_y,
                    self.crop_end_x, self.crop_end_y,
                    outline="red", width=2, tags="crop"
                )
                
                self.crop_button.config(state="normal")
                self.clear_button.config(state="normal")
            else:
                self.clear_crop_selection()
        else:
            self.clear_crop_selection()
    
    def clear_crop_selection(self):
        """Clear crop selection"""
        if self.crop_rectangle:
            self.canvas.delete(self.crop_rectangle)
            self.crop_rectangle = None
        
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_end_x = None
        self.crop_end_y = None
        self.crop_button.config(state="disabled")
        self.clear_button.config(state="disabled")
    
    def crop_and_save(self):
        """Crop the original image and save to cropped subdirectory"""
        if not self.current_image or not self.current_thumbnail or not self.image_directory:
            return
            
        if not self.has_valid_crop_selection():
            messagebox.showerror("Error", "Please select a crop area first")
            return
            
        try:
            # Ensure we have all required coordinates
            if (self.crop_start_x is None or self.crop_start_y is None or 
                self.crop_end_x is None or self.crop_end_y is None):
                messagebox.showerror("Error", "Invalid crop selection")
                return
            
            # Calculate crop coordinates on original image
            left = min(self.crop_start_x, self.crop_end_x) - self.image_x_offset
            top = min(self.crop_start_y, self.crop_end_y) - self.image_y_offset
            right = max(self.crop_start_x, self.crop_end_x) - self.image_x_offset
            bottom = max(self.crop_start_y, self.crop_end_y) - self.image_y_offset
            
            # Ensure coordinates are within thumbnail bounds
            left = max(0, left)
            top = max(0, top)
            right = min(self.current_thumbnail.width, right)
            bottom = min(self.current_thumbnail.height, bottom)
            
            # Scale coordinates to original image
            left = int(left * self.scale_factor)
            top = int(top * self.scale_factor)
            right = int(right * self.scale_factor)
            bottom = int(bottom * self.scale_factor)
            
            # Ensure coordinates are within original image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(self.current_image.width, right)
            bottom = min(self.current_image.height, bottom)
            
            # Validate final crop area
            if left >= right or top >= bottom:
                messagebox.showerror("Error", "Invalid crop area")
                return
            
            # Crop the original image
            cropped_image = self.current_image.crop((left, top, right, bottom))
            
            # Apply resizing rules before saving
            cropped_image = self.apply_resizing_rules(cropped_image)
            
            # Create cropped subdirectory if it doesn't exist
            cropped_dir = self.image_directory / "cropped"
            cropped_dir.mkdir(exist_ok=True)
            
            # Save cropped image with dimensions in filename
            current_file = self.image_files[self.current_index]
            original_stem = current_file.stem
            original_suffix = current_file.suffix
            
            # Add dimensions to filename
            dimensions_suffix = f"_{cropped_image.width}x{cropped_image.height}"
            new_filename = f"{original_stem}{dimensions_suffix}{original_suffix}"
            output_path = cropped_dir / new_filename
            
            # Handle file conflicts
            counter = 1
            base_output_path = output_path
            while output_path.exists():
                new_filename = f"{original_stem}{dimensions_suffix}_crop_{counter}{original_suffix}"
                output_path = cropped_dir / new_filename
                counter += 1
            
            cropped_image.save(output_path)
            
            # Update status with resize information
            if hasattr(self, '_resize_applied') and self._resize_applied:
                status_msg = f"Saved: {output_path.name} (resized from {self._original_size[0]}×{self._original_size[1]} to {self._final_size[0]}×{self._final_size[1]})"
            else:
                status_msg = f"Saved: {output_path.name}"
            
            self.status_var.set(status_msg)
            
            # Move to next image
            self.next_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to crop and save image: {str(e)}")
    
    def apply_resizing_rules(self, image):
        """
        Apply image resizing rules before saving:
        - If width > 720px: reduce to 720px maintaining aspect ratio
        - If height > 1600px: reduce to 1600px maintaining aspect ratio  
        - If width < 720px and height < (20/9) * width: increase width to 720px
        - If height < 1600px and width < (9/20) * height: increase width to 720px
        
        Final check ensures no dimension exceeds maximum limits.
        """
        original_width = image.width
        original_height = image.height
        current_width = image.width
        current_height = image.height
        resized = False
        
        # Rule 1: If width > 720px, reduce to 720px maintaining aspect ratio
        if current_width > 720:
            new_width = 720
            new_height = int((new_width * current_height) / current_width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_width, current_height = new_width, new_height
            resized = True
        
        # Rule 2: If height > 1600px, reduce to 1600px maintaining aspect ratio
        if current_height > 1600:
            new_height = 1600
            new_width = int((new_height * current_width) / current_height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_width, current_height = new_width, new_height
            resized = True
        
        # Rule 3: If width < 720px and height < (20/9) * width, increase width to 720px
        if current_width < 720 and current_height < (20/9) * current_width:
            new_width = 720
            new_height = int((new_width * current_height) / current_width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_width, current_height = new_width, new_height
            resized = True
        
        # Rule 4: If height < 1600px and width < (9/20) * height, increase width to 720px
        elif current_height < 1600 and current_width < (9/20) * current_height:
            new_width = 720
            new_height = int((new_width * current_height) / current_width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            current_width, current_height = new_width, new_height
            resized = True
        
        # Final constraint check: ensure no dimension exceeds limits
        final_width = image.width
        final_height = image.height
        
        # If width still > 720px, reduce it
        if final_width > 720:
            new_width = 720
            new_height = int((new_width * final_height) / final_width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized = True
        
        # If height still > 1600px, reduce it
        if image.height > 1600:
            new_height = 1600
            new_width = int((new_height * image.width) / image.height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized = True
        
        # Store resize info for status message
        self._resize_applied = resized
        self._original_size = (original_width, original_height)
        self._final_size = (image.width, image.height)
        
        return image

    def has_valid_crop_selection(self):
        """Check if there's a valid crop selection"""
        return (self.crop_start_x is not None and self.crop_start_y is not None and
                self.crop_end_x is not None and self.crop_end_y is not None)
    
    def previous_image(self):
        """Load previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
            self.update_controls()
    
    def next_image(self):
        """Load next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
            self.update_controls()
        else:
            # All images processed
            messagebox.showinfo("Complete", "All images have been processed!")
            self.status_var.set("All images processed")
    
    def update_controls(self):
        """Update button states"""
        if not self.image_files:
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            self.skip_button.config(state="disabled")
            return
        
        # Previous button
        if self.current_index > 0:
            self.prev_button.config(state="normal")
        else:
            self.prev_button.config(state="disabled")
        
        # Next and Skip buttons
        if self.current_index < len(self.image_files) - 1:
            self.next_button.config(state="normal")
            self.skip_button.config(state="normal")
        else:
            self.next_button.config(state="disabled")
            self.skip_button.config(state="normal")  # Can still skip the last image


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ImageCropper(root)
    root.mainloop()


if __name__ == "__main__":
    main()
