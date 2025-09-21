"""
Interactive L-System Fern Generator
Creates fern-like patterns using L-System iteration with real-time parameter control.
Supports both geometric ferns and image-based iterative processing.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import threading
import time

class InteractiveLSystemFern:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive L-System Fern Generator")
        self.root.geometry("1400x900")
        
        # Processing parameters
        self.depth = tk.IntVar(value=4)
        self.angle = tk.IntVar(value=25)
        self.length_ratio = tk.DoubleVar(value=0.7)
        self.width_ratio = tk.DoubleVar(value=0.8)
        self.branch_angle = tk.IntVar(value=30)
        self.draw_mode = tk.StringVar(value="geometric")
        self.color_mode = tk.StringVar(value="depth_gradient")
        
        # Image processing
        self.current_image = None
        self.original_image = None
        self.canvas_size = (800, 600)
        
        # Auto-update control
        self.auto_update = tk.BooleanVar(value=True)
        self.update_pending = False
        
        self.setup_ui()
        self.generate_fern()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(control_frame)
        self.setup_display(display_frame)
        
    def setup_controls(self, parent):
        """Setup control panel"""
        # Title
        title_label = ttk.Label(parent, text="L-System Fern Parameters", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(parent, text="Draw Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Geometric Fern", variable=self.draw_mode, 
                       value="geometric", command=self.on_parameter_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Image Processing", variable=self.draw_mode, 
                       value="image", command=self.on_parameter_change).pack(anchor=tk.W)
        
        # Image controls
        img_frame = ttk.LabelFrame(parent, text="Image Controls", padding=10)
        img_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(img_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(img_frame, text="Reset to Geometric", command=self.reset_to_geometric).pack(fill=tk.X, pady=2)
        
        # Parameter controls
        param_frame = ttk.LabelFrame(parent, text="Fern Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Depth
        self.create_slider(param_frame, "Recursion Depth:", self.depth, 1, 7, self.on_parameter_change)
        
        # Base angle
        self.create_slider(param_frame, "Base Angle (°):", self.angle, 5, 60, self.on_parameter_change)
        
        # Branch angle
        self.create_slider(param_frame, "Branch Angle (°):", self.branch_angle, 10, 90, self.on_parameter_change)
        
        # Length ratio
        self.create_slider(param_frame, "Length Ratio:", self.length_ratio, 0.3, 0.9, self.on_parameter_change)
        
        # Width ratio
        self.create_slider(param_frame, "Width Ratio:", self.width_ratio, 0.3, 1.0, self.on_parameter_change)
        
        # Color mode
        color_frame = ttk.LabelFrame(parent, text="Color Mode", padding=10)
        color_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(color_frame, text="Depth Gradient", variable=self.color_mode, 
                       value="depth_gradient", command=self.on_parameter_change).pack(anchor=tk.W)
        ttk.Radiobutton(color_frame, text="Angle Based", variable=self.color_mode, 
                       value="angle_based", command=self.on_parameter_change).pack(anchor=tk.W)
        ttk.Radiobutton(color_frame, text="Natural Green", variable=self.color_mode, 
                       value="natural", command=self.on_parameter_change).pack(anchor=tk.W)
        
        # Auto-update checkbox
        ttk.Checkbutton(parent, text="Auto Update", variable=self.auto_update).pack(pady=10)
        
        # Manual update button
        ttk.Button(parent, text="Update Fern", command=self.generate_fern).pack(fill=tk.X, pady=5)
        
        # Export button
        ttk.Button(parent, text="Export Image", command=self.export_image).pack(fill=tk.X, pady=5)
        
    def create_slider(self, parent, label, variable, min_val, max_val, callback):
        """Create a labeled slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable, 
                          orient=tk.HORIZONTAL, command=callback)
        slider.pack(fill=tk.X)
        
        # Value label
        value_label = ttk.Label(frame, text=f"{variable.get():.1f}")
        value_label.pack(anchor=tk.E)
        
        # Update value label when slider moves
        def update_label(*args):
            if isinstance(variable.get(), int):
                value_label.config(text=f"{variable.get()}")
            else:
                value_label.config(text=f"{variable.get():.1f}")
        variable.trace('w', update_label)
        
    def setup_display(self, parent):
        """Setup display canvas"""
        # Canvas for drawing
        self.canvas = tk.Canvas(parent, width=self.canvas_size[0], height=self.canvas_size[1], 
                               bg='white', relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(parent, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
    def on_parameter_change(self, *args):
        """Handle parameter changes"""
        if self.auto_update.get() and not self.update_pending:
            self.update_pending = True
            self.root.after(100, self.delayed_update)  # Debounce updates
            
    def delayed_update(self):
        """Delayed update to prevent too frequent refreshes"""
        self.update_pending = False
        self.generate_fern()
        
    def load_image(self):
        """Load an image for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and process image
                img = Image.open(file_path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to fit canvas while maintaining aspect ratio
                img.thumbnail(self.canvas_size, Image.Resampling.LANCZOS)
                
                # Create 2:1 aspect ratio version
                target_w = min(self.canvas_size[0], img.size[0])
                target_h = target_w // 2
                
                final_img = Image.new('RGB', (target_w, target_h), (128, 128, 128))
                
                # Center the image
                paste_x = (target_w - img.size[0]) // 2
                paste_y = (target_h - img.size[1]) // 2
                final_img.paste(img, (paste_x, paste_y))
                
                self.original_image = final_img
                self.current_image = final_img.copy()
                self.draw_mode.set("image")
                
                self.status_label.config(text=f"Loaded: {target_w}x{target_h} image")
                self.generate_fern()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def reset_to_geometric(self):
        """Reset to geometric mode"""
        self.draw_mode.set("geometric")
        self.original_image = None
        self.current_image = None
        self.generate_fern()
        
    def generate_fern(self):
        """Generate the fern pattern"""
        self.status_label.config(text="Generating...")
        self.root.update()
        
        try:
            if self.draw_mode.get() == "geometric":
                self.generate_geometric_fern()
            else:
                self.generate_image_fern()
                
            self.status_label.config(text="Ready")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Generation error: {e}")
            
    def generate_geometric_fern(self):
        """Generate geometric fern pattern"""
        # Create image
        img = Image.new('RGB', self.canvas_size, (240, 248, 255))  # Light blue background
        draw = ImageDraw.Draw(img)
        
        # Starting parameters
        start_x = self.canvas_size[0] // 2
        start_y = int(self.canvas_size[1] * 0.9)  # Near bottom
        initial_length = self.canvas_size[1] // 3
        
        # Draw main stem
        self.draw_fern_branch(draw, start_x, start_y, -90, initial_length, 
                             self.depth.get(), 0, initial_length * 0.1)
        
        # Display on canvas
        self.display_image(img)
        
    def draw_fern_branch(self, draw, x, y, angle, length, depth, cumulative_angle, thickness):
        """Recursively draw fern branches"""
        if depth <= 0 or length < 2:
            return
            
        # Calculate end point
        rad = math.radians(angle)
        end_x = x + length * math.cos(rad)
        end_y = y + length * math.sin(rad)
        
        # Get color based on mode
        color = self.get_branch_color(depth, cumulative_angle)
        
        # Draw main branch
        if thickness > 1:
            # Draw thick line as polygon for stems
            offset = thickness / 2
            perp_rad = rad + math.pi / 2
            offset_x = offset * math.cos(perp_rad)
            offset_y = offset * math.sin(perp_rad)
            
            points = [
                (x - offset_x, y - offset_y),
                (x + offset_x, y + offset_y),
                (end_x + offset_x, end_y + offset_y),
                (end_x - offset_x, end_y - offset_y)
            ]
            draw.polygon(points, fill=color)
        else:
            # Draw thin line for leaves
            draw.line([(x, y), (end_x, end_y)], fill=color, width=max(1, int(thickness)))
        
        if depth > 1:
            # Calculate new parameters
            new_length = length * self.length_ratio.get()
            new_thickness = thickness * self.width_ratio.get()
            branch_spread = self.branch_angle.get()
            angle_increment = self.angle.get()
            
            # Left branch
            left_angle = angle - branch_spread + cumulative_angle
            self.draw_fern_branch(draw, end_x, end_y, left_angle, new_length, 
                                 depth - 1, cumulative_angle + angle_increment, new_thickness)
            
            # Right branch  
            right_angle = angle + branch_spread + cumulative_angle
            self.draw_fern_branch(draw, end_x, end_y, right_angle, new_length,
                                 depth - 1, cumulative_angle - angle_increment, new_thickness)
            
            # Continue main stem (slightly curved)
            main_angle = angle + cumulative_angle * 0.3
            self.draw_fern_branch(draw, end_x, end_y, main_angle, new_length * 1.2,
                                 depth - 1, cumulative_angle, new_thickness)
                                 
    def get_branch_color(self, depth, cumulative_angle):
        """Get color for branch based on mode"""
        if self.color_mode.get() == "depth_gradient":
            # Green gradient by depth
            intensity = int(255 * (depth / self.depth.get()))
            return (max(0, 255 - intensity), intensity, max(0, 128 - intensity // 2))
            
        elif self.color_mode.get() == "angle_based":
            # Color based on angle
            angle_norm = (cumulative_angle % 360) / 360
            r = int(255 * abs(math.sin(angle_norm * math.pi)))
            g = int(255 * abs(math.cos(angle_norm * math.pi)))
            b = int(128 * abs(math.sin(2 * angle_norm * math.pi)))
            return (r, g, b)
            
        else:  # natural
            # Natural fern colors
            base_green = (34, 139, 34)  # Forest green
            leaf_green = (107, 142, 35)  # Olive drab
            
            if depth > self.depth.get() // 2:
                return base_green
            else:
                return leaf_green
                
    def generate_image_fern(self):
        """Generate fern pattern on loaded image"""
        if self.original_image is None:
            self.generate_geometric_fern()
            return
            
        # Process image using L-System
        result_img = self.process_image_l_system(self.original_image.copy())
        self.display_image(result_img)
        
    def process_image_l_system(self, img):
        """Process image using L-System subdivision"""
        w, h = img.size
        
        # Ensure 2:1 ratio for proper L-System processing
        if abs(w - h * 2) > 1:  # Allow small rounding differences
            if w > h * 2:
                # Image is too wide, crop from center
                new_w = h * 2
                crop_x = (w - new_w) // 2
                img = img.crop((crop_x, 0, crop_x + new_w, h))
            else:
                # Image is too narrow, pad with gray
                new_w = h * 2
                new_img = Image.new('RGB', (new_w, h), (128, 128, 128))
                paste_x = (new_w - w) // 2
                new_img.paste(img, (paste_x, 0))
                img = new_img
            
        print(f"Processing image: {img.size[0]}x{img.size[1]} at depth {self.depth.get()}")
        
        # Create canvas with padding for rotations
        canvas_w = img.size[0] + 400  # Extra space for rotations
        canvas_h = img.size[1] + 400
        canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))
        
        # Center original image on canvas
        offset_x = (canvas_w - img.size[0]) // 2
        offset_y = (canvas_h - img.size[1]) // 2
        canvas.paste(img, (offset_x, offset_y))
        
        # Process with L-System
        result = self.recursive_image_split_advanced(canvas, offset_x, offset_y, 
                                                   img.size[0], img.size[1], 
                                                   self.depth.get(), self.angle.get(), 0)
        
        # Trim back to reasonable size
        bbox = result.getbbox()
        if bbox:
            # Add some padding around the content
            pad = 50
            crop_box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                       min(result.size[0], bbox[2] + pad), min(result.size[1], bbox[3] + pad))
            result = result.crop(crop_box)
            
        return result
        
    def recursive_image_split_advanced(self, img, x, y, w, h, depth, angle, cumulative_angle):
        """Advanced recursive split with proper buffering to prevent overwrites"""
        if depth <= 0:
            return img

        print(f"Level {depth}: Processing area ({x},{y}) {w}x{h}")
        
        # For L-System fern patterns, we need 2:1 rectangles split into squares
        sq = h  # Square size (height of the rectangle)

        # Define boxes for left/right squares from 2:1 rectangle
        left_box  = (x, y, x+sq, y+sq)        # Left square
        right_box = (x+sq, y, x+w, y+sq)      # Right square

        # The hinge point where pieces connect - at BASE of image, center of width
        hinge_x = x + w // 2                  # Center of width
        hinge_y = y + h                       # Base of image (bottom edge)

        # Calculate cumulative angles for unfurling effect
        left_total_angle = cumulative_angle + angle      # Unfurl counterclockwise
        right_total_angle = cumulative_angle - angle     # Unfurl clockwise

        # ============================================================================
        # STEP 1: EXTRACT PIECES INTO SEPARATE BUFFERS
        # ============================================================================
        
        try:
            # Extract pieces from the current image state
            left_piece = img.crop(left_box)
            right_piece = img.crop(right_box)
            
            # Create completely separate working canvases for each piece
            buffer_size = max(img.size[0], img.size[1]) + 200  # Extra space for rotations
            left_buffer = Image.new('RGB', (buffer_size, buffer_size), (240, 240, 240))
            right_buffer = Image.new('RGB', (buffer_size, buffer_size), (240, 240, 240))
            
            # Calculate buffer centers
            buffer_center = buffer_size // 2
            
            # Place pieces in buffers - position them so the base-center hinge aligns with buffer center
            left_buffer.paste(left_piece, (buffer_center - sq//2, buffer_center - sq))
            right_buffer.paste(right_piece, (buffer_center - sq//2, buffer_center - sq))

            # ============================================================================
            # STEP 2: ROTATE PIECES AROUND HINGE POINTS IN ISOLATION
            # ============================================================================
            
            left_hinge_in_buffer = (buffer_center, buffer_center)
            left_rotated_buffer = left_buffer.rotate(left_total_angle, center=left_hinge_in_buffer, expand=False)
            
            right_hinge_in_buffer = (buffer_center, buffer_center)
            right_rotated_buffer = right_buffer.rotate(right_total_angle, center=right_hinge_in_buffer, expand=False)
            
            # ============================================================================
            # STEP 3: PROCESS CHILDREN IN COMPLETE ISOLATION
            # ============================================================================
            
            if depth > 1:
                # Process left piece completely in its own buffer
                left_rect_h = sq // 2  # Half height for 2:1 ratio
                left_final_buffer = self.recursive_image_split_advanced(
                    left_rotated_buffer, 
                    buffer_center - sq, buffer_center - sq//2, sq, left_rect_h,
                    depth-1, angle, left_total_angle
                )
                
                # Process right piece completely in its own buffer  
                right_rect_h = sq // 2  # Half height for 2:1 ratio
                right_final_buffer = self.recursive_image_split_advanced(
                    right_rotated_buffer,
                    buffer_center, buffer_center - sq//2, sq, right_rect_h, 
                    depth-1, angle, right_total_angle
                )
            else:
                left_final_buffer = left_rotated_buffer
                right_final_buffer = right_rotated_buffer

            # ============================================================================
            # STEP 4: COMPOSITE BACK TO MAIN IMAGE
            # ============================================================================
            
            # Create result image as copy of input
            result_img = img.copy()
            
            # Clear the original area with debug outline
            draw_result = ImageDraw.Draw(result_img)
            outline_color = (255, max(50, 255 - depth * 30), max(50, 255 - depth * 30))
            draw_result.rectangle([x, y, x+w, y+h], outline=outline_color, width=2)
            draw_result.rectangle([x, y, x+w, y+h], fill=(240, 240, 240))
            
            # Position buffer content back to original hinge point
            left_paste_x = hinge_x - buffer_center
            left_paste_y = hinge_y - buffer_center
            
            right_paste_x = hinge_x - buffer_center
            right_paste_y = hinge_y - buffer_center
            
            # Create masks for clean compositing
            def create_content_mask(buffer):
                """Create a mask of non-background pixels"""
                buffer_array = np.array(buffer)
                # Background is (240, 240, 240)
                mask = ~((buffer_array[:,:,0] == 240) & 
                        (buffer_array[:,:,1] == 240) & 
                        (buffer_array[:,:,2] == 240))
                return Image.fromarray((mask * 255).astype('uint8')).convert('L')
            
            left_mask = create_content_mask(left_final_buffer)
            right_mask = create_content_mask(right_final_buffer)
            
            # Paste using masks to avoid overwriting
            if (left_paste_x > -buffer_size and left_paste_y > -buffer_size and 
                left_paste_x < img.size[0] and left_paste_y < img.size[1]):
                result_img.paste(left_final_buffer, (left_paste_x, left_paste_y), left_mask)
            
            if (right_paste_x > -buffer_size and right_paste_y > -buffer_size and 
                right_paste_x < img.size[0] and right_paste_y < img.size[1]):
                result_img.paste(right_final_buffer, (right_paste_x, right_paste_y), right_mask)
                
        except Exception as e:
            print(f"Image processing error at depth {depth}: {e}")
            return img
            
        return result_img
        
    def display_image(self, img):
        """Display image on canvas"""
        # Resize to fit canvas if needed
        display_img = img.copy()
        if img.size[0] > self.canvas_size[0] or img.size[1] > self.canvas_size[1]:
            display_img.thumbnail(self.canvas_size, Image.Resampling.LANCZOS)
            
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)
        
        # Clear canvas and display
        self.canvas.delete("all")
        
        # Center image on canvas
        x = (self.canvas_size[0] - display_img.size[0]) // 2
        y = (self.canvas_size[1] - display_img.size[1]) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_photo_refs'):
            self._photo_refs = []
        self._photo_refs.append(photo)
        
    def export_image(self):
        """Export current image"""
        try:
            # Generate high-res version
            if self.draw_mode.get() == "geometric":
                # Create larger version for export
                export_size = (1600, 1200)
                img = Image.new('RGB', export_size, (240, 248, 255))
                draw = ImageDraw.Draw(img)
                
                start_x = export_size[0] // 2
                start_y = int(export_size[1] * 0.9)
                initial_length = export_size[1] // 3
                
                self.draw_fern_branch(draw, start_x, start_y, -90, initial_length,
                                     self.depth.get(), 0, initial_length * 0.1)
            else:
                if self.original_image:
                    img = self.process_image_l_system(self.original_image.copy())
                else:
                    messagebox.showwarning("Warning", "No image to export")
                    return
                    
            # Save file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                img.save(file_path)
                messagebox.showinfo("Success", f"Image exported to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

def main():
    """Main function"""
    root = tk.Tk()
    app = InteractiveLSystemFern(root)
    root.mainloop()

if __name__ == "__main__":
    main()