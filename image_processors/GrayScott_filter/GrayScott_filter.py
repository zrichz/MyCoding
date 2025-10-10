from tkinter import Tk, Label, Button, Entry, filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk, ImageFilter
import numpy as np

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Morphological operations will use basic numpy implementation.")


def sharpen_image(image):
    return image.filter(ImageFilter.SHARPEN)

def blur_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=1))

class GrayScottFilterApp:
    def __init__(self, master):
        self.master = master
        master.title("Gray-Scott Filter - Image Processor")
        
        # Set window size for 1920x1080 screen with 1070 pixel height
        window_width = 1200
        window_height = 1070
        master.geometry(f"{window_width}x{window_height}")
        
        # Center the window on screen
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        master.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Create main frame with padding
        from tkinter import ttk, Frame
        main_frame = Frame(master, bg='lightgray', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        # Controls frame at the top
        controls_frame = Frame(main_frame, bg='lightgray')
        controls_frame.pack(fill='x', pady=(0, 20))

        self.label = Label(controls_frame, text="Gray-Scott Filter Image Processor", 
                          font=("Arial", 14, "bold"), bg='lightgray')
        self.label.pack(pady=(0, 10))

        # Button frame for horizontal layout
        button_frame = Frame(controls_frame, bg='lightgray')
        button_frame.pack(pady=5)

        self.load_button = Button(button_frame, text="Load Image", command=self.load_image,
                                 width=12, height=2, font=("Arial", 10))
        self.load_button.pack(side='left', padx=5)

        # Iterations frame
        iter_frame = Frame(button_frame, bg='lightgray')
        iter_frame.pack(side='left', padx=20)

        self.iterations_label = Label(iter_frame, text="Iterations:", bg='lightgray', font=("Arial", 10))
        self.iterations_label.pack(side='left')

        self.iterations_entry = Entry(iter_frame, width=8, font=("Arial", 10))
        self.iterations_entry.pack(side='left', padx=5)
        self.iterations_entry.insert(0, "100")  # Default value

        self.process_button = Button(button_frame, text="Process Image", command=self.process_image,
                                   width=12, height=2, font=("Arial", 10))
        self.process_button.pack(side='left', padx=5)

        self.save_button = Button(button_frame, text="Save Image", command=self.save_image,
                                width=12, height=2, font=("Arial", 10))
        self.save_button.pack(side='left', padx=5)

        # Second row for morphological operations
        morph_frame = Frame(controls_frame, bg='lightgray')
        morph_frame.pack(pady=10)

        self.binarize_button = Button(morph_frame, text="Binarize Image", command=self.binarize_image,
                                    width=12, height=2, font=("Arial", 10), state="disabled")
        self.binarize_button.pack(side='left', padx=5)

        self.erode_button = Button(morph_frame, text="Erode", command=self.erode_image,
                                 width=12, height=2, font=("Arial", 10), state="disabled")
        self.erode_button.pack(side='left', padx=5)

        self.dilate_button = Button(morph_frame, text="Dilate", command=self.dilate_image,
                                  width=12, height=2, font=("Arial", 10), state="disabled")
        self.dilate_button.pack(side='left', padx=5)

        self.reset_button = Button(morph_frame, text="Reset to Original", command=self.reset_image,
                                 width=15, height=2, font=("Arial", 10), state="disabled")
        self.reset_button.pack(side='left', padx=5)

        # Canvas frame with scrollbars for large images
        canvas_frame = Frame(main_frame, bg='white', relief='sunken', bd=2)
        canvas_frame.pack(fill='both', expand=True)

        # Create scrollbars
        from tkinter import Scrollbar
        v_scrollbar = Scrollbar(canvas_frame, orient='vertical')
        h_scrollbar = Scrollbar(canvas_frame, orient='horizontal')
        
        self.canvas = Canvas(canvas_frame, bg='white', 
                            yscrollcommand=v_scrollbar.set,
                            xscrollcommand=h_scrollbar.set)
        
        # Configure scrollbars
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        self.canvas.pack(side='left', fill='both', expand=True)
        
        # Bind mouse wheel scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_horizontal_mousewheel)
        
        # Make canvas focusable for mouse wheel events
        self.canvas.focus_set()

        # Status bar at the bottom
        status_frame = Frame(main_frame, bg='lightgray')
        status_frame.pack(fill='x', pady=(10, 0))

        self.status_label = Label(status_frame, text="Ready - Load an image to begin", 
                                 bg='lightgray', font=("Arial", 10), anchor='w')
        self.status_label.pack(fill='x')

        self.image = None
        self.processed_image = None
        self.original_image = None  # Store original for reset
        self.is_binarized = False   # Track if image is binarized

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                      ("PNG files", "*.png"),
                      ("JPEG files", "*.jpg *.jpeg"),
                      ("All files", "*.*")]
        )
        if file_path:
            self.status_label.config(text="Loading image...")
            self.master.update()
            
            self.image = Image.open(file_path)
            self.original_image = self.image.copy()  # Store original for reset
            self.is_binarized = False
            
            # No resizing - keep original image size
            self.status_label.config(text=f"Image loaded: {self.image.width}x{self.image.height} pixels")
            
            # Enable reset button when image is loaded
            self.reset_button.config(state="normal")
                
            self.display_image(self.image)

    def display_image(self, image):
        self.processed_image = image
        self.tk_image = ImageTk.PhotoImage(image)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Display image at full size (no centering, start at top-left)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        
        # Update scroll region to match image size
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
        
        # Add image info text (positioned to be visible even when scrolled)
        info_bg = self.canvas.create_rectangle(5, 5, 300, 30, fill='white', outline='black')
        info_text = f"Image: {image.width}x{image.height} pixels"
        self.canvas.create_text(10, 10, anchor='nw', text=info_text, 
                               fill='black', font=("Arial", 10))
        
        # Reset scroll position to top-left
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def process_image(self):
        if self.image:
            try:
                iterations = int(self.iterations_entry.get())
                if iterations <= 0:
                    self.status_label.config(text="Error: Please enter a positive number of iterations")
                    return
                    
                self.status_label.config(text=f"Processing image with {iterations} iterations...")
                self.master.update()
                
                # Convert to greyscale as first step
                processed = self.image.convert('L')
                for i in range(iterations):
                    self.status_label.config(text=f"Processing iteration {i+1}/{iterations}...")
                    self.master.update()
                    processed = sharpen_image(processed)
                    processed = sharpen_image(processed) # applying sharpening multiple times
                    processed = blur_image(processed)
                    
                self.display_image(processed)
                self.status_label.config(text=f"Processing complete! Applied {iterations} iterations of Gray-Scott filter")
                
                # Enable binarize button after processing
                self.binarize_button.config(state="normal")
                
            except ValueError:
                self.status_label.config(text="Error: Please enter a valid number of iterations")
        else:
            self.status_label.config(text="Error: Please load an image first")

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(
                title="Save Processed Image",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"),
                          ("JPEG files", "*.jpg"),
                          ("BMP files", "*.bmp"),
                          ("TIFF files", "*.tiff"),
                          ("All files", "*.*")]
            )
            if file_path:
                self.status_label.config(text="Saving image...")
                self.master.update()
                self.processed_image.save(file_path)
                self.status_label.config(text=f"Image saved successfully: {file_path}")
        else:
            self.status_label.config(text="Error: No processed image to save")
    
    def binarize_image(self):
        """Convert image to pure black and white using 50% threshold"""
        if self.processed_image:
            self.status_label.config(text="Binarizing image...")
            self.master.update()
            
            # Convert to grayscale if not already
            if self.processed_image.mode != 'L':
                grayscale = self.processed_image.convert('L')
            else:
                grayscale = self.processed_image.copy()
            
            # Convert to numpy array for thresholding
            img_array = np.array(grayscale)
            
            # Apply 50% threshold (127.5 for 0-255 range)
            binary_array = (img_array > 127).astype(np.uint8) * 255
            
            # Convert back to PIL Image
            binary_image = Image.fromarray(binary_array, mode='L')
            
            self.display_image(binary_image)
            self.is_binarized = True
            
            # Enable morphological operations after binarization
            self.erode_button.config(state="normal")
            self.dilate_button.config(state="normal")
            
            self.status_label.config(text="Image binarized using 50% threshold")
        else:
            self.status_label.config(text="Error: No processed image to binarize")
    
    def erode_image(self):
        """Apply erosion morphological operation"""
        if self.processed_image and self.is_binarized:
            self.status_label.config(text="Applying erosion...")
            self.master.update()
            
            # Convert to numpy array
            img_array = np.array(self.processed_image)
            binary_array = img_array > 127
            
            if SCIPY_AVAILABLE:
                # Use scipy for proper morphological operations
                from scipy import ndimage
                structure = np.ones((3, 3), dtype=bool)
                eroded = ndimage.binary_erosion(binary_array, structure=structure)
            else:
                # Basic erosion implementation using numpy
                eroded = np.zeros_like(binary_array)
                h, w = binary_array.shape
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        # Check if all pixels in 3x3 neighborhood are white
                        if np.all(binary_array[i-1:i+2, j-1:j+2]):
                            eroded[i, j] = True
            
            # Convert back to 0-255 range
            eroded_array = (eroded.astype(np.uint8)) * 255
            
            # Convert back to PIL Image
            eroded_image = Image.fromarray(eroded_array, mode='L')
            
            self.display_image(eroded_image)
            self.status_label.config(text="Erosion applied (3x3 structuring element)")
        else:
            self.status_label.config(text="Error: Image must be binarized first")
    
    def dilate_image(self):
        """Apply dilation morphological operation"""
        if self.processed_image and self.is_binarized:
            self.status_label.config(text="Applying dilation...")
            self.master.update()
            
            # Convert to numpy array
            img_array = np.array(self.processed_image)
            binary_array = img_array > 127
            
            if SCIPY_AVAILABLE:
                # Use scipy for proper morphological operations
                from scipy import ndimage
                structure = np.ones((3, 3), dtype=bool)
                dilated = ndimage.binary_dilation(binary_array, structure=structure)
            else:
                # Basic dilation implementation using numpy
                dilated = np.zeros_like(binary_array)
                h, w = binary_array.shape
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        # Check if any pixel in 3x3 neighborhood is white
                        if np.any(binary_array[i-1:i+2, j-1:j+2]):
                            dilated[i, j] = True
            
            # Convert back to 0-255 range
            dilated_array = (dilated.astype(np.uint8)) * 255
            
            # Convert back to PIL Image
            dilated_image = Image.fromarray(dilated_array, mode='L')
            
            self.display_image(dilated_image)
            self.status_label.config(text="Dilation applied (3x3 structuring element)")
        else:
            self.status_label.config(text="Error: Image must be binarized first")
    
    def reset_image(self):
        """Reset to original loaded image"""
        if self.original_image:
            self.image = self.original_image.copy()
            self.is_binarized = False
            
            self.display_image(self.image)
            
            # Reset button states
            self.binarize_button.config(state="disabled")
            self.erode_button.config(state="disabled")
            self.dilate_button.config(state="disabled")
            
            self.status_label.config(text="Reset to original image")
        else:
            self.status_label.config(text="Error: No original image to reset to")
    
    def _on_mousewheel(self, event):
        """Handle vertical mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_horizontal_mousewheel(self, event):
        """Handle horizontal mouse wheel scrolling (Shift + wheel)"""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

if __name__ == "__main__":
    root = Tk()
    app = GrayScottFilterApp(root)
    root.mainloop()
