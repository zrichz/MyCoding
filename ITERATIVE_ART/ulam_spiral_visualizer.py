#!/usr/bin/env python3
"""
Ulam Spiral Visualizer
Creates a visual representation of the Ulam Spiral where prime numbers are arranged in a spiral pattern.
White dots represent prime numbers, black dots represent composite numbers.
Grid size: 800x800 pixels
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from math import sqrt
import threading

class UlamSpiralVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Ulam Spiral Visualizer")
        self.root.geometry("900x900")
        
        # Parameters
        self.max_display_size = 600  # Maximum display size (600x600 pixels)
        self.grid_size = 200    # 200x200 grid of numbers (adjustable)
        self.start_number = 1   # Starting number for the spiral
        
        # Data
        self.spiral_data = None
        self.prime_data = None
        
        # Setup GUI
        self.setup_gui()
        
        # Generate initial spiral
        self.generate_spiral()
    
    def setup_gui(self):
        """Setup the GUI with controls and matplotlib canvas"""
        
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Grid size control
        grid_frame = tk.Frame(control_frame)
        grid_frame.pack(fill='x', pady=5)
        
        tk.Label(grid_frame, text="Grid Size:", 
                font=('Arial', 12, 'bold')).pack(side='left')
        
        self.grid_var = tk.IntVar(value=self.grid_size)
        self.grid_scale = tk.Scale(
            grid_frame,
            from_=50,
            to=1000,
            orient='horizontal',
            variable=self.grid_var,
            command=self.on_grid_change,
            length=300,
            resolution=10
        )
        self.grid_scale.pack(side='left', padx=(10, 10))
        
        self.grid_label = tk.Label(grid_frame, text=f"{self.grid_size}x{self.grid_size}", 
                                  font=('Arial', 10, 'bold'))
        self.grid_label.pack(side='left')
        
        # Start number control
        start_frame = tk.Frame(control_frame)
        start_frame.pack(fill='x', pady=5)
        
        tk.Label(start_frame, text="Start Number:", 
                font=('Arial', 12, 'bold')).pack(side='left')
        
        self.start_var = tk.IntVar(value=self.start_number)
        start_entry = tk.Entry(start_frame, textvariable=self.start_var, width=10)
        start_entry.pack(side='left', padx=(10, 10))
        start_entry.bind('<Return>', self.on_start_change)
        
        tk.Button(start_frame, text="Update", 
                 command=self.on_start_change).pack(side='left')
        
        # Info labels
        info_frame = tk.Frame(control_frame)
        info_frame.pack(fill='x', pady=5)
        
        self.total_label = tk.Label(info_frame, text="Total Numbers: 0", 
                                   font=('Arial', 10))
        self.total_label.pack(side='left', padx=(0, 20))
        
        self.primes_label = tk.Label(info_frame, text="Primes: 0", 
                                    font=('Arial', 10))
        self.primes_label.pack(side='left', padx=(0, 20))
        
        self.percent_label = tk.Label(info_frame, text="Prime %: 0.0", 
                                     font=('Arial', 10))
        self.percent_label.pack(side='left')
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill='x', pady=5)
        
        tk.Button(button_frame, text="Regenerate Spiral", 
                 command=self.generate_spiral,
                 font=('Arial', 10, 'bold'),
                 bg='lightblue').pack(side='left', padx=(0, 10))
        
        tk.Button(button_frame, text="Save Image", 
                 command=self.save_image,
                 font=('Arial', 10),
                 bg='lightgreen').pack(side='left', padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to generate spiral")
        progress_label = tk.Label(control_frame, textvariable=self.progress_var)
        progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(control_frame, length=400, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        
        # Matplotlib figure - will be sized dynamically based on grid
        display_size = min(self.max_display_size, self.grid_size)  # Cap at 600 pixels
        fig_size = display_size / 100  # Convert pixels to inches (100 pixels per inch for display)
        self.figure, self.ax = plt.subplots(figsize=(fig_size, fig_size))
        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Configure plot - will be updated when spiral is generated
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#404040')  # Dark grey background
        self.ax.axis('off')  # Remove axes, labels, and ticks
    
    def cell(self, n, x, y, start=1):
        """Calculate the number at position (x, y) in an n×n Ulam spiral"""
        d, y, x = 0, y - n//2, x - (n - 1)//2
        l = 2*max(abs(x), abs(y))
        d = (l*3 + x + y) if y >= x else (l - x - y)
        return (l - 1)**2 + d + start - 1
    
    def sieve_of_eratosthenes(self, limit):
        """Generate prime numbers up to limit using Sieve of Eratosthenes"""
        is_prime = [False, False, True] + [True, False] * (limit // 2)
        
        for x in range(3, 1 + int(sqrt(limit)), 2):
            if not is_prime[x]:
                continue
            for i in range(x*x, limit, x*2):
                is_prime[i] = False
        
        return is_prime
    
    def generate_spiral_data(self):
        """Generate the Ulam spiral data"""
        n = self.grid_size
        start = self.start_number
        
        # Calculate the maximum number in the spiral
        max_number = start + n * n - 1
        
        # Generate prime sieve
        self.progress_var.set("Generating prime sieve...")
        self.root.update()
        
        is_prime = self.sieve_of_eratosthenes(max_number + 1)
        
        # Generate spiral grid
        self.progress_var.set("Generating spiral grid...")
        self.root.update()
        
        spiral_grid = np.zeros((n, n), dtype=int)
        prime_grid = np.zeros((n, n), dtype=bool)
        
        for y in range(n):
            for x in range(n):
                number = self.cell(n, x, y, start)
                spiral_grid[y, x] = number
                if number < len(is_prime):
                    prime_grid[y, x] = is_prime[number]
        
        return spiral_grid, prime_grid, is_prime
    
    def on_grid_change(self, value):
        """Handle grid size slider change"""
        self.grid_size = int(value)
        self.grid_label.config(text=f"{self.grid_size}x{self.grid_size}")
    
    def on_start_change(self, event=None):
        """Handle start number change"""
        try:
            self.start_number = max(1, self.start_var.get())
        except:
            self.start_number = 1
            self.start_var.set(1)
    
    def generate_spiral(self):
        """Generate and display the Ulam spiral"""
        # Start progress bar
        self.progress_bar.start()
        
        # Generate in separate thread to keep UI responsive
        thread = threading.Thread(target=self._generate_spiral_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_spiral_thread(self):
        """Generate spiral in separate thread"""
        try:
            # Generate spiral data
            spiral_data, prime_data, is_prime = self.generate_spiral_data()
            
            # Count primes
            total_numbers = self.grid_size * self.grid_size
            prime_count = np.sum(prime_data)
            prime_percentage = (prime_count / total_numbers) * 100 if total_numbers > 0 else 0
            
            # Update UI on main thread
            self.root.after(0, lambda: self._update_display(spiral_data, prime_data, 
                                                           total_numbers, prime_count, prime_percentage))
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_var.set(f"Error: {str(e)}"))
            self.root.after(0, self.progress_bar.stop)
    
    def _update_display(self, spiral_data, prime_data, total_numbers, prime_count, prime_percentage):
        """Update the display with generated spiral data"""
        try:
            self.progress_var.set("Rendering visualization...")
            
            # Store data
            self.spiral_data = spiral_data
            self.prime_data = prime_data
            
            # Update info labels
            self.total_label.config(text=f"Total Numbers: {total_numbers:,}")
            self.primes_label.config(text=f"Primes: {prime_count:,}")
            self.percent_label.config(text=f"Prime %: {prime_percentage:.2f}")
            
            # Clear the plot
            self.ax.clear()
            
            # Calculate display size (capped at 600x600)
            display_size = min(self.max_display_size, self.grid_size)
            pixel_scale = display_size / self.grid_size
            
            # Create image array for proper pixel display
            display_array = np.full((display_size, display_size, 3), 64, dtype=np.uint8)  # Dark grey background
            
            # Fill in scaled pixel blocks
            for grid_y in range(self.grid_size):
                for grid_x in range(self.grid_size):
                    # Calculate pixel block position and size in display
                    start_x = int(grid_x * pixel_scale)
                    end_x = int((grid_x + 1) * pixel_scale)
                    start_y = int(grid_y * pixel_scale)
                    end_y = int((grid_y + 1) * pixel_scale)
                    
                    # Set color: white (255) for primes, black (0) for composites
                    if prime_data[grid_y, grid_x]:
                        color = [255, 255, 255]  # White
                    else:
                        color = [0, 0, 0]        # Black
                    
                    # Fill pixel block in display array
                    display_array[start_y:end_y, start_x:end_x] = color
            
            # Display the pixel array as an image
            self.ax.imshow(display_array, origin='upper', interpolation='nearest', extent=[0, display_size, display_size, 0])
            
            # Configure plot appearance
            self.ax.set_xlim(0, display_size)
            self.ax.set_ylim(0, display_size)
            self.ax.set_aspect('equal')
            self.ax.set_facecolor('#404040')  # Dark grey background
            self.ax.axis('off')
            
            # Refresh canvas
            self.canvas.draw()
            
            self.progress_var.set("Spiral generated successfully!")
            self.progress_bar.stop()
            
        except Exception as e:
            self.progress_var.set(f"Display error: {str(e)}")
            self.progress_bar.stop()
    
    def save_image(self):
        """Save exact pixel image where each prime is exactly 2x2 pixels"""
        if self.spiral_data is None:
            messagebox.showwarning("No Data", "Please generate a spiral first")
            return
        
        try:
            self.progress_var.set("Generating pixel-exact image...")
            self.progress_bar.start()
            
            # Calculate exact pixel dimensions: grid_size * 2
            save_width = self.grid_size * 2
            save_height = self.grid_size * 2
            
            # Create numpy array for pixel-perfect image
            image_array = np.full((save_height, save_width, 3), 64, dtype=np.uint8)  # Dark grey background
            
            # Fill in 2x2 blocks for each grid cell
            for grid_y in range(self.grid_size):
                for grid_x in range(self.grid_size):
                    # Calculate 2x2 pixel block position
                    pixel_x = grid_x * 2
                    pixel_y = grid_y * 2
                    
                    # Set color: white (255) for primes, black (0) for composites
                    if self.prime_data[grid_y, grid_x]:
                        color = [255, 255, 255]  # White
                    else:
                        color = [0, 0, 0]        # Black
                    
                    # Fill 2x2 block
                    image_array[pixel_y:pixel_y+2, pixel_x:pixel_x+2] = color
            
            # Save directly using PIL for exact pixel control
            from PIL import Image as PILImage
            
            # Convert numpy array to PIL Image
            pil_image = PILImage.fromarray(image_array, mode='RGB')
            
            # Save with no compression or modification
            filename = f"ulam_spiral_{self.grid_size}x{self.grid_size}_start{self.start_number}.png"
            pil_image.save(filename, format='PNG', optimize=False)
            
            self.progress_var.set(f"Pixel-exact image saved as {filename}")
            self.progress_bar.stop()
            print(f"Saved pixel-exact Ulam spiral ({save_width}x{save_height} pixels) as {filename}")
            
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Save Error", f"Could not save image: {str(e)}")

def main():
    """Main function to run the application"""
    # Check dependencies
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install with: pip install matplotlib numpy")
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = UlamSpiralVisualizer(root)
    
    print("Ulam Spiral Visualizer Started")
    print("- White dots represent prime numbers")
    print("- Black dots represent composite numbers") 
    print("- Display capped at 600x600 pixels for performance")
    print("- Saved images are grid_size * 2 pixels (each prime = 2x2 pixels)")
    print("- Adjust grid size with the slider (50x50 to 1000x1000)")
    print("- Change start number to explore different ranges")
    print("- Click 'Regenerate Spiral' after changes")
    print("- Click 'Save Image' to export pixel-exact visualization")
    
    root.mainloop()

if __name__ == "__main__":
    main()
