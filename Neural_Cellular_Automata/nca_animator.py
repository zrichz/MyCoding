"""
NCA Animation Generator
======================

This script loads a trained Neural Cellular Automata model and creates an animated GIF
showing the step-by-step evolution from seed to final image. Perfect for social media sharing!

Features:
- Load any saved NCA model (.pth files)
- Multiple initialization options (center, random, edge, etc.)
- Configurable animation parameters (steps, frame interval, output size)
- Small size scaling (32x32 → 128x128, 64x64 → 128x128) with nearest neighbor
- High-quality GIF output optimized for social media at fixed 30fps

Author: GitHub Copilot
Date: August 31, 2025
"""

"""
HOW TO USE THIS SCRIPT:

1. Train and save a model using NCA_baseline.py
2. Run this script and select your saved model
3. Choose initialization type and animation settings
4. Watch the magic happen as your NCA evolves step by step!
5. Share the resulting GIF on Mastodon, Twitter, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
from pathlib import Path
import sys

# Import the NCA model from the main script
try:
    from NCA_baseline import NeuralCellularAutomata
except ImportError:
    print("Error: Could not import NeuralCellularAutomata from NCA_baseline.py")
    print("Make sure NCA_baseline.py is in the same directory as this script.")
    sys.exit(1)


class NCAAnimationGenerator:
    """Generates animated GIFs from trained NCA models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.frames = []
        self.animation_params = {
            'total_steps': 128,
            'frame_interval': 1,  # Capture every step
            'gif_duration': 33,   # Fixed at 30fps (33ms per frame)
            'image_size': 16,     # Output image size
            'add_labels': True,   # Add step numbers to frames
        }
        
    def load_model(self, model_path):
        """Load a trained NCA model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model with saved parameters
            self.model = NeuralCellularAutomata(
                channel_n=checkpoint['channel_n'],
                fire_rate=checkpoint['fire_rate']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Channels: {checkpoint['channel_n']}")
            print(f"   Fire Rate: {checkpoint['fire_rate']}")
            print(f"   Training Epochs: {checkpoint.get('epoch', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            return False
    
    def create_seed(self, size, init_type='center', seed_value=None):
        """Create initial seed state with various initialization options"""
        if self.model is None:
            raise ValueError("Model must be loaded before creating seed")
            
        if seed_value is not None:
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            
        seed = torch.zeros(1, self.model.channel_n, size, size, device=self.device)
        
        if init_type == 'center':
            # Single center pixel
            seed[:, 3:, size//2, size//2] = 1.0
            
        elif init_type == 'random_single':
            # Single random point
            x = int(torch.randint(size//4, 3*size//4, (1,)).item())
            y = int(torch.randint(size//4, 3*size//4, (1,)).item())
            seed[:, 3:, x, y] = 1.0
            
        elif init_type == 'random_multi':
            # Multiple random points (3-5 points)
            num_seeds = int(torch.randint(3, 6, (1,)).item())
            for _ in range(num_seeds):
                x = int(torch.randint(0, size, (1,)).item())
                y = int(torch.randint(0, size, (1,)).item())
                seed[:, 3:, x, y] = torch.rand(self.model.channel_n-3, device=self.device) * 0.8 + 0.2
                
        elif init_type == 'sparse':
            # Sparse random pixels
            density = 0.01  # 1% density
            mask = torch.rand(size, size, device=self.device) < density
            num_pixels = int(mask.sum().item())
            if num_pixels > 0:
                seed[:, 3:, mask] = torch.rand(self.model.channel_n-3, num_pixels, device=self.device) * 0.7 + 0.3
                
        elif init_type == 'edge':
            # Edge initialization
            edge = torch.randint(0, 4, (1,)).item()
            if edge == 0:  # Top
                seed[:, 3:, 0, size//4:3*size//4] = 1.0
            elif edge == 1:  # Bottom
                seed[:, 3:, -1, size//4:3*size//4] = 1.0
            elif edge == 2:  # Left
                seed[:, 3:, size//4:3*size//4, 0] = 1.0
            else:  # Right
                seed[:, 3:, size//4:3*size//4, -1] = 1.0
                
        elif init_type == 'circle':
            # Small circle in center
            center = size // 2
            radius = 3
            y, x = torch.meshgrid(torch.arange(size, device=self.device), 
                                torch.arange(size, device=self.device), indexing='ij')
            mask = (x - center)**2 + (y - center)**2 <= radius**2
            seed[:, 3:, mask] = 1.0
            
        return seed
    
    def generate_animation_frames(self, seed, total_steps, frame_interval=2):
        """Generate animation frames by running the CA step by step"""
        if self.model is None:
            raise ValueError("Model must be loaded before generating frames")
            
        print(f"🎬 Generating {total_steps} animation frames...")
        print(f"   Capturing every {frame_interval} steps")
        
        self.frames = []
        x = seed.clone()
        
        with torch.no_grad():
            # Capture initial state
            if 0 % frame_interval == 0:
                frame = self.tensor_to_image(x)
                self.frames.append((0, frame))
                
            # Generate frames step by step
            for step in range(1, total_steps + 1):
                x = self.model.update(x)
                
                # Capture frame at specified intervals
                if step % frame_interval == 0 or step == total_steps:
                    frame = self.tensor_to_image(x)
                    self.frames.append((step, frame))
                    
                # Progress indicator
                if step % 20 == 0 or step == total_steps:
                    progress = (step / total_steps) * 100
                    print(f"   Progress: {progress:.1f}% (Step {step}/{total_steps})")
        
        print(f"✅ Generated {len(self.frames)} frames")
        return self.frames
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        # Extract RGB channels and convert to numpy
        rgb = tensor[0, :3].detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
        
        # Convert to PIL Image and resize if needed
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        
        if img.size[0] != self.animation_params['image_size']:
            img = img.resize((self.animation_params['image_size'], self.animation_params['image_size']), 
                           Image.Resampling.NEAREST)
        
        return img
    
    def add_step_label(self, img, step):
        """Add step number label to image"""
        if not self.animation_params['add_labels']:
            return img
            
        # Create a copy to avoid modifying original
        img_with_label = img.copy()
        draw = ImageDraw.Draw(img_with_label)
        
        # Try to load a nice font, fallback to default
        try:
            font_size = max(12, self.animation_params['image_size'] // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Add step number with background
        text = f"Step {step}"
        
        # Get text size for background
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position in top-left corner with padding
        x, y = 5, 5
        
        # Draw background rectangle
        draw.rectangle([x-2, y-2, x+text_width+4, y+text_height+4], 
                      fill=(0, 0, 0, 180))
        
        # Draw text
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        return img_with_label
    
    def create_gif(self, output_path, optimize_for_social=True):
        """Create animated GIF from frames"""
        if not self.frames:
            raise ValueError("No frames to create GIF. Generate frames first.")
            
        print(f"🎨 Creating animated GIF...")
        print(f"   Output: {output_path}")
        print(f"   Frames: {len(self.frames)}")
        print(f"   Duration: {self.animation_params['gif_duration']}ms per frame")
        
        # Prepare frames with labels and scaling
        gif_frames = []
        output_size = self.animation_params['image_size']
        
        for step, img in self.frames:
            labeled_img = self.add_step_label(img, step)
            
            # Apply scaling for small outputs
            if output_size == 16:
                # 16x16 with 4x nearest neighbor scaling to 64x64
                labeled_img = labeled_img.resize((64, 64), Image.Resampling.NEAREST)
            elif output_size == 32:
                # 32x32 with 4x nearest neighbor scaling to 128x128
                labeled_img = labeled_img.resize((128, 128), Image.Resampling.NEAREST)
            elif output_size == 64:
                # 64x64 with 2x nearest neighbor scaling to 128x128  
                labeled_img = labeled_img.resize((128, 128), Image.Resampling.NEAREST)
            
            gif_frames.append(labeled_img)
        
        # Add a few extra frames at the end to pause on final result
        final_frame = gif_frames[-1]
        for _ in range(10):  # Hold final frame for 1 second (10 * 100ms)
            gif_frames.append(final_frame)
        
        # Optimize settings for social media
        if optimize_for_social:
            # Reduce colors for smaller file size
            gif_frames = [img.quantize(colors=128, dither=Image.Dither.FLOYDSTEINBERG) 
                         for img in gif_frames]
        
        # Save GIF
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=self.animation_params['gif_duration'],
            loop=0,  # Infinite loop
            optimize=True
        )
        
        # Get file size for info
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        actual_dimensions = gif_frames[0].size  # Get actual saved dimensions
        print(f"✅ GIF created successfully!")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Original size: {self.animation_params['image_size']}x{self.animation_params['image_size']}")
        print(f"   Output dimensions: {actual_dimensions[0]}x{actual_dimensions[1]}")
        print(f"   Total duration: {len(gif_frames) * self.animation_params['gif_duration'] / 1000:.1f} seconds")
        
        return output_path


class NCAAnimationGUI:
    """GUI for creating NCA animations"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NCA Animation Generator")
        self.root.geometry("800x600")
        
        self.generator = NCAAnimationGenerator()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="1. Load Trained Model", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        ttk.Button(model_frame, text="Select Model File (.pth)", 
                  command=self.load_model).pack(side=tk.LEFT)
        self.model_status = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="2. Animation Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky="nw", padx=(0, 10))
        
        # Initialization type
        ttk.Label(settings_frame, text="Initialization:").grid(row=0, column=0, sticky=tk.W)
        self.init_type = tk.StringVar(value="center")
        init_combo = ttk.Combobox(settings_frame, textvariable=self.init_type, width=15)
        init_combo['values'] = ('center', 'random_single', 'random_multi', 'sparse', 'edge', 'circle')
        init_combo.grid(row=0, column=1, padx=(5, 0))
        init_combo.state(['readonly'])
        
        # Animation steps
        ttk.Label(settings_frame, text="Total Steps:").grid(row=1, column=0, sticky=tk.W)
        self.total_steps = tk.StringVar(value="128")
        ttk.Spinbox(settings_frame, from_=50, to=500, textvariable=self.total_steps, width=15).grid(row=1, column=1, padx=(5, 0))
        
        # Frame interval
        ttk.Label(settings_frame, text="Frame Interval:").grid(row=2, column=0, sticky=tk.W)
        self.frame_interval = tk.StringVar(value="2")
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.frame_interval, width=15).grid(row=2, column=1, padx=(5, 0))
        
        # Image size
        ttk.Label(settings_frame, text="Output Size:").grid(row=3, column=0, sticky=tk.W)
        self.image_size = tk.StringVar(value="16")
        size_combo = ttk.Combobox(settings_frame, textvariable=self.image_size, width=15)
        size_combo['values'] = ('16', '32', '64', '128', '256', '512')
        size_combo.grid(row=3, column=1, padx=(5, 0))
        size_combo.state(['readonly'])
        
        # Frame interval (skip frames)
        ttk.Label(settings_frame, text="Frame Interval:").grid(row=4, column=0, sticky=tk.W)
        self.frame_interval = tk.StringVar(value="1")
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.frame_interval, width=15).grid(row=4, column=1, padx=(5, 0))
        
        # Random seed
        ttk.Label(settings_frame, text="Random Seed:").grid(row=5, column=0, sticky=tk.W)
        self.seed_value = tk.StringVar(value="42")
        ttk.Entry(settings_frame, textvariable=self.seed_value, width=15).grid(row=5, column=1, padx=(5, 0))
        
        # Add labels checkbox
        self.add_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Add step numbers", 
                       variable=self.add_labels).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Info label for fixed GIF speed
        info_label = ttk.Label(settings_frame, text="GIF Speed: Fixed at 30fps (33ms per frame)", 
                              font=('TkDefaultFont', 8), foreground='gray')
        info_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Size info display
        self.size_info = ttk.Label(settings_frame, text="", 
                                  font=('TkDefaultFont', 8), foreground='blue')
        self.size_info.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        
        # Bind size change to update info
        self.image_size.trace('w', self.update_size_info)
        
        # Initialize size info
        self.update_size_info()
        
        # Generation frame
        gen_frame = ttk.LabelFrame(main_frame, text="3. Generate Animation", padding="10")
        gen_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        ttk.Button(gen_frame, text="Create GIF", 
                  command=self.create_gif).pack(side=tk.LEFT)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Load a model to begin")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
    def update_size_info(self, *args):
        """Update the size information display"""
        try:
            size = int(self.image_size.get())
            if size == 16:
                info_text = "Input: 16x16 → Output: 64x64 (4x nearest neighbor scaling)"
            elif size == 32:
                info_text = "Input: 32x32 → Output: 128x128 (4x nearest neighbor scaling)"
            elif size == 64:
                info_text = "Input: 64x64 → Output: 128x128 (2x nearest neighbor scaling)"
            else:
                info_text = f"Input: {size}x{size} → Output: {size}x{size} (no scaling)"
            
            self.size_info.config(text=info_text)
        except:
            self.size_info.config(text="")
        
    def load_model(self):
        """Load a trained NCA model"""
        file_path = filedialog.askopenfilename(
            title="Select trained NCA model",
            filetypes=[("PyTorch models", "*.pth")]
        )
        
        if file_path:
            if self.generator.load_model(file_path):
                self.model_status.config(text=f"✅ {os.path.basename(file_path)}", foreground="green")
                device_info = "GPU" if torch.cuda.is_available() else "CPU"
                self.status_var.set(f"Model loaded - Device: {device_info}")
            else:
                messagebox.showerror("Error", "Failed to load model")
                
    def update_animation_params(self):
        """Update animation parameters from GUI"""
        try:
            self.generator.animation_params.update({
                'total_steps': int(self.total_steps.get()),
                'frame_interval': int(self.frame_interval.get()),
                'gif_duration': 33,  # Fixed at 30fps
                'image_size': int(self.image_size.get()),
                'add_labels': self.add_labels.get(),
            })
        except ValueError as e:
            raise ValueError(f"Invalid parameter value: {e}")
            
    def create_gif(self):
        """Create the full animated GIF"""
        if self.generator.model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
            
        # Ask for output file
        output_path = filedialog.asksaveasfilename(
            title="Save animated GIF",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")]
        )
        
        if not output_path:
            return
            
        try:
            self.update_animation_params()
            self.status_var.set("Generating full animation...")
            self.root.update()
            
            # Create seed
            grid_size = 128  # Good balance of detail and performance
            seed_val = int(self.seed_value.get()) if self.seed_value.get().isdigit() else None
            seed = self.generator.create_seed(grid_size, self.init_type.get(), seed_val)
            
            # Generate all frames
            frames = self.generator.generate_animation_frames(
                seed, 
                self.generator.animation_params['total_steps'],
                self.generator.animation_params['frame_interval']
            )
            
            self.status_var.set("Creating GIF file...")
            self.root.update()
            
            # Create GIF
            self.generator.create_gif(output_path, optimize_for_social=True)
            
            self.status_var.set(f"GIF created: {os.path.basename(output_path)}")
            
            # Ask if user wants to open the file
            if messagebox.askyesno("Success", f"GIF created successfully!\n\nFile: {output_path}\n\nOpen in default viewer?"):
                os.startfile(output_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"GIF creation failed: {str(e)}")
            self.status_var.set("GIF creation failed")


def main():
    """Main function"""
    root = tk.Tk()
    app = NCAAnimationGUI(root)
    
    def on_closing():
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
