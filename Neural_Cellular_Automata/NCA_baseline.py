"""
2D Neural Cellular Automata for Image Recreation
===============================================

This script implements a Neural Cellular Automata that can learn to recreate
target images from various random initializations. It includes a tkinter GUI for easy interaction.

Features:
- Load target images
- Train NCA to recreate the image using random initialization for robustness
- Real-time visualization of training progress
- Save/load trained models
- Adjustable hyperparameters
- Test different initialization methods (center pixel vs random)

Author: GitHub Copilot
Date: August 31, 2025
"""

"""
WHAT IS NEURAL CELLULAR AUTOMATA (NCA)?

Think of it like teaching pixels to paint themselves:
- Start with mostly empty pixels (like a blank canvas)
- Each pixel looks at its neighbors and decides how to change its color
- The AI learns the "rules" for how pixels should behave to recreate your target image
- After many small steps, the pixels self-organize into the desired picture

This script trains an AI to learn these pixel behavior rules, then tests if it can
recreate your image starting from random scattered pixels (proving it really learned
the pattern, not just memorized one specific path).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import os


class NeuralCellularAutomata(nn.Module):
    """
    Neural Cellular Automata model that learns to recreate target images.
    """
    
    def __init__(self, channel_n=16, fire_rate=0.5):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        
        # Perception network - detects local patterns
        self.perception = nn.Conv2d(channel_n*3, 128, 1)
        
        # Update network - decides how to update cells
        self.fc0 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, channel_n, bias=False)
        
        # Initialize weights
        with torch.no_grad():
            self.fc1.weight.zero_()
            
    def perceive(self, x, angle=0.0):
        """Perceive the environment using sobel filters"""
        device = x.device
        
        identify = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=device)
        
        dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device) / 8.0
        dy = dx.T
        
        c, s = torch.cos(torch.tensor(angle, device=device)), torch.sin(torch.tensor(angle, device=device))
        kernel = torch.stack([identify, c*dx-s*dy, s*dx+c*dy], 0)[:, None, ...]
        kernel = kernel.repeat(self.channel_n, 1, 1, 1)
        
        y = F.conv2d(x, kernel, groups=self.channel_n, padding=1)
        return y
    
    def update(self, x, fire_rate=None, angle=0.0):
        """Update the cellular automata state"""
        if fire_rate is None:
            fire_rate = self.fire_rate
            
        pre_life_mask = self.get_living_mask(x)
        
        y = self.perceive(x, angle)
        dx = self.perception(y)
        dx = F.relu(dx)
        dx = self.fc0(dx.permute(0, 2, 3, 1))
        dx = F.relu(dx)
        dx = self.fc1(dx)
        dx = dx.permute(0, 3, 1, 2)
        
        if fire_rate < 1.0:
            fire_mask = (torch.rand_like(x[:, :1, :, :]) <= fire_rate).float()
            dx = dx * fire_mask
            
        x = x + dx
        
        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        
        return x * life_mask
    
    def get_living_mask(self, x):
        """Determine which cells are alive based on alpha channel"""
        alpha = x[:, 3:4, :, :]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
    
    def forward(self, x, steps=64, fire_rate=None, angle=0.0):
        """Run the CA for specified number of steps"""
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x


class NCATrainer:
    """Handles training of the Neural Cellular Automata"""
    
    def __init__(self, model, target_image, device='cpu'):
        self.model = model
        self.target_image = target_image
        self.device = torch.device(device) if isinstance(device, str) else device
        self.optimizer = optim.Adam(model.parameters(), lr=2e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9999)
        
        # Training state
        self.current_epoch = 0
        self.losses = []
        self.is_training = False
        
    def create_seed(self, size, random_init=True):
        """Create initial seed state with random or center initialization"""
        # Create seed tensor directly on the correct device
        seed = torch.zeros(1, self.model.channel_n, size, size, device=self.device)
        
        if random_init:
            # Random initialization for training robustness
            init_type = torch.randint(0, 4, (1,)).item()
            
            if init_type == 0:
                # Single random center point
                center_x = int(torch.randint(size//4, 3*size//4, (1,)).item())
                center_y = int(torch.randint(size//4, 3*size//4, (1,)).item())
                seed[:, 3:, center_x, center_y] = 1.0
                
            elif init_type == 1:
                # Multiple random seed points (2-5 points)
                num_seeds = int(torch.randint(2, 6, (1,)).item())
                for _ in range(num_seeds):
                    x = int(torch.randint(0, size, (1,)).item())
                    y = int(torch.randint(0, size, (1,)).item())
                    # Create random values on the correct device
                    rand_values = torch.rand(self.model.channel_n-3, device=self.device) * 0.8 + 0.2
                    seed[:, 3:, x, y] = rand_values
                    
            elif init_type == 2:
                # Sparse random pixels (0.5-2% density)
                density = torch.rand(1).item() * 0.015 + 0.005
                mask = torch.rand(size, size, device=self.device) < density
                num_pixels = int(mask.sum().item())
                if num_pixels > 0:
                    rand_values = torch.rand(self.model.channel_n-3, num_pixels, device=self.device) * 0.7 + 0.3
                    seed[:, 3:, mask] = rand_values
                
            else:
                # Edge/corner initialization
                if torch.rand(1).item() < 0.5:
                    # Random edge
                    edge = torch.randint(0, 4, (1,)).item()
                    if edge == 0:  # Top
                        seed[:, 3:, 0, size//4:3*size//4] = 1.0
                    elif edge == 1:  # Bottom
                        seed[:, 3:, -1, size//4:3*size//4] = 1.0
                    elif edge == 2:  # Left
                        seed[:, 3:, size//4:3*size//4, 0] = 1.0
                    else:  # Right
                        seed[:, 3:, size//4:3*size//4, -1] = 1.0
                else:
                    # Random corner
                    corner_idx = int(torch.randint(0, 4, (1,)).item())
                    corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
                    x_idx, y_idx = corners[corner_idx]
                    if x_idx == -1:
                        x_slice = slice(size-3, size)
                    else:
                        x_slice = slice(0, 3)
                    if y_idx == -1:
                        y_slice = slice(size-3, size)
                    else:
                        y_slice = slice(0, 3)
                    seed[:, 3:, x_slice, y_slice] = 1.0
        else:
            # Traditional center initialization for manual generation
            seed[:, 3:, size//2, size//2] = 1.0
            
        return seed
    
    def train_step(self, steps_range=(64, 96)):
        """Perform one training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Random number of steps
        steps = np.random.randint(*steps_range)
        
        # Create seed with random initialization for robust training
        size = self.target_image.shape[-1]
        x = self.create_seed(size, random_init=True)
        
        # Run CA
        x = self.model(x, steps=steps)
        
        # Calculate loss (only on RGB channels)
        loss = F.mse_loss(x[:, :4], self.target_image)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), x
    
    def train_epoch(self, steps_per_epoch=100, steps_range=(64, 96)):
        """Train for one epoch"""
        total_loss = 0
        latest_output = None
        
        for step in range(steps_per_epoch):
            if not self.is_training:
                break
                
            loss, output = self.train_step(steps_range)
            total_loss += loss
            latest_output = output
            
        self.current_epoch += 1
        avg_loss = total_loss / steps_per_epoch
        self.losses.append(avg_loss)
        
        return avg_loss, latest_output


class NCAGUI:
    """Main GUI application for Neural Cellular Automata"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Cellular Automata - Image Recreation")
        self.root.geometry("1800x1000")  # Larger window for 1920x1080 displays
        self.root.state('zoomed')  # Start maximized on Windows
        
        # Initialize variables
        self.model = None
        self.trainer = None
        self.target_image = None
        self.current_output = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_thread = None
        self.auto_generate_enabled = False
        self.last_auto_generate = 0
        self.alpha_colorbar = None  # Track colorbar for alpha channel
        
        # Setup GUI
        self.setup_gui()
        
        # Start update loop for real-time feedback
        self.update_display_loop()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Visualization panel
        self.setup_visualization_panel(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup control panel with buttons and parameters"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="new", padx=(0, 10))
        
        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(file_frame, text="Save Model", command=self.save_model).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(file_frame, text="Load Model", command=self.load_model).grid(row=0, column=2)
        
        # Training parameters with optimal settings info
        params_frame = ttk.LabelFrame(control_frame, text="Parameters (Optimal for 64x64: Ch=12-16, Fire=0.6-0.8, Steps=32-64)", padding="5")
        params_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Channel count
        ttk.Label(params_frame, text="Channels:").grid(row=0, column=0, sticky=tk.W)
        self.channels_var = tk.StringVar(value="16")
        channels_spin = ttk.Spinbox(params_frame, from_=8, to=32, textvariable=self.channels_var, width=10)
        channels_spin.grid(row=0, column=1, padx=(5, 0))
        
        # Fire rate
        ttk.Label(params_frame, text="Fire Rate:").grid(row=1, column=0, sticky=tk.W)
        self.fire_rate_var = tk.StringVar(value="0.5")
        fire_rate_spin = ttk.Spinbox(params_frame, from_=0.1, to=1.0, increment=0.1, 
                                   textvariable=self.fire_rate_var, width=10)
        fire_rate_spin.grid(row=1, column=1, padx=(5, 0))
        
        # Training steps range
        ttk.Label(params_frame, text="Min Steps:").grid(row=2, column=0, sticky=tk.W)
        self.min_steps_var = tk.StringVar(value="64")
        min_steps_spin = ttk.Spinbox(params_frame, from_=16, to=128, increment=8,
                                   textvariable=self.min_steps_var, width=10)
        min_steps_spin.grid(row=2, column=1, padx=(5, 0))
        
        ttk.Label(params_frame, text="Max Steps:").grid(row=3, column=0, sticky=tk.W)
        self.max_steps_var = tk.StringVar(value="96")
        max_steps_spin = ttk.Spinbox(params_frame, from_=32, to=256, increment=8,
                                   textvariable=self.max_steps_var, width=10)
        max_steps_spin.grid(row=3, column=1, padx=(5, 0))
        
        # Add preset buttons for quick configuration
        preset_frame = ttk.Frame(params_frame)
        preset_frame.grid(row=4, column=0, columnspan=2, pady=(5, 0), sticky="ew")
        
        ttk.Button(preset_frame, text="64x64 Preset", command=self.apply_64x64_preset, width=12).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(preset_frame, text="High Quality", command=self.apply_hq_preset, width=12).grid(row=0, column=1)
        
        # Training controls
        train_frame = ttk.LabelFrame(control_frame, text="Training", padding="5")
        train_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        self.init_button = ttk.Button(train_frame, text="Initialize Model", command=self.initialize_model)
        self.init_button.grid(row=0, column=0, pady=(0, 5), sticky="ew")
        
        self.train_button = ttk.Button(train_frame, text="Start Training", command=self.toggle_training)
        self.train_button.grid(row=1, column=0, pady=(0, 5), sticky="ew")
        
        ttk.Button(train_frame, text="Generate Now", command=self.generate_image).grid(row=2, column=0, pady=(0, 5), sticky="ew")
        
        ttk.Button(train_frame, text="Test Random Init", command=self.generate_random_image).grid(row=3, column=0, pady=(0, 5), sticky="ew")
        
        # Auto-generate checkbox
        self.auto_generate_var = tk.BooleanVar()
        auto_check = ttk.Checkbutton(train_frame, text="Auto-generate every 8s", 
                                   variable=self.auto_generate_var,
                                   command=self.toggle_auto_generate)
        auto_check.grid(row=4, column=0, pady=(5, 0), sticky="w")
        
        train_frame.columnconfigure(0, weight=1)
        
        # Enhanced Progress Display
        progress_frame = ttk.LabelFrame(control_frame, text="Training Progress", padding="5")
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        
        # Status indicator
        status_subframe = ttk.Frame(progress_frame)
        status_subframe.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        
        ttk.Label(status_subframe, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.model_status_label = ttk.Label(status_subframe, text="No model", foreground="red")
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Training metrics
        ttk.Label(progress_frame, text="Epoch:").grid(row=1, column=0, sticky=tk.W)
        self.epoch_label = ttk.Label(progress_frame, text="0", font=("TkDefaultFont", 10, "bold"))
        self.epoch_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(progress_frame, text="Loss:").grid(row=2, column=0, sticky=tk.W)
        self.loss_label = ttk.Label(progress_frame, text="N/A", font=("TkDefaultFont", 9))
        self.loss_label.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(progress_frame, text="Training Rate:").grid(row=3, column=0, sticky=tk.W)
        self.rate_label = ttk.Label(progress_frame, text="N/A")
        self.rate_label.grid(row=3, column=1, sticky=tk.W, padx=(5, 0))
        
        # Add current steps display
        ttk.Label(progress_frame, text="Current Steps:").grid(row=4, column=0, sticky=tk.W)
        self.steps_label = ttk.Label(progress_frame, text="N/A")
        self.steps_label.grid(row=4, column=1, sticky=tk.W, padx=(5, 0))
        
        # Device info
        device_frame = ttk.Frame(control_frame)
        device_frame.grid(row=4, column=0, sticky="ew")
        
        ttk.Label(device_frame, text="Device:").grid(row=0, column=0, sticky=tk.W)
        device_info = f"{self.device.type.upper()}"
        if self.device.type == 'cuda':
            device_info += f" ({torch.cuda.get_device_name()})"
        self.device_label = ttk.Label(device_frame, text=device_info, foreground="blue")
        self.device_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
    def setup_visualization_panel(self, parent):
        """Setup visualization panel with matplotlib"""
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        viz_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        
        # Create larger matplotlib figure for better visibility
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle("Neural Cellular Automata Training", fontsize=16)
        
        # Setup subplots with larger fonts
        self.axes[0, 0].set_title("Target Image", fontsize=14)
        self.axes[0, 1].set_title("Current Output", fontsize=14)
        self.axes[1, 0].set_title("Training Loss", fontsize=14)
        self.axes[1, 1].set_title("Cell States (Alpha)", fontsize=14)
        
        # Initialize with placeholder content
        for i, ax in enumerate(self.axes.flat):
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 2:  # Loss plot
                ax.text(0.5, 0.5, 'No training data yet', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xlabel("Epoch", fontsize=12)
                ax.set_ylabel("Loss", fontsize=12)
                ax.grid(True, alpha=0.3)
            elif i == 3:  # Alpha channel
                ax.text(0.5, 0.5, 'No model output yet', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_status_bar(self, parent):
        """Setup status bar"""
        device_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        initial_status = f"Ready - Device: {device_info}"
        self.status_var = tk.StringVar(value=initial_status)
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
    def load_image(self):
        """Load target image"""
        file_path = filedialog.askopenfilename(
            title="Select target image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            try:
                # Load and process image
                img = Image.open(file_path).convert('RGBA')
                
                # Resize to maximum 256x256 pixels for better detail
                if max(img.size) > 256:
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                # Convert to tensor
                import torchvision.transforms.functional as TF
                img_tensor = TF.to_tensor(img).unsqueeze(0).to(self.device)
                self.target_image = img_tensor
                
                # Display in GUI
                self.axes[0, 0].clear()
                self.axes[0, 0].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy())
                self.axes[0, 0].set_title("Target Image")
                self.axes[0, 0].set_xticks([])
                self.axes[0, 0].set_yticks([])
                self.canvas.draw()
                
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def apply_64x64_preset(self):
        """Apply optimal settings for 64x64 images"""
        self.channels_var.set("14")
        self.fire_rate_var.set("0.7")
        self.min_steps_var.set("32")
        self.max_steps_var.set("64")
        self.status_var.set("Applied 64x64 optimal preset (Ch=14, Fire=0.7, Steps=32-64)")
        
    def apply_hq_preset(self):
        """Apply high quality settings for larger images"""
        self.channels_var.set("20")
        self.fire_rate_var.set("0.5")
        self.min_steps_var.set("64")
        self.max_steps_var.set("128")
        self.status_var.set("Applied high quality preset (Ch=20, Fire=0.5, Steps=64-128)")

    def initialize_model(self):
        """Initialize the NCA model"""
        if self.target_image is None:
            messagebox.showwarning("Warning", "Please load a target image first")
            return
            
        try:
            # Validate steps range
            min_steps = int(self.min_steps_var.get())
            max_steps = int(self.max_steps_var.get())
            
            if min_steps >= max_steps:
                messagebox.showerror("Error", "Min steps must be less than max steps")
                return
                
            self.init_button.config(state='disabled')
            self.model_status_label.config(text="Initializing...", foreground="orange")
            self.status_var.set("Initializing Neural Cellular Automata model...")
            self.root.update()
            
            channels = int(self.channels_var.get())
            fire_rate = float(self.fire_rate_var.get())
            
            self.model = NeuralCellularAutomata(channel_n=channels, fire_rate=fire_rate)
            self.model.to(self.device)
            
            self.trainer = NCATrainer(self.model, self.target_image, str(self.device))
            
            # Update UI to show model is ready
            self.model_status_label.config(text="Model Ready", foreground="green")
            self.init_button.config(state='normal')
            self.train_button.config(state='normal')
            
            # Initialize loss plot
            self.axes[1, 0].clear()
            self.axes[1, 0].set_title("Training Loss", fontsize=14)
            self.axes[1, 0].set_xlabel("Epoch", fontsize=12)
            self.axes[1, 0].set_ylabel("Loss", fontsize=12)
            self.axes[1, 0].grid(True, alpha=0.3)
            self.axes[1, 0].text(0.5, 0.5, 'Training not started', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.axes[1, 0].transAxes, fontsize=12)
            
            # Initialize alpha plot and reset colorbar
            self.axes[1, 1].clear()
            self.alpha_colorbar = None  # Reset colorbar tracking
            self.axes[1, 1].set_title("Cell States (Alpha)", fontsize=14)
            self.axes[1, 1].text(0.5, 0.5, 'No output yet', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=self.axes[1, 1].transAxes, fontsize=12)
            
            self.canvas.draw()
            self.status_var.set(f"Model initialized! Steps: {min_steps}-{max_steps}, Channels: {channels}, Fire Rate: {fire_rate}")
            
        except ValueError as e:
            self.model_status_label.config(text="Error", foreground="red")
            self.init_button.config(state='normal')
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")
            self.status_var.set("Model initialization failed")
        except Exception as e:
            self.model_status_label.config(text="Error", foreground="red")
            self.init_button.config(state='normal')
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")
            self.status_var.set("Model initialization failed")
            
    def toggle_training(self):
        """Start or stop training"""
        if self.trainer is None:
            messagebox.showwarning("Warning", "Please initialize model first")
            return
            
        if not self.trainer.is_training:
            # Start training - reset colorbar tracking
            self.alpha_colorbar = None
            self.trainer.is_training = True
            self.train_button.config(text="Stop Training")
            self.training_thread = threading.Thread(target=self.training_loop)
            self.training_thread.daemon = True
            self.training_thread.start()
            self.status_var.set("Training started...")
        else:
            # Stop training
            self.trainer.is_training = False
            self.train_button.config(text="Start Training")
            self.status_var.set("Training stopped")
            
    def toggle_auto_generate(self):
        """Toggle auto-generation every 8 seconds"""
        self.auto_generate_enabled = self.auto_generate_var.get()
        if self.auto_generate_enabled:
            self.last_auto_generate = time.time()
            self.status_var.set("Auto-generation enabled (every 8 seconds)")
        else:
            self.status_var.set("Auto-generation disabled")
    
    def update_display_loop(self):
        """Continuous update loop for real-time feedback"""
        try:
            # Auto-generate if enabled and enough time has passed
            if (self.auto_generate_enabled and self.model is not None and 
                time.time() - self.last_auto_generate >= 8.0):
                self.generate_image()
                self.last_auto_generate = time.time()
        except Exception:
            pass  # Ignore errors in display loop
        
        # Schedule next update
        self.root.after(500, self.update_display_loop)  # Update every 500ms
    
    def training_loop(self):
        """Main training loop (runs in separate thread)"""
        if self.trainer is None:
            return
        
        epoch_start_time = time.time()
        epochs_completed = 0
            
        while self.trainer.is_training:
            try:
                # Get current steps range from GUI
                min_steps = int(self.min_steps_var.get())
                max_steps = int(self.max_steps_var.get())
                steps_range = (min_steps, max_steps)
                
                loss, output = self.trainer.train_epoch(steps_per_epoch=25, steps_range=steps_range)
                self.current_output = output
                
                epochs_completed += 1
                elapsed_time = time.time() - epoch_start_time
                
                if elapsed_time > 0:
                    epochs_per_sec = epochs_completed / elapsed_time
                else:
                    epochs_per_sec = 0
                
                # Update GUI in main thread
                self.root.after(0, self.update_training_display, loss, epochs_per_sec)
                
                time.sleep(0.05)  # Shorter delay for more responsive updates
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
                break
                
        self.root.after(0, lambda: self.train_button.config(text="Start Training"))
        
    def update_training_display(self, loss, epochs_per_sec=None):
        """Update training display (called from main thread)"""
        if self.trainer is None:
            return
            
        # Update labels with enhanced formatting
        self.epoch_label.config(text=str(self.trainer.current_epoch))
        self.loss_label.config(text=f"{loss:.6f}")
        
        # Update steps range display
        try:
            min_steps = self.min_steps_var.get()
            max_steps = self.max_steps_var.get()
            self.steps_label.config(text=f"{min_steps}-{max_steps}")
        except:
            self.steps_label.config(text="N/A")
        
        if epochs_per_sec is not None:
            self.rate_label.config(text=f"{epochs_per_sec:.2f} epochs/sec")
        
        # Update plots with current output
        if self.current_output is not None:
            # Current output (RGB)
            self.axes[0, 1].clear()
            output_rgb = self.current_output[0, :3].permute(1, 2, 0).detach().cpu().numpy()
            output_rgb = np.clip(output_rgb, 0, 1)
            self.axes[0, 1].imshow(output_rgb, interpolation='nearest')
            self.axes[0, 1].set_title(f"Current Output (Epoch {self.trainer.current_epoch})", fontsize=14)
            self.axes[0, 1].set_xticks([])
            self.axes[0, 1].set_yticks([])
            
            # Alpha channel with better visualization
            self.axes[1, 1].clear()
            alpha = self.current_output[0, 3].detach().cpu().numpy()
            im = self.axes[1, 1].imshow(alpha, cmap='viridis', interpolation='nearest')
            self.axes[1, 1].set_title(f"Cell States (Alpha) - Live: {np.sum(alpha > 0.1)}", fontsize=14)
            self.axes[1, 1].set_xticks([])
            self.axes[1, 1].set_yticks([])
            
            # Add colorbar for alpha channel only if not already present
            if self.alpha_colorbar is None:
                try:
                    self.alpha_colorbar = plt.colorbar(im, ax=self.axes[1, 1], shrink=0.8)
                except:
                    pass  # Ignore colorbar errors
            
        # Enhanced loss plot
        if self.trainer and len(self.trainer.losses) > 1:
            self.axes[1, 0].clear()
            losses = self.trainer.losses
            epochs = range(1, len(losses) + 1)
            
            self.axes[1, 0].plot(epochs, losses, 'b-', linewidth=2, alpha=0.8)
            self.axes[1, 0].set_title(f"Training Loss (Current: {loss:.6f})", fontsize=14)
            self.axes[1, 0].set_xlabel("Epoch", fontsize=12)
            self.axes[1, 0].set_ylabel("Loss", fontsize=12)
            self.axes[1, 0].grid(True, alpha=0.3)
            
            # Add trend line for recent epochs
            if len(losses) > 10:
                recent_losses = losses[-10:]
                recent_epochs = list(range(len(losses) - 9, len(losses) + 1))
                z = np.polyfit(recent_epochs, recent_losses, 1)
                p = np.poly1d(z)
                self.axes[1, 0].plot(recent_epochs, p(recent_epochs), "r--", alpha=0.6, label="Trend")
                self.axes[1, 0].legend(fontsize=10)
            
            # Set reasonable y-axis limits
            if len(losses) > 1:
                min_loss = min(losses)
                max_loss = max(losses)
                margin = (max_loss - min_loss) * 0.1
                self.axes[1, 0].set_ylim(max(0, min_loss - margin), max_loss + margin)
        
        # Adjust layout and refresh
        try:
            plt.tight_layout()
        except:
            pass
            
        self.canvas.draw()
        
    def generate_image(self):
        """Generate image from current model"""
        if self.model is None:
            if not self.auto_generate_enabled:  # Only show warning if not auto-generating
                messagebox.showwarning("Warning", "Please initialize model first")
            return
            
        if self.target_image is None:
            if not self.auto_generate_enabled:  # Only show warning if not auto-generating
                messagebox.showwarning("Warning", "Please load target image first")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                size = self.target_image.shape[-1]
                # Use deterministic center initialization for manual generation
                if self.trainer:
                    seed = self.trainer.create_seed(size, random_init=False)
                else:
                    # Fallback if no trainer
                    seed = torch.zeros(1, self.model.channel_n, size, size)
                    seed[:, 3:, size//2, size//2] = 1.0
                    seed = seed.to(self.device)
                
                # Use configured max steps for generation (higher quality)
                try:
                    max_steps = int(self.max_steps_var.get())
                    generation_steps = min(max_steps * 2, 256)  # Use 2x max training steps, capped at 256
                except:
                    generation_steps = 128  # Fallback
                
                output = self.model(seed, steps=generation_steps)
                
                # Display result with enhanced visualization
                self.axes[0, 1].clear()
                output_rgb = output[0, :3].permute(1, 2, 0).detach().cpu().numpy()
                output_rgb = np.clip(output_rgb, 0, 1)
                self.axes[0, 1].imshow(output_rgb, interpolation='nearest')
                
                # Calculate similarity to target
                if self.target_image is not None:
                    target_rgb = self.target_image[0, :3].permute(1, 2, 0).cpu().numpy()
                    similarity = 1.0 - np.mean((output_rgb - target_rgb) ** 2)
                    self.axes[0, 1].set_title(f"Generated Image (Similarity: {similarity:.3f})", fontsize=14)
                else:
                    self.axes[0, 1].set_title("Generated Image", fontsize=14)
                    
                self.axes[0, 1].set_xticks([])
                self.axes[0, 1].set_yticks([])
                
                # Update alpha visualization
                self.axes[1, 1].clear()
                alpha = output[0, 3].detach().cpu().numpy()
                im = self.axes[1, 1].imshow(alpha, cmap='viridis', interpolation='nearest')
                live_cells = np.sum(alpha > 0.1)
                self.axes[1, 1].set_title(f"Generated Alpha ({live_cells} live cells)", fontsize=14)
                self.axes[1, 1].set_xticks([])
                self.axes[1, 1].set_yticks([])
                
                self.canvas.draw()
                
                if not self.auto_generate_enabled:
                    self.status_var.set(f"Image generated with {generation_steps} steps")
                
        except Exception as e:
            if not self.auto_generate_enabled:  # Only show error if not auto-generating
                messagebox.showerror("Error", f"Failed to generate image: {str(e)}")
                
    def generate_random_image(self):
        """Generate image from random initialization to test robustness"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please initialize model first")
            return
            
        if self.target_image is None:
            messagebox.showwarning("Warning", "Please load target image first")
            return
            
        if self.trainer is None:
            messagebox.showwarning("Warning", "Please initialize model first")
            return
            
        try:
            self.model.eval()
            with torch.no_grad():
                size = self.target_image.shape[-1]
                # Use random initialization to test robustness
                seed = self.trainer.create_seed(size, random_init=True)
                
                # Use configured max steps for generation (higher quality)
                try:
                    max_steps = int(self.max_steps_var.get())
                    generation_steps = min(max_steps * 2, 256)  # Use 2x max training steps, capped at 256
                except:
                    generation_steps = 128  # Fallback
                
                output = self.model(seed, steps=generation_steps)
                
                # Display result with enhanced visualization
                self.axes[0, 1].clear()
                output_rgb = output[0, :3].permute(1, 2, 0).detach().cpu().numpy()
                output_rgb = np.clip(output_rgb, 0, 1)
                self.axes[0, 1].imshow(output_rgb, interpolation='nearest')
                
                # Calculate similarity to target
                if self.target_image is not None:
                    target_rgb = self.target_image[0, :3].permute(1, 2, 0).cpu().numpy()
                    similarity = 1.0 - np.mean((output_rgb - target_rgb) ** 2)
                    self.axes[0, 1].set_title(f"Random Init Result (Similarity: {similarity:.3f})", fontsize=14)
                else:
                    self.axes[0, 1].set_title("Random Init Result", fontsize=14)
                    
                self.axes[0, 1].set_xticks([])
                self.axes[0, 1].set_yticks([])
                
                # Update alpha visualization
                self.axes[1, 1].clear()
                alpha = output[0, 3].detach().cpu().numpy()
                im = self.axes[1, 1].imshow(alpha, cmap='viridis', interpolation='nearest')
                live_cells = np.sum(alpha > 0.1)
                self.axes[1, 1].set_title(f"Random Init Alpha ({live_cells} live cells)", fontsize=14)
                self.axes[1, 1].set_xticks([])
                self.axes[1, 1].set_yticks([])
                
                self.canvas.draw()
                
                self.status_var.set(f"Random init image generated with {generation_steps} steps")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate random image: {str(e)}")
            
    def save_model(self):
        """Save trained model"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pth",
            filetypes=[("PyTorch models", "*.pth")]
        )
        
        if file_path:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'channel_n': self.model.channel_n,
                    'fire_rate': self.model.fire_rate,
                    'epoch': self.trainer.current_epoch if self.trainer else 0,
                    'losses': self.trainer.losses if self.trainer else []
                }, file_path)
                
                self.status_var.set(f"Model saved: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                
    def load_model(self):
        """Load trained model"""
        file_path = filedialog.askopenfilename(
            title="Load model",
            filetypes=[("PyTorch models", "*.pth")]
        )
        
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
                
                # Create model with saved parameters
                self.model = NeuralCellularAutomata(
                    channel_n=checkpoint['channel_n'],
                    fire_rate=checkpoint['fire_rate']
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                
                # Update GUI
                self.channels_var.set(str(checkpoint['channel_n']))
                self.fire_rate_var.set(str(checkpoint['fire_rate']))
                
                if self.target_image is not None:
                    self.trainer = NCATrainer(self.model, self.target_image, str(self.device))
                    self.trainer.current_epoch = checkpoint.get('epoch', 0)
                    self.trainer.losses = checkpoint.get('losses', [])
                    
                self.status_var.set(f"Model loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = NCAGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.trainer and app.trainer.is_training:
            app.trainer.is_training = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
