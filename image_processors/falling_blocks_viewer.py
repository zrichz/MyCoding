"""
Progressive Image Viewer - Interactive Image Display Tool

This script provides an engaging way to view images using a progressive reveal effect.
Images are divided into a 6x6 grid of blocks that fade in from black over random intervals
between 2-5 seconds, followed by gaussian blur removal to reveal the sharp image.

Key Features:
- Automatic directory selection dialog at startup
- Progressive reveal through 16x32 grid system (512 blocks total, regardless of image size)
- Blocks fade in from black over random 5-15 second intervals
- Gaussian blur effect that removes after fade-in completes
- Keyboard controls for navigation with auto-starting animations
- Arrow keys can skip current animation to move immediately to next/previous image
- Automatic progression through image directory

Controls:
- SPACE: Start/pause reveal animation
- RIGHT ARROW / N: Next image (skips current animation if running, auto-starts new animation)
- LEFT ARROW / P: Previous image (skips current animation if running, auto-starts new animation)
- R: Restart current image animation
- ESC / Q: Quit application

Technical Implementation:
- Uses pygame for graphics and animation
- PIL for image loading and Gaussian blur processing
- Block-based rendering system with fade-in effects
- Random timing for organic reveal progression
- Configurable animation speeds and blur effects
"""

import pygame
import sys
import os
from PIL import Image, ImageFilter
import random
import math
from tkinter import filedialog
import tkinter as tk

class ProgressiveBlock:
    def __init__(self, x, y, grid_x, grid_y, color_surface, block_width, block_height):
        self.x = x
        self.y = y
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.color_surface = color_surface
        self.block_width = block_width
        self.block_height = block_height

        # Fade-in timing (5-15 seconds)
        self.fade_duration = random.uniform(5000, 15000)  # milliseconds
        self.fade_start_time = 0
        self.fade_delay = random.uniform(0, 5000)  # Random delay before starting
        
        # Animation state
        self.fade_started = False
        self.fade_complete = False
        self.blur_removal_started = False
        self.blur_removal_complete = False
        
        # Fade and blur effects
        self.alpha = 0.0  # 0.0 = black, 1.0 = full color
        self.blur_amount = 1.0  # 1.0 = full blur, 0.0 = no blur
        self.blur_fade_speed = 0.003  # How fast blur fades away
        
    def start_animation(self, start_time):
        """Start the fade-in animation at the given time"""
        self.fade_start_time = start_time + self.fade_delay
        self.fade_started = False
        self.fade_complete = False
        self.blur_removal_started = False
        self.blur_removal_complete = False
        self.alpha = 0.0
        self.blur_amount = 1.0
        
    def update(self, current_time):
        """Update the block's animation state"""
        # Check if fade-in should start
        if not self.fade_started and current_time >= self.fade_start_time:
            self.fade_started = True
        
        # Update fade-in progress
        if self.fade_started and not self.fade_complete:
            elapsed = current_time - self.fade_start_time
            progress = min(1.0, elapsed / self.fade_duration)
            self.alpha = progress
            
            if progress >= 1.0:
                self.fade_complete = True
                self.blur_removal_started = True
        
        # Update blur removal after fade-in completes
        if self.blur_removal_started and not self.blur_removal_complete:
            if self.blur_amount > 0.0:
                self.blur_amount = max(0.0, self.blur_amount - self.blur_fade_speed)
            else:
                self.blur_removal_complete = True
    
    def get_current_surface(self):
        """Get the current surface with fade and blur effects"""
        if self.alpha <= 0:
            # Return black surface
            black_surface = pygame.Surface((self.block_width, self.block_height))
            black_surface.fill((0, 0, 0))
            return black_surface
        
        # Get the base surface (with blur if needed)
        if self.blur_amount <= 0:
            current_surface = self.color_surface.copy()
        else:
            # Apply gaussian blur
            w, h = self.color_surface.get_size()
            raw = pygame.image.tostring(self.color_surface, 'RGB')
            pil_image = Image.frombytes('RGB', (w, h), raw)
            
            # Apply gaussian blur
            blur_radius = self.blur_amount * 30.0  # Scale blur amount
            if blur_radius > 0:
                blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            else:
                blurred_image = pil_image
            
            # Convert back to pygame surface
            raw = blurred_image.tobytes()
            current_surface = pygame.image.fromstring(raw, (w, h), 'RGB')
        
        # Apply alpha blending with black
        if self.alpha < 1.0:
            # Create black surface
            black_surface = pygame.Surface((self.block_width, self.block_height))
            black_surface.fill((0, 0, 0))
            
            # Blend current surface with black based on alpha
            fade_surface = pygame.Surface((self.block_width, self.block_height))
            fade_surface.fill((0, 0, 0))
            
            # Scale and blend
            current_surface.set_alpha(int(self.alpha * 255))
            fade_surface.blit(black_surface, (0, 0))
            fade_surface.blit(current_surface, (0, 0))
            
            return fade_surface
        
        return current_surface
    
    def draw(self, screen):
        current_surface = self.get_current_surface()
        screen.blit(current_surface, (self.x, self.y))

class ProgressiveImageViewer:
    def __init__(self):
        pygame.init()
        
        # Screen settings
        self.screen_width = 2560
        self.screen_height = 1440
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Progressive Image Viewer")
        
        # Grid settings - eg: 6x6
        self.grid_cols = 16
        self.grid_rows = 32
        
        # Colors
        self.bg_color = (128, 128, 128)  # Mid-grey
        self.text_color = (255, 255, 255)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Image management
        self.image_directory = None
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        self.image_surface = None
        
        # Animation
        self.blocks = []
        self.animation_running = False
        self.animation_complete = False
        self.animation_start_time = 0
        
        # Timing
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # UI state
        self.showing_instructions = False
        self.directory_selected = False
        
    def select_directory(self):
        """Select directory containing images using file dialog"""
        # Hide pygame window temporarily
        pygame.display.iconify()
        
        # Create tkinter root and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        directory = filedialog.askdirectory(
            title="Select Directory Containing Images",
            parent=root
        )
        
        root.destroy()
        
        # Restore pygame window
        pygame.display.set_mode((self.screen_width, self.screen_height))
        
        if directory:
            self.image_directory = directory
            self.load_image_list()
            return True
        return False
    
    def load_image_list(self):
        """Load list of supported image files from directory"""
        if not self.image_directory:
            return
            
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = []
        
        for filename in os.listdir(self.image_directory):
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                self.image_files.append(filename)
        
        self.image_files.sort()
        self.current_image_index = 0
        
        if self.image_files:
            self.load_current_image()
            # Automatically start animation for the first image
            if self.blocks:
                self.start_animation()
    
    def load_current_image(self):
        """Load and prepare the current image"""
        if not self.image_files:
            return
            
        filename = self.image_files[self.current_image_index]
        filepath = os.path.join(self.image_directory, filename)
        
        try:
            # Load image with PIL
            pil_image = Image.open(filepath)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Hybrid scaling approach: use full screen, only downscale if necessary
            img_width, img_height = pil_image.size
            scale_x = self.screen_width / img_width
            scale_y = self.screen_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale small images
            
            # Only resize if image is larger than screen
            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                # Use original size for smaller images
                new_width = img_width
                new_height = img_height
            
            # Convert to pygame surface
            image_string = pil_image.tobytes()
            self.image_surface = pygame.image.fromstring(image_string, pil_image.size, 'RGB')
            
            # Calculate position to center image
            self.image_x = (self.screen_width - new_width) // 2
            self.image_y = (self.screen_height - new_height) // 2
            
            self.current_image = pil_image
            self.create_blocks()
            
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    def create_blocks(self):
        """Create progressive blocks from the current image in a 6x6 grid"""
        if not self.current_image:
            return
            
        self.blocks = []
        img_width, img_height = self.current_image.size
        
        # Calculate block dimensions for 6x6 grid
        block_width = img_width // self.grid_cols
        block_height = img_height // self.grid_rows
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate block position in image coordinates
                left = col * block_width
                top = row * block_height
                right = min(left + block_width, img_width)
                bottom = min(top + block_height, img_height)
                
                if right > left and bottom > top:
                    # Extract block from image
                    color_block = self.current_image.crop((left, top, right, bottom))
                    
                    # Convert to pygame surface
                    color_string = color_block.tobytes()
                    color_surface = pygame.image.fromstring(color_string, color_block.size, 'RGB')
                    
                    # Calculate screen position
                    screen_x = self.image_x + left
                    screen_y = self.image_y + top
                    
                    # Create progressive block
                    block = ProgressiveBlock(
                        screen_x, screen_y, col, row,
                        color_surface, color_block.size[0], color_block.size[1]
                    )
                    
                    self.blocks.append(block)
        
        self.animation_running = False
        self.animation_complete = False
    
    def start_animation(self):
        """Start the progressive reveal animation"""
        if self.blocks:
            self.animation_running = True
            self.animation_complete = False
            self.animation_start_time = pygame.time.get_ticks()
            
            # Start all blocks' animations
            for block in self.blocks:
                block.start_animation(self.animation_start_time)
    
    def update_animation(self):
        """Update the progressive reveal animation"""
        if not self.animation_running:
            return
            
        current_time = pygame.time.get_ticks()
        all_complete = True
        
        for block in self.blocks:
            block.update(current_time)
            
            if not (block.fade_complete and block.blur_removal_complete):
                all_complete = False
        
        if all_complete:
            self.animation_complete = True
    
    def skip_animation(self):
        """Skip the current animation by completing all blocks immediately"""
        if self.blocks and self.animation_running:
            for block in self.blocks:
                # Force complete the fade-in
                block.fade_started = True
                block.fade_complete = True 
                block.alpha = 1.0
                
                # Force complete the blur removal
                block.blur_removal_started = True
                block.blur_removal_complete = True
                block.blur_amount = 0.0
            
            self.animation_complete = True
            self.animation_running = False
    
    def next_image(self):
        """Move to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            # Automatically start the animation
            if self.blocks:
                self.start_animation()
    
    def previous_image(self):
        """Move to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            # Automatically start the animation
            if self.blocks:
                self.start_animation()
    
    def draw_instructions(self):
        """Draw instruction screen"""
        self.screen.fill(self.bg_color)
        
        instructions = [
            "Progressive Image Revealer",
            "",
            "Controls:",
            "SPACE - Start/pause",
            "ARROW KEYS  - Prev/Next image", 
            "R - Restart anim",
            "ESC / Q - Quit",
            "",
            "Press a key to select image directory..."
        ]
        
        y_offset = 120
        for line in instructions:
            if line == "Progressive Image Viewer":
                text = self.font.render(line, True, self.text_color)
            else:
                text = self.small_font.render(line, True, self.text_color)
            
            text_rect = text.get_rect(center=(self.screen_width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 40 if line == "Progressive Image Viewer" else 30
    
    def draw_ui(self):
        """Draw UI elements - only when not animating"""
        if not self.image_files:
            return
            
        # Don't show any text during animations
        if self.animation_running:
            return
            
        # Only show UI when animation is not running
        # Current image info
        filename = self.image_files[self.current_image_index]
        info_text = f"{self.current_image_index + 1}/{len(self.image_files)}: {filename}"
        text = self.small_font.render(info_text, True, self.text_color)
        self.screen.blit(text, (10, 10))
        
        # Status
        if self.animation_complete:
            status = "Reveal complete - Use arrows for next/prev or SPACE to restart"
        else:
            status = "Press SPACE to start reveal or arrows to navigate"
            
        status_text = self.small_font.render(status, True, self.text_color)
        self.screen.blit(status_text, (10, self.screen_height - 30))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                    
                elif event.key == pygame.K_SPACE:
                    if self.blocks:
                        if not self.animation_running or self.animation_complete:
                            self.start_animation()
                        else:
                            self.animation_running = not self.animation_running
                            
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_n:
                    # Skip current animation and move to next image immediately
                    if self.animation_running and not self.animation_complete:
                        self.skip_animation()
                    self.next_image()
                    
                elif event.key == pygame.K_LEFT or event.key == pygame.K_p:
                    # Skip current animation and move to previous image immediately
                    if self.animation_running and not self.animation_complete:
                        self.skip_animation()
                    self.previous_image()
                    
                elif event.key == pygame.K_r:
                    if self.blocks:
                        self.start_animation()
        
        return True
    
    def draw(self):
        """Main draw function"""
        self.screen.fill(self.bg_color)
        
        if self.showing_instructions:
            self.draw_instructions()
        else:
            # Draw blocks
            for block in self.blocks:
                block.draw(self.screen)
            
            # Draw UI
            self.draw_ui()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        # Automatically show directory selection at startup
        if not self.directory_selected:
            if not self.select_directory():
                pygame.quit()
                return
            self.directory_selected = True
        
        running = True
        
        while running:
            running = self.handle_events()
            
            if not self.showing_instructions:
                self.update_animation()
            
            self.draw()
            self.clock.tick(self.fps)
        
        pygame.quit()

if __name__ == "__main__":
    viewer = ProgressiveImageViewer()
    viewer.run()
