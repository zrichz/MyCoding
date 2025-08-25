#!/usr/bin/env python3
"""
Falling Blocks Image Viewer - Interactive Image Display Tool

This script provides an engaging way to view images using a "falling blocks" effect.
Images are divided into 32x32 pixel blocks that fall into place as greyscale,
then slowly transition to their original colors.

Key Features:
- Select directory containing images for sequential viewing
- 32x32 pixel block grid system
- Slow, realistic falling animation with physics simulation
- Greyscale to color transition effect after blocks settle
- Keyboard controls for navigation
- Automatic progression through image directory

Controls:
- SPACE: Start/pause falling animation
- RIGHT ARROW / N: Next image
- LEFT ARROW / P: Previous image
- R: Restart current image animation
- ESC / Q: Quit application

Technical Implementation:
- Uses pygame for graphics and animation
- PIL for image loading and processing
- Block-based rendering system with smooth color transitions
- Realistic physics with gravity, bounce, and settling detection
- Configurable animation speeds and effects
"""

import pygame
import sys
import os
from PIL import Image
import random
import math
from tkinter import filedialog
import tkinter as tk

class FallingBlock:
    def __init__(self, x, y, grid_x, grid_y, gray_surface, color_surface, block_size=32):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = -block_size - random.randint(0, 200)  # Start above screen
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.target_x = x
        self.target_y = y
        self.gray_surface = gray_surface
        self.color_surface = color_surface
        self.block_size = block_size
        
        # Physics (10x slower falling)
        self.velocity_y = 0
        self.gravity = 0.05  # Reduced from 0.5 to 0.05 (10x slower)
        self.bounce = 0.3
        self.friction = 0.95
        
        # Animation state
        self.has_landed = False
        self.color_transition = 0.0  # 0.0 = full gray, 1.0 = full color
        self.transition_speed = 0.01  # Slower color transition
        self.settle_time = 0
        self.settled = False  # Track if block has stopped bouncing
        
    def update(self):
        if not self.has_landed:
            # Apply gravity
            self.velocity_y += self.gravity
            self.y += self.velocity_y
            
            # Check if landed
            if self.y >= self.target_y:
                self.y = self.target_y
                self.has_landed = True
                # Small bounce effect
                self.velocity_y = -self.velocity_y * self.bounce
                
                # If bounce is very small, stop bouncing and start settling
                if abs(self.velocity_y) < 0.2:  # Lower threshold for settling
                    self.velocity_y = 0
                    self.settled = True
                    self.settle_time = pygame.time.get_ticks()
        
        # Color transition after settling (or immediately if no bounce)
        if self.settled and self.settle_time > 0:
            # Wait a bit before starting color transition
            if pygame.time.get_ticks() - self.settle_time > 100:
                if self.color_transition < 1.0:
                    self.color_transition = min(1.0, self.color_transition + self.transition_speed)
    
    def get_current_surface(self):
        """Get the current surface based on color transition state"""
        if self.color_transition <= 0:
            return self.gray_surface
        elif self.color_transition >= 1:
            return self.color_surface
        else:
            # Create a new surface for blending
            blended = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
            blended.fill((0, 0, 0, 0))  # Transparent background
            
            # Create surfaces with proper alpha
            gray_surf = self.gray_surface.copy()
            color_surf = self.color_surface.copy()
            
            # Set alpha for blending
            gray_alpha = int(255 * (1 - self.color_transition))
            color_alpha = int(255 * self.color_transition)
            
            # First blit the grayscale with its alpha
            if gray_alpha > 0:
                gray_surf.set_alpha(gray_alpha)
                blended.blit(gray_surf, (0, 0))
            
            # Then blit the color on top with its alpha
            if color_alpha > 0:
                color_surf.set_alpha(color_alpha)
                blended.blit(color_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
            
            return blended
    
    def draw(self, screen):
        current_surface = self.get_current_surface()
        screen.blit(current_surface, (self.x, self.y))

class FallingBlocksViewer:
    def __init__(self):
        pygame.init()
        
        # Screen settings
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Falling Blocks Image Viewer")
        
        # Block settings
        self.block_size = 32
        
        # Colors
        self.bg_color = (20, 20, 30)
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
        
        # Timing
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # UI state
        self.showing_instructions = True
        
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
            
            # Calculate scaling to fit screen while maintaining aspect ratio
            img_width, img_height = pil_image.size
            scale_x = (self.screen_width - 100) / img_width
            scale_y = (self.screen_height - 150) / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
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
        """Create falling blocks from the current image"""
        if not self.current_image:
            return
            
        self.blocks = []
        img_width, img_height = self.current_image.size
        
        # Calculate grid dimensions
        cols = math.ceil(img_width / self.block_size)
        rows = math.ceil(img_height / self.block_size)
        
        for row in range(rows):
            for col in range(cols):
                # Calculate block position
                block_x = col * self.block_size
                block_y = row * self.block_size
                
                # Extract block from image
                left = col * self.block_size
                top = row * self.block_size
                right = min(left + self.block_size, img_width)
                bottom = min(top + self.block_size, img_height)
                
                if right > left and bottom > top:
                    # Extract color block
                    color_block = self.current_image.crop((left, top, right, bottom))
                    
                    # Create grayscale version
                    gray_block = color_block.convert('L').convert('RGB')
                    
                    # Convert to pygame surfaces
                    color_string = color_block.tobytes()
                    gray_string = gray_block.tobytes()
                    
                    color_surface = pygame.image.fromstring(color_string, color_block.size, 'RGB')
                    gray_surface = pygame.image.fromstring(gray_string, gray_block.size, 'RGB')
                    
                    # If block is smaller than block_size, pad it
                    if color_block.size != (self.block_size, self.block_size):
                        padded_color = pygame.Surface((self.block_size, self.block_size))
                        padded_gray = pygame.Surface((self.block_size, self.block_size))
                        padded_color.fill((0, 0, 0))
                        padded_gray.fill((0, 0, 0))
                        padded_color.blit(color_surface, (0, 0))
                        padded_gray.blit(gray_surface, (0, 0))
                        color_surface = padded_color
                        gray_surface = padded_gray
                    
                    # Create falling block
                    screen_x = self.image_x + block_x
                    screen_y = self.image_y + block_y
                    
                    block = FallingBlock(
                        screen_x, screen_y, col, row,
                        gray_surface, color_surface, self.block_size
                    )
                    
                    self.blocks.append(block)
        
        # Shuffle blocks for random falling order
        random.shuffle(self.blocks)
        
        # Stagger the start times
        for i, block in enumerate(self.blocks):
            block.y -= i * 2  # Spread them out vertically
        
        self.animation_running = False
        self.animation_complete = False
    
    def start_animation(self):
        """Start the falling blocks animation"""
        if self.blocks:
            self.animation_running = True
            self.animation_complete = False
            
            # Reset all blocks
            for i, block in enumerate(self.blocks):
                block.y = -self.block_size - random.randint(0, 200) - i * 2
                block.velocity_y = 0
                block.has_landed = False
                block.settled = False
                block.color_transition = 0.0
                block.settle_time = 0
    
    def update_animation(self):
        """Update the falling blocks animation"""
        if not self.animation_running:
            return
            
        all_settled = True
        
        for block in self.blocks:
            block.update()
            
            if not block.settled or block.color_transition < 1.0:
                all_settled = False
        
        if all_settled:
            self.animation_complete = True
    
    def next_image(self):
        """Move to next image"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
    
    def previous_image(self):
        """Move to previous image"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def draw_instructions(self):
        """Draw instruction screen"""
        self.screen.fill(self.bg_color)
        
        instructions = [
            "Falling Blocks Image Viewer",
            "",
            "Controls:",
            "SPACE - Start/pause falling animation",
            "RIGHT ARROW / N - Next image", 
            "LEFT ARROW / P - Previous image",
            "R - Restart current image animation",
            "ESC / Q - Quit",
            "",
            "Press any key to select image directory..."
        ]
        
        y_offset = 150
        for line in instructions:
            if line == "Falling Blocks Image Viewer":
                text = self.font.render(line, True, self.text_color)
            else:
                text = self.small_font.render(line, True, self.text_color)
            
            text_rect = text.get_rect(center=(self.screen_width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 40 if line == "Falling Blocks Image Viewer" else 30
    
    def draw_ui(self):
        """Draw UI elements"""
        if not self.image_files:
            return
            
        # Current image info
        filename = self.image_files[self.current_image_index]
        info_text = f"{self.current_image_index + 1}/{len(self.image_files)}: {filename}"
        text = self.small_font.render(info_text, True, self.text_color)
        self.screen.blit(text, (10, 10))
        
        # Status
        if self.animation_running:
            status = "Animation running..."
        elif self.animation_complete:
            status = "Animation complete - Press SPACE to restart"
        else:
            status = "Press SPACE to start animation"
            
        status_text = self.small_font.render(status, True, self.text_color)
        self.screen.blit(status_text, (10, self.screen_height - 30))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if self.showing_instructions:
                    self.showing_instructions = False
                    if not self.select_directory():
                        return False
                        
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                    
                elif event.key == pygame.K_SPACE:
                    if self.blocks:
                        if not self.animation_running or self.animation_complete:
                            self.start_animation()
                        else:
                            self.animation_running = not self.animation_running
                            
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_n:
                    self.next_image()
                    
                elif event.key == pygame.K_LEFT or event.key == pygame.K_p:
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
        running = True
        
        while running:
            running = self.handle_events()
            
            if not self.showing_instructions:
                self.update_animation()
            
            self.draw()
            self.clock.tick(self.fps)
        
        pygame.quit()

if __name__ == "__main__":
    viewer = FallingBlocksViewer()
    viewer.run()
