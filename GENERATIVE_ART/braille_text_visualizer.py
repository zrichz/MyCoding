"""
Braille Text Visualizer
Draws text using Braille letter patterns with customizable styling.
Each Braille character is represented as a 2x3 grid of dots.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Configuration: Change this text to display different Braille messages
BRAILLE_TEXT = "CAT DOG COW BIRD FISH"

class BrailleTextVisualizer:
    def __init__(self, width=1600, height=600, text=None):
        # Resolution
        self.width = width
        self.height = height
        self.text = (text if text is not None else BRAILLE_TEXT).upper()
        
        # Braille configuration
        self.dot_radius = 6  # Radius of each Braille dot (reduced from 8)
        self.cell_width = 30  # Width of each Braille cell (reduced from 40)
        self.cell_height = 60  # Height of each Braille cell (reduced from 80)
        self.letter_spacing = 8  # Space between letters (reduced from 10)
        self.word_spacing = 40  # Space between words (reduced from 60)
        self.line_spacing = 20  # Space between lines
        
        # Colors
        self.background_color = (0.05, 0.05, 0.15)  # Dark blue background
        self.dot_color = (1.0, 0.9, 0.2)  # Bright yellow dots
        
        # Braille patterns for letters A-Z
        self.braille_patterns = {
            'A': [1, 0, 0, 0, 0, 0],  # ⠁
            'B': [1, 1, 0, 0, 0, 0],  # ⠃
            'C': [1, 0, 0, 1, 0, 0],  # ⠉
            'D': [1, 0, 0, 1, 1, 0],  # ⠙
            'E': [1, 0, 0, 0, 1, 0],  # ⠑
            'F': [1, 1, 0, 1, 0, 0],  # ⠋
            'G': [1, 1, 0, 1, 1, 0],  # ⠛
            'H': [1, 1, 0, 0, 1, 0],  # ⠓
            'I': [0, 1, 0, 1, 0, 0],  # ⠊
            'J': [0, 1, 0, 1, 1, 0],  # ⠚
            'K': [1, 0, 1, 0, 0, 0],  # ⠅
            'L': [1, 1, 1, 0, 0, 0],  # ⠇
            'M': [1, 0, 1, 1, 0, 0],  # ⠍
            'N': [1, 0, 1, 1, 1, 0],  # ⠝
            'O': [1, 0, 1, 0, 1, 0],  # ⠕
            'P': [1, 1, 1, 1, 0, 0],  # ⠏
            'Q': [1, 1, 1, 1, 1, 0],  # ⠟
            'R': [1, 1, 1, 0, 1, 0],  # ⠗
            'S': [0, 1, 1, 1, 0, 0],  # ⠎
            'T': [0, 1, 1, 1, 1, 0],  # ⠞
            'U': [1, 0, 1, 0, 0, 1],  # ⠥
            'V': [1, 1, 1, 0, 0, 1],  # ⠧
            'W': [0, 1, 0, 1, 1, 1],  # ⠺
            'X': [1, 0, 1, 1, 0, 1],  # ⠭
            'Y': [1, 0, 1, 1, 1, 1],  # ⠽
            'Z': [1, 0, 1, 0, 1, 1],  # ⠵
            ' ': [0, 0, 0, 0, 0, 0],  # Space (no dots)
        }
        
        # Create the visualization
        self.create_image()
    
    def get_dot_positions(self):
        """Get the relative positions of the 6 dots in a Braille cell"""
        # Braille cell layout:
        # 1 4
        # 2 5  
        # 3 6
        dot_positions = [
            (0, 0),      # Dot 1 (top-left)
            (0, 1),      # Dot 2 (middle-left)
            (0, 2),      # Dot 3 (bottom-left)
            (1, 0),      # Dot 4 (top-right)
            (1, 1),      # Dot 5 (middle-right)
            (1, 2),      # Dot 6 (bottom-right)
        ]
        return dot_positions
    
    def calculate_text_dimensions(self):
        """Calculate the total dimensions needed for the text with each word on separate line"""
        words = self.text.split(' ')
        lines = []
        
        # Put each word on its own line
        for word in words:
            if word.strip():  # Only add non-empty words
                lines.append([word])
        
        # Calculate total dimensions
        max_width = 0
        for line in lines:
            line_width = 0
            for i, word in enumerate(line):
                word_width = len(word) * (self.cell_width + self.letter_spacing) - self.letter_spacing
                line_width += word_width
                if i < len(line) - 1:
                    line_width += self.word_spacing
            max_width = max(max_width, line_width)
        
        total_height = len(lines) * self.cell_height + (len(lines) - 1) * self.line_spacing
        
        return max_width, total_height, lines
    
    def create_image(self):
        """Create the Braille text image"""
        print(f"Generating Braille text: '{self.text}'")
        print(f"Resolution: {self.width}x{self.height}")
        
        # Initialize image with background color
        image = np.full((self.height, self.width, 3), self.background_color, dtype=np.float32)
        
        # Calculate text positioning with line wrapping
        text_width, text_height, lines = self.calculate_text_dimensions()
        start_x = (self.width - text_width) // 2
        start_y = (self.height - text_height) // 2
        
        # Get dot positions
        dot_positions = self.get_dot_positions()
        
        # Process each line
        current_y = start_y
        for line_idx, line in enumerate(lines):
            # Calculate line width for centering
            line_width = 0
            for i, word in enumerate(line):
                word_width = len(word) * (self.cell_width + self.letter_spacing) - self.letter_spacing
                line_width += word_width
                if i < len(line) - 1:
                    line_width += self.word_spacing
            
            # Center the line horizontally
            current_x = (self.width - line_width) // 2
            
            # Process each word in the line
            for word_idx, word in enumerate(line):
                for char_idx, char in enumerate(word):
                    if char in self.braille_patterns:
                        pattern = self.braille_patterns[char]
                        
                        # Draw each dot in the pattern
                        for dot_idx, is_raised in enumerate(pattern):
                            if is_raised:
                                dot_x_offset, dot_y_offset = dot_positions[dot_idx]
                                
                                # Calculate absolute dot position
                                dot_x = current_x + dot_x_offset * (self.cell_width // 2)
                                dot_y = current_y + dot_y_offset * (self.cell_height // 3)
                                
                                # Draw the dot
                                self.draw_dot(image, dot_x, dot_y)
                    
                    # Move to next character position
                    current_x += self.cell_width + self.letter_spacing
                
                # Add word spacing (except for last word in line)
                if word_idx < len(line) - 1:
                    current_x += self.word_spacing - self.letter_spacing
            
            # Move to next line
            current_y += self.cell_height + self.line_spacing
        
        # Convert to 0-255 range for display
        self.image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        print("Braille text generation complete!")
    
    def draw_dot(self, image, center_x, center_y):
        """Draw a single Braille dot at the specified position"""
        # Create coordinate grids
        y_indices, x_indices = np.ogrid[:self.height, :self.width]
        
        # Calculate distance from center
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        
        # Create circular mask
        dot_mask = distances <= self.dot_radius
        
        # Apply dot color
        image[dot_mask] = self.dot_color
    
    def display_and_save(self):
        """Display and save the image"""
        # Create figure with exact pixel dimensions
        dpi = 100
        fig_width = self.width / dpi
        fig_height = self.height / dpi
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display image
        ax.imshow(self.image, origin='upper')
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)  # Flip Y axis for correct orientation
        ax.axis('off')  # Remove axes
        
        # Remove margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save image
        safe_text = self.text.replace(' ', '_').replace('/', '_')
        filename = f"braille_text_{safe_text}_{self.width}x{self.height}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
        
        print(f"Image saved as: {filename}")
        
        # Print Braille information
        print(f"\nBraille Text: '{self.text}'")
        print("Braille patterns used:")
        for char in self.text:
            if char in self.braille_patterns and char != ' ':
                pattern = self.braille_patterns[char]
                print(f"  {char}: {pattern}")
        
        # Show image
        plt.show()

class BraillePatternGenerator:
    """Helper class to visualize individual Braille patterns"""
    
    @staticmethod
    def print_braille_chart():
        """Print a chart of all Braille letters"""
        patterns = {
            'A': [1, 0, 0, 0, 0, 0], 'B': [1, 1, 0, 0, 0, 0], 'C': [1, 0, 0, 1, 0, 0],
            'D': [1, 0, 0, 1, 1, 0], 'E': [1, 0, 0, 0, 1, 0], 'F': [1, 1, 0, 1, 0, 0],
            'G': [1, 1, 0, 1, 1, 0], 'H': [1, 1, 0, 0, 1, 0], 'I': [0, 1, 0, 1, 0, 0],
            'J': [0, 1, 0, 1, 1, 0], 'K': [1, 0, 1, 0, 0, 0], 'L': [1, 1, 1, 0, 0, 0],
            'M': [1, 0, 1, 1, 0, 0], 'N': [1, 0, 1, 1, 1, 0], 'O': [1, 0, 1, 0, 1, 0],
            'P': [1, 1, 1, 1, 0, 0], 'Q': [1, 1, 1, 1, 1, 0], 'R': [1, 1, 1, 0, 1, 0],
            'S': [0, 1, 1, 1, 0, 0], 'T': [0, 1, 1, 1, 1, 0], 'U': [1, 0, 1, 0, 0, 1],
            'V': [1, 1, 1, 0, 0, 1], 'W': [0, 1, 0, 1, 1, 1], 'X': [1, 0, 1, 1, 0, 1],
            'Y': [1, 0, 1, 1, 1, 1], 'Z': [1, 0, 1, 0, 1, 1]
        }
        
        print("Braille Alphabet Chart:")
        print("=" * 40)
        for letter, pattern in patterns.items():
            # Convert pattern to visual representation
            dots = ["⠁", "⠂", "⠄", "⠈", "⠐", "⠠"]  # Unicode Braille patterns
            visual = ""
            for i, dot in enumerate(pattern):
                if dot:
                    if i == 0: visual += "●"  # Top-left
                    elif i == 1: visual += "●"  # Middle-left  
                    elif i == 2: visual += "●"  # Bottom-left
                    elif i == 3: visual += "●"  # Top-right
                    elif i == 4: visual += "●"  # Middle-right
                    elif i == 5: visual += "●"  # Bottom-right
                else:
                    visual += "○"
            
            print(f"{letter}: {pattern} - {visual[:3]} {visual[3:]}")

def main():
    """Main function to create and display the Braille text"""
    print("Braille Text Visualizer")
    print("=" * 40)
    
    # Show Braille chart
    BraillePatternGenerator.print_braille_chart()
    print("\n")
    
    # Create visualizer using the global BRAILLE_TEXT variable
    visualizer = BrailleTextVisualizer(width=1600, height=600)
    
    # Display and save
    visualizer.display_and_save()

def create_custom_text(text, width=1600, height=600):
    """Create Braille visualization for custom text"""
    visualizer = BrailleTextVisualizer(width=width, height=height, text=text)
    visualizer.display_and_save()

if __name__ == "__main__":
    main()
