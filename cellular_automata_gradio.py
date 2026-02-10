import numpy as np
import gradio as gr
from PIL import Image
import colorsys

class CellularAutomata:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        
    def apply_rule(self, left, center, right, rule_number):
        """
        Apply Wolfram rule to three cells.
        Rule number is converted to binary to determine output for each configuration.
        """
        # Convert rule number to 8-bit binary
        rule_binary = format(rule_number, '08b')
        
        # Calculate the index based on the three cells
        index = left * 4 + center * 2 + right
        
        # Return the corresponding bit from the rule (reversed because of binary ordering)
        return int(rule_binary[7 - index])
    
    def generate(self, rule_number):
        """
        Generate cellular automata pattern.
        
        Args:
            rule_number: Wolfram rule number (0-255)
        """
        # Initialize grid with random initial state
        grid = np.zeros((self.height, self.width), dtype=np.uint8)
        grid[0] = np.random.randint(0, 2, self.width)
        
        # Generate subsequent rows
        for row in range(1, self.height):
            for col in range(self.width):
                left = grid[row-1, (col-1) % self.width]
                center = grid[row-1, col]
                right = grid[row-1, (col+1) % self.width]
                
                grid[row, col] = self.apply_rule(left, center, right, rule_number)
        
        # Create binary image (black and white)
        binary_image = self.create_binary_image(grid)
        
        # Create colored image (pass binary grid for neighbor counting)
        colored_image = self.create_colored_image(grid)
        
        # Scale up both images to 1024x1024 using nearest neighbor
        scaled_binary = binary_image.resize((1024, 1024), Image.NEAREST)
        scaled_colored = colored_image.resize((1024, 1024), Image.NEAREST)
        
        return scaled_binary, scaled_colored
    
    def create_binary_image(self, grid):
        """
        Create a simple black and white image from the binary grid.
        """
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for row in range(self.height):
            for col in range(self.width):
                if grid[row, col] == 1:
                    img_array[row, col] = [255, 255, 255]  # White
                else:
                    img_array[row, col] = [0, 0, 0]  # Black
        
        return Image.fromarray(img_array)
    
    def create_colored_image(self, grid):
        """
        Create a colored image from the binary grid.
        Red: number of consecutive FALSE cells directly above (0-20 maps to 0-255)
        Green: number of consecutive FALSE cells directly to the left (0-20 maps to 0-255)
        Blue: always 0
        """
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for row in range(self.height):
            for col in range(self.width):
                # Count consecutive FALSE cells directly above
                false_above = 0
                for i in range(1, min(row + 1, 21)):  # Check up to 20 cells above
                    if grid[row - i, col] == 0:  # FALSE
                        false_above += 1
                    else:
                        break  # Stop at first TRUE cell
                
                # Count consecutive FALSE cells directly to the left
                false_left = 0
                for i in range(1, min(col + 1, 21)):  # Check up to 20 cells left
                    if grid[row, col - i] == 0:  # FALSE
                        false_left += 1
                    else:
                        break  # Stop at first TRUE cell
                
                # Map counts to RGB values (0-20 maps to 0-255)
                red = min(255, int((false_above / 20.0) * 255))
                green = min(255, int((false_left / 20.0) * 255))
                blue = 0
                
                img_array[row, col] = [red, green, blue]
        
        return Image.fromarray(img_array)


def generate_ca(rule_number):
    """
    Gradio interface function to generate cellular automata.
    Returns both binary and colored images.
    """
    ca = CellularAutomata(width=256, height=256)
    binary_image, colored_image = ca.generate(rule_number)
    return binary_image, colored_image


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="1D Cellular Automata Explorer") as demo:
        gr.Markdown("""
        # 1D Cellular Automata Explorer
        """)
        
        # Controls row at top
        with gr.Row():
            rule_slider = gr.Slider(
                minimum=0, 
                maximum=255, 
                value=115, 
                step=1, 
                label="Rule Number (0-255)",
                info="Try 30, 90, 110, 115, 184"
            )
            
            generate_btn = gr.Button("Generate", variant="primary", size="lg")
        
        # Images side by side
        with gr.Row():
            binary_output = gr.Image(
                label="Binary Representation",
                type="pil"
            )
            
            colored_output = gr.Image(
                label="Colored Version",
                type="pil"
            )
        
        # Generate on button click
        generate_btn.click(
            fn=generate_ca,
            inputs=[rule_slider],
            outputs=[binary_output, colored_output]
        )
        
        # Generate on parameter change (auto-update)
        rule_slider.change(
            fn=generate_ca,
            inputs=[rule_slider],
            outputs=[binary_output, colored_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True
    )
