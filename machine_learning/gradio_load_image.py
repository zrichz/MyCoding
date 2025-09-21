import gradio as gr
from PIL import Image
import numpy as np

def process_image(image, slider1_value, slider2_value, slider3_value):
    if image is None:
        return None, "No image uploaded"
    
    # Convert the image to a PIL Image
    pil_image = Image.fromarray(np.array(image))
    
    # Adjust width of image non-linearly, using the slider values
    # Define the coefficients
    a = slider1_value
    b = slider2_value
    c = slider3_value
    d = 1.0

    # Get image dimensions
    width, height = pil_image.size
    pixels = np.array(pil_image)

    # Calc max possible new_x value
    max_new_x = 0
    for x in range(width):
        new_x = int(a * (x / width)**3 + b * (x / width)**2 + c * (x / width) + d * width)
        max_new_x = max(max_new_x, new_x)
    
    # Create a new array with appropriate width (add 1 because indexing starts at 0)
    new_width = max_new_x + 1
    transformed_pixels = np.zeros((height, new_width, pixels.shape[2]), dtype=pixels.dtype)

    # Apply transform
    for y in range(height):
        for x in range(width):
            new_x = int(a * (x / width)**3 + b * (x / width)**2 + c * (x / width) + d * width)
            if 0 <= new_x < new_width:  # Ensure new_x is within bounds
                transformed_pixels[y, new_x] = pixels[y, x]

    # Convert transformed pixels back to a PIL Image
    pil_image = Image.fromarray(transformed_pixels)
    
    return pil_image, f"processed image, values: {slider1_value}, {slider2_value}, {slider3_value}"

# Create interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload an Image")
        
        with gr.Column():
            sl_a = gr.Slider(minimum=-40, maximum=40, step=0.1, value=0, label="a")
            sl_b = gr.Slider(minimum=-40, maximum=40, step=0.1, value=0, label="b")
            sl_c = gr.Slider(minimum=-800, maximum=800, step=0.1, value=0, label="c")
    
    with gr.Row():
        output_image = gr.Image(label="Processed Image")
        output_text = gr.Textbox(label="Output")
    
    submit_button = gr.Button("Process Image")
    
    submit_button.click(
        fn=process_image,  
        inputs=[image_input, sl_a, sl_b, sl_c],
        outputs=[output_image, output_text]
    )

# Restart the server completely
if __name__ == "__main__":
    demo.launch(inbrowser=True)
