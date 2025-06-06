import gradio as gr
from PIL import Image, ImageEnhance
import numpy as np

def process_image(image, slider1_value, slider2_value, slider3_value):
    if image is None:
        return None, "No image uploaded"
    
    # Convert the image to a PIL Image
    pil_image = Image.fromarray(np.array(image))
    
    # Adjust contrast, brightness, and hue based on slider values
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(slider1_value)
    
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(slider2_value)
    
    # Fix the overflow error by using proper data type conversion
    hsv_image = pil_image.convert("HSV")
    hsv_array = np.array(hsv_image)
    
    # Convert to int32 before performing arithmetic to avoid overflow
    h_channel = hsv_array[..., 0].astype(np.int32)
    hue_shift = int(slider3_value)
    
    # Now perform the arithmetic and modulo, then convert back to uint8
    h_channel = (h_channel + hue_shift) % 256
    hsv_array[..., 0] = h_channel.astype(np.uint8)
    
    pil_image = Image.fromarray(hsv_array, "HSV").convert("RGB")
    
    return pil_image, f"Image processed with slider values: {slider1_value}, {slider2_value}, {slider3_value}"

# Create interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload an Image")
        
        with gr.Column():
            sl_contrast = gr.Slider(minimum=0, maximum=2, step=0.1, value=1, label="Contrast")
            sl_brightness = gr.Slider(minimum=0, maximum=2, step=0.1, value=1, label="Brightness")
            sl_hue = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Hue")
    
    with gr.Row():
        output_image = gr.Image(label="Processed Image")
        output_text = gr.Textbox(label="Output")
    
    submit_button = gr.Button("Process Image")
    
    submit_button.click(
        fn=process_image,  
        inputs=[image_input, sl_contrast, sl_brightness, sl_hue],
        outputs=[output_image, output_text]
    )

# Restart the server completely
if __name__ == "__main__":
    demo.launch(inbrowser=True)
