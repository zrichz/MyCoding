# Random 2D Cross-section Of (3D) Mandelbulb Fractal
# http://en.wikipedia.org/wiki/Mandelbulb
# FB - 20120707
import math
import random
from PIL import Image
import gradio as gr
import numpy as np

def generate_mandelbulb(imgx, imgy, n, max_iterations, seed=None):
    """
    Generate a 2D cross-section of a 3D Mandelbulb fractal.
    
    Parameters:
    - imgx, imgy: Image dimensions
    - n: Power parameter for the Mandelbulb formula
    - max_iterations: Maximum number of iterations
    - seed: Random seed for reproducible rotation angles (None for random)
    """
    if seed is not None:
        random.seed(seed)
    
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()
    
    # drawing area (xa < xb & ya < yb)
    xa = -1.5
    xb = 1.5
    ya = -1.5
    yb = 1.5
    pi2 = math.pi * 2.0
    
    # random rotation angles to convert 2d plane to 3d plane
    xy = random.random() * pi2
    xz = random.random() * pi2
    yz = random.random() * pi2
    sxy = math.sin(xy) ; cxy = math.cos(xy)
    sxz = math.sin(xz) ; cxz = math.cos(xz)
    syz = math.sin(yz) ; cyz = math.cos(yz)

    origx = (xa + xb) / 2.0 ; origy = (ya + yb) / 2.0
    
    for ky in range(imgy):
        b = ky * (yb - ya) / (imgy - 1)  + ya
        for kx in range(imgx):
            a = kx * (xb - xa) / (imgx - 1)  + xa
            x = a ; y = b ; z = 0.0
            # 3d rotation around center of the plane
            x = x - origx ; y = y - origy
            x0=x*cxy-y*sxy;y=x*sxy+y*cxy;x=x0 # xy-plane rotation
            x0=x*cxz-z*sxz;z=x*sxz+z*cxz;x=x0 # xz-plane rotation 
            y0=y*cyz-z*syz;z=y*syz+z*cyz;y=y0 # yz-plane rotation
            x = x + origx ; y = y + origy

            cx = x ; cy = y ; cz = z
            for i in range(max_iterations):
                r = math.sqrt(x * x + y * y + z * z)
                t = math.atan2(math.hypot(x, y), z)
                p = math.atan2(y, x)
                rn = r ** n
                x = rn * math.sin(t * n) * math.cos(p * n) + cx
                y = rn * math.sin(t * n) * math.sin(p * n) + cy
                z = rn * math.cos(t * n) + cz
                if x * x + y * y + z * z > 4.0: break
            pixels[kx, ky] = (i % 4 * 64, i % 8 * 32, i % 16 * 16)
    
    return image

def generate_wrapper(width, height, power, iterations, use_seed, seed_value):
    """Wrapper function for Gradio interface"""
    actual_seed = int(seed_value) if use_seed else None
    image = generate_mandelbulb(width, height, power, iterations, actual_seed)
    return image

# Create Gradio interface
with gr.Blocks(title="Mandelbulb Fractal Generator") as demo:
    gr.Markdown("# 2D Mandelbulb Fractal Generator")
    gr.Markdown("Generate random 2D cross-sections of the 3D Mandelbulb fractal")
    
    with gr.Row():
        with gr.Column():
            width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, 
                            label="Width")
            height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, 
                             label="Height")
            power = gr.Slider(minimum=2, maximum=16, value=8, step=1, 
                            label="Power (n)")
            iterations = gr.Slider(minimum=64, maximum=512, value=256, step=64, 
                                 label="Max Iterations")
            
            with gr.Row():
                use_seed = gr.Checkbox(label="Use Fixed Seed", value=False)
                seed_value = gr.Number(label="Seed Value", value=42, precision=0)
            
            generate_btn = gr.Button("Generate Mandelbulb", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Fractal", type="pil")
    
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[width, height, power, iterations, use_seed, seed_value],
        outputs=output_image
    )
    
    gr.Markdown("""
    ### Parameters:
    - Width/Height: Output image dimensions
    - Power: Controls the formula exponent (traditional Mandelbulb uses 8)
    - Max Iterations: Higher values reveal more detail but take longer
    - Use Fixed Seed: Enable for reproducible results
    """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
