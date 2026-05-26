#!/home/rich/MyCoding/venvmycoding313/bin/python
"""
Gaussian-weighted CLAHE-style local contrast enhancement with Gradio interface

- Operates in perceptual Lab color space to preserve hue and saturation
- Splits luminance channel into overlapping tiles
- Applies histogram equalization with Gaussian weighting in each tile
- Blends tiles smoothly using Hanning window to prevent seams
"""

import numpy as np
from skimage import color
from scipy.ndimage import convolve
from PIL import Image
import gradio as gr
import os
import glob

def generate_gaussian_kernel(kernel_size):
    sigma = kernel_size / 6.0
    center = kernel_size // 2
    x, y = np.meshgrid(np.arange(kernel_size) - center, np.arange(kernel_size) - center)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def equalize_tile(tile, kernel):
    smoothed = convolve(tile, kernel, mode='reflect')
    hist, bins = np.histogram(smoothed.flatten(), 256, [0, 256], density=True)
    cdf = 255 * hist.cumsum() / hist.sum()
    return np.interp(tile.flatten(), bins[:-1], cdf).reshape(tile.shape)

def process_luminance_channel_blend(L_channel, tile_size, stride, kernel, progress=gr.Progress()):
    h, w = L_channel.shape
    result = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    blend_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))
    total_tiles = len(y_steps) * len(x_steps)
    tile_count = 0

    for y in y_steps:
        for x in x_steps:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = L_channel[y:y_end, x:x_end]

            kernel_resized = kernel[:tile.shape[0], :tile.shape[1]]
            blend_resized = blend_window[:tile.shape[0], :tile.shape[1]]
            eq_tile = equalize_tile(tile, kernel_resized)

            result[y:y_end, x:x_end] += eq_tile * blend_resized
            weight_sum[y:y_end, x:x_end] += blend_resized

            tile_count += 1
            if tile_count % 10 == 0:
                progress(tile_count / total_tiles, desc=f"Processing tiles {tile_count}/{total_tiles}")

    return np.divide(result, weight_sum, out=np.zeros_like(result), where=weight_sum != 0).astype(np.uint8)

def gaussian_clahe_color(image, tile_size=64, kernel_size=21, stride=None, progress=gr.Progress()):
    if stride is None:
        stride = tile_size // 2

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    img_np = np.asarray(image, dtype=np.float32) / 255.0
    lab = color.rgb2lab(img_np)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    L_scaled = (L / 100.0 * 255).astype(np.uint8)

    kernel = generate_gaussian_kernel(kernel_size)
    L_eq = process_luminance_channel_blend(L_scaled, tile_size, stride, kernel, progress)
    L_eq = (L_eq.astype(np.float32) / 255.0) * 100

    lab_eq = np.stack([L_eq, A, B], axis=2)
    rgb_eq = np.clip(color.lab2rgb(lab_eq), 0, 1)
    return Image.fromarray((rgb_eq * 255).astype(np.uint8))

def process_single_image(image, tile_size, kernel_size, stride, max_size, progress=gr.Progress()):
    if image is None:
        return None, "No image uploaded"
    
    progress(0, desc="Loading image")
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        img = image.convert('RGB')
    
    # Resize based on max_size
    img.thumbnail((max_size, max_size))
    
    progress(0.1, desc="Applying enhancement")
    result = gaussian_clahe_color(img, tile_size, kernel_size, stride, progress)
    
    message = (f"Successfully enhanced image\n"
               f"Tile size: {tile_size}, Kernel size: {kernel_size}, Stride: {stride}\n"
               f"Max dimension: {max_size}px\n"
               f"Output size: {result.size[0]}x{result.size[1]}")
    
    return result, message

def batch_process_directory(input_files, tile_size, kernel_size, stride, max_size, progress=gr.Progress()):
    if not input_files:
        return None, "No files selected"
    
    output_images = []
    total_files = len(input_files)
    
    for idx, file_path in enumerate(input_files):
        progress((idx / total_files), desc=f"Processing file {idx+1}/{total_files}")
        
        img = Image.open(file_path).convert('RGB')
        img.thumbnail((max_size, max_size))
        
        result = gaussian_clahe_color(img, tile_size, kernel_size, stride, progress)
        output_images.append(result)
    
    message = f"Processed {total_files} images\nTile size: {tile_size}, Kernel size: {kernel_size}, Stride: {stride}"
    
    return output_images, message

# Create Gradio interface
with gr.Blocks(title="Gaussian CLAHE Image Enhancer") as demo:
    gr.Markdown("# Gaussian CLAHE Image Enhancer")
    gr.Markdown("Enhance image contrast using Gaussian-weighted CLAHE in Lab color space")
    
    with gr.Tabs():
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Input Image")
                    with gr.Row():
                        tile_size = gr.Slider(8, 256, value=64, step=8, label="Tile Size")
                        kernel_size = gr.Slider(3, 101, value=21, step=2, label="Kernel Size")
                    with gr.Row():
                        stride = gr.Slider(4, 128, value=16, step=4, label="Stride")
                        max_size = gr.Radio([360, 512, 1024, 2048], value=512, label="Max Image Size")
                    process_btn = gr.Button("Apply Enhancement", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(type="pil", label="Enhanced Image")
                    status_text = gr.Textbox(label="Status", lines=4)
            
            process_btn.click(
                fn=process_single_image,
                inputs=[input_image, tile_size, kernel_size, stride, max_size],
                outputs=[output_image, status_text]
            )
        
        with gr.Tab("Batch Process"):
            with gr.Row():
                with gr.Column():
                    batch_input = gr.File(file_count="multiple", label="Upload Images", file_types=["image"])
                    with gr.Row():
                        batch_tile_size = gr.Slider(8, 256, value=64, step=8, label="Tile Size")
                        batch_kernel_size = gr.Slider(3, 101, value=21, step=2, label="Kernel Size")
                    with gr.Row():
                        batch_stride = gr.Slider(4, 128, value=16, step=4, label="Stride")
                        batch_max_size = gr.Radio([360, 512, 1024, 2048], value=512, label="Max Image Size")
                    batch_process_btn = gr.Button("Process All Images", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Gallery(label="Enhanced Images", columns=2, height="auto")
                    batch_status = gr.Textbox(label="Status", lines=3)
            
            batch_process_btn.click(
                fn=batch_process_directory,
                inputs=[batch_input, batch_tile_size, batch_kernel_size, batch_stride, batch_max_size],
                outputs=[batch_output, batch_status]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
