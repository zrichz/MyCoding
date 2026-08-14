"""
K-Means Color Palette Reducer
Reduces the color palette of an image using k-means clustering
"""

import gradio as gr
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def multiply_blend(base, top):
    """
    Blend two images using multiply mode
    
    Args:
        base: PIL Image (bottom layer)
        top: PIL Image (top layer)
        
    Returns:
        PIL Image with multiply blend applied
    """
    base_array = np.array(base).astype(float) / 255.0
    top_array = np.array(top).astype(float) / 255.0
    
    blended = base_array * top_array
    blended = (blended * 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def find_closest_color(pixel, palette):
    """Find the closest color in the palette to the given pixel"""
    distances = np.sum((palette - pixel) ** 2, axis=1)
    return np.argmin(distances)


def apply_floyd_steinberg_dithering(img_array, palette):
    """
    Apply Floyd-Steinberg dithering to reduce color banding
    
    Args:
        img_array: numpy array of shape (height, width, 3) in float
        palette: numpy array of available colors
        
    Returns:
        numpy array with dithered colors
    """
    height, width = img_array.shape[:2]
    output = np.copy(img_array).astype(float)
    
    for y in range(height):
        for x in range(width):
            old_pixel = output[y, x]
            new_pixel_idx = find_closest_color(old_pixel, palette)
            new_pixel = palette[new_pixel_idx]
            output[y, x] = new_pixel
            
            # Calculate quantization error
            quant_error = old_pixel - new_pixel
            
            # Distribute error to neighboring pixels (Floyd-Steinberg weights)
            if x + 1 < width:
                output[y, x + 1] += quant_error * 7/16
            if y + 1 < height:
                if x > 0:
                    output[y + 1, x - 1] += quant_error * 3/16
                output[y + 1, x] += quant_error * 5/16
                if x + 1 < width:
                    output[y + 1, x + 1] += quant_error * 1/16
    
    return np.clip(output, 0, 255).astype(np.uint8)


def reduce_palette(image, num_colors, apply_dithering, use_half_res, multiply_overlay):
    """
    Reduce image color palette using k-means clustering
    
    Args:
        image: PIL Image or numpy array
        num_colors: number of colors to reduce to
        apply_dithering: whether to apply Floyd-Steinberg dithering
        use_half_res: process at half resolution for speed
        multiply_overlay: blend result onto original using multiply mode
        
    Returns:
        PIL Image with reduced palette
    """
    if image is None:
        return None
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    
    # Ensure RGB
    img = img.convert("RGB")
    original_img = img.copy()
    original_size = img.size
    
    # Resize to half resolution if requested
    if use_half_res:
        new_width = img.size[0] // 2
        new_height = img.size[1] // 2
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array of shape (num_pixels, 3)
    pixels = np.array(img).reshape(-1, 3)
    
    # Run k-means clustering
    kmeans = KMeans(n_clusters=int(num_colors), n_init="auto", random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers (the distilled colors)
    colors = kmeans.cluster_centers_
    
    if apply_dithering:
        # Apply Floyd-Steinberg dithering
        img_array = np.array(img)
        dithered = apply_floyd_steinberg_dithering(img_array, colors)
        new_img = Image.fromarray(dithered)
    else:
        # Simple quantization without dithering
        colors_int = colors.astype(int)
        labels = kmeans.labels_
        new_pixels = colors_int[labels]
        
        # Reshape back to original image dimensions
        new_img = Image.fromarray(
            new_pixels.reshape(img.size[1], img.size[0], 3).astype("uint8")
        )
    
    # Resize back to original size if we used half resolution
    if use_half_res:
        new_img = new_img.resize(original_size, Image.Resampling.NEAREST)
    
    # Apply multiply blend if requested
    if multiply_overlay:
        new_img = multiply_blend(original_img, new_img)
    
    return new_img


# Create Gradio interface
with gr.Blocks(title="K-Means Color Palette Reducer") as demo:
    gr.Markdown("# K-Means Color Palette Reducer")
    gr.Markdown("Upload an image and reduce its color palette using k-means clustering")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image",
                type="pil"
            )
            num_colors = gr.Slider(
                minimum=2,
                maximum=64,
                value=8,
                step=1,
                label="Number of Colors"
            )
            apply_dithering = gr.Checkbox(
                label="Apply Dithering (Floyd-Steinberg)",
                value=True,
                info="Reduces banding and creates smoother gradients"
            )
            use_half_res = gr.Checkbox(
                label="Half Resolution Processing",
                value=False,
                info="Process at 50% size for faster results with large images"
            )
            multiply_overlay = gr.Checkbox(
                label="Multiply Blend with Original",
                value=False,
                info="Overlay result onto original image using multiply mode"
            )
            process_btn = gr.Button("Process Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(
                label="Reduced Palette Image",
                type="pil"
            )
    
    # Process button click
    process_btn.click(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res, multiply_overlay],
        outputs=output_image
    )
    
    # Also process on slider change if image is loaded
    num_colors.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res, multiply_overlay],
        outputs=output_image
    )
    
    # Process on checkbox changes
    apply_dithering.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res, multiply_overlay],
        outputs=output_image
    )
    
    use_half_res.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res, multiply_overlay],
        outputs=output_image
    )
    
    multiply_overlay.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res, multiply_overlay],
        outputs=output_image
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
