"""
K-Means Color Palette Reducer
Reduces the color palette of an image using k-means clustering
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from skimage import color


def find_closest_color(pixel, palette):
    """Find the closest color in the palette to the given pixel"""
    distances = np.sum((palette - pixel) ** 2, axis=1)
    return np.argmin(distances)


def apply_bayer_dithering(img_array, palette):
    """
    Apply 4x4 Bayer ordered dithering pattern
    
    Args:
        img_array: numpy array of shape (height, width, 3) in float
        palette: numpy array of available colors
        
    Returns:
        numpy array with dithered colors
    """
    # 4x4 Bayer matrix (normalized to -0.5 to 0.5 range)
    bayer_matrix = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) / 16.0 - 0.5
    
    height, width = img_array.shape[:2]
    output = np.zeros_like(img_array, dtype=np.uint8)
    
    # Apply dithering threshold based on Bayer matrix
    for y in range(height):
        for x in range(width):
            # Get the threshold from the Bayer matrix
            threshold = bayer_matrix[y % 4, x % 4]
            
            # Add threshold to pixel value
            dithered_pixel = img_array[y, x] + threshold * 50
            dithered_pixel = np.clip(dithered_pixel, 0, 255)
            
            # Find closest color in palette
            new_pixel_idx = find_closest_color(dithered_pixel, palette)
            output[y, x] = palette[new_pixel_idx]
    
    return output


def reduce_palette(image, num_colors, apply_dithering, use_half_res):
    """
    Reduce image color palette using k-means clustering in LAB color space
    
    Args:
        image: PIL Image or numpy array
        num_colors: number of colors to reduce to
        apply_dithering: whether to apply 4x4 Bayer pattern dithering
        use_half_res: process at half resolution for speed
        
    Returns:
        Tuple of reduced-palette and RGB-distance images
    """
    if image is None:
        return None, None
    
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
    
    # Convert RGB to LAB color space for perceptually uniform clustering
    rgb_array = np.array(img)
    lab_array = color.rgb2lab(rgb_array)
    
    # Reshape to (num_pixels, 3) for k-means
    pixels = lab_array.reshape(-1, 3)
    
    # Run k-means clustering on LAB data
    kmeans = KMeans(n_clusters=int(num_colors), n_init="auto", random_state=42)
    kmeans.fit(pixels)
    
    # Get cluster centers in LAB space
    colors_lab = kmeans.cluster_centers_
    
    # Convert palette back to RGB for display
    colors_rgb = color.lab2rgb(colors_lab.reshape(1, -1, 3)).reshape(-1, 3) * 255
    
    if apply_dithering:
        # Apply 4x4 Bayer pattern dithering in RGB space
        img_array = rgb_array.astype(float)
        dithered = apply_bayer_dithering(img_array, colors_rgb)
        new_img = Image.fromarray(dithered)
    else:
        # Simple quantization without dithering
        colors_int = colors_rgb.astype(int)
        labels = kmeans.labels_
        new_pixels = colors_int[labels]
        
        # Reshape back to original image dimensions
        new_img = Image.fromarray(
            new_pixels.reshape(img.size[1], img.size[0], 3).astype("uint8")
        )
    
    # Resize back to original size if we used half resolution
    if use_half_res:
        new_img = new_img.resize(original_size, Image.Resampling.NEAREST)
    
    original_array = np.asarray(original_img, dtype=np.float32)
    reduced_array = np.asarray(new_img, dtype=np.float32)
    distance = np.linalg.norm(original_array - reduced_array, axis=2)
    distance_image = Image.fromarray(
        np.clip(distance / (255.0 * np.sqrt(3.0)) * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    distance_image = ImageOps.equalize(distance_image)

    return new_img, distance_image


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# K-Means Color Palette Reducer")

    with gr.Row():
        num_colors = gr.Slider(
            minimum=2,
            maximum=64,
            value=8,
            step=1,
            label="Number of Colors",
            scale=2,
        )
        apply_dithering = gr.Checkbox(
            label="Apply Dithering (4x4 Bayer Pattern)",
            value=True,
            info="Ordered dither",
        )
        use_half_res = gr.Checkbox(
            label="Half Res Processing",
            value=False,
            info="Process at 50% size",
        )
        process_btn = gr.Button("Process", variant="primary")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil", format="png", height=620)
        output_image = gr.Image(label="Reduced Palette", type="pil", format="png", height=620)
        distance_image = gr.Image(label="RGB Distance", type="pil", format="png", height=620)
    
    # Process button click
    process_btn.click(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res],
        outputs=[output_image, distance_image]
    )
    
    # Also process on slider change if image is loaded
    num_colors.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res],
        outputs=[output_image, distance_image]
    )
    
    # Process on checkbox changes
    apply_dithering.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res],
        outputs=[output_image, distance_image]
    )
    
    use_half_res.change(
        fn=reduce_palette,
        inputs=[input_image, num_colors, apply_dithering, use_half_res],
        outputs=[output_image, distance_image]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
