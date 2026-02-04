"""
Optimal Transport Color Matcher
A Gradio app that uses Optimal Transport algorithms to match color palettes between two images.
"""

import gradio as gr
import numpy as np
from PIL import Image
import ot
from sklearn.cluster import MiniBatchKMeans
import cv2


def resize_for_processing(image, max_size=128):
    """
    Resize image for faster processing while maintaining aspect ratio.
    
    Args:
        image: Input image (numpy array)
        max_size: Maximum dimension size
    
    Returns:
        resized_image: Resized numpy array
        scale_factor: Scale factor used for resizing
    """
    h, w = image.shape[:2]
    
    # Check if resizing is needed
    if h <= max_size and w <= max_size:
        return image, 1.0
    
    # Calculate scale factor
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized, scale


def extract_color_palette(image, n_colors=256):
    """
    Extract dominant colors from an image using K-means clustering in LAB color space.
    
    Args:
        image: PIL Image or numpy array (RGB)
        n_colors: Number of colors to extract
    
    Returns:
        colors: Array of shape (actual_n_colors, 3) containing LAB values (normalized)
        weights: Array of shape (actual_n_colors,) containing color weights
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Handle alpha channel: ignore fully transparent pixels
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        rgb_array = img_array[:, :, :3]
        mask = alpha > 0
        if not np.any(mask):
            # Fallback to a single neutral color if no opaque pixels exist
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32), np.array([1.0], dtype=np.float32)
    else:
        rgb_array = img_array
        mask = None
    
    # Convert RGB to LAB color space
    # OpenCV LAB format: L [0, 255], a [0, 255], b [0, 255] (uint8)
    # where a and b have 128 as neutral
    img_lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Normalize to [0, 1] range for clustering
    img_lab = img_lab / 255.0
    
    # Reshape image to be a list of pixels
    if mask is not None:
        pixels = img_lab[mask].reshape(-1, 3)
    else:
        pixels = img_lab.reshape(-1, 3)

    if pixels.size == 0:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32), np.array([1.0], dtype=np.float32)
    
    # Use MiniBatchKMeans for efficiency with large images
    n_clusters = min(n_colors, len(pixels))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    labels = kmeans.fit_predict(pixels)
    
    # Get cluster centers (colors) and their frequencies (weights)
    colors = kmeans.cluster_centers_
    
    # Calculate weights based on frequency - use actual labels to avoid mismatch
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create weight array matching the actual clusters found
    weights = np.zeros(len(colors))
    for label, count in zip(unique_labels, counts):
        weights[label] = count
    weights = weights / weights.sum()
    
    # Remove any empty clusters
    valid_indices = weights > 0
    colors = colors[valid_indices]
    weights = weights[valid_indices]
    
    return colors, weights


def apply_color_mapping(image, original_colors, target_colors, transport_plan, morph_amount=1.0):
    """
    Apply optimal transport color mapping to an image in LAB color space.
    
    Args:
        image: Input image (PIL Image or numpy array) in RGB
        original_colors: Original color palette in LAB space (normalized 0-1)
        target_colors: Target color palette in LAB space (normalized 0-1)
        transport_plan: OT transport plan matrix
        morph_amount: Amount of morphing (0=original, 1=full transport)
    
    Returns:
        Transformed image as numpy array in RGB
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # Preserve alpha channel and ignore transparent pixels
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        rgb_array = img_array[:, :, :3]
        mask = alpha > 0
    else:
        alpha = None
        rgb_array = img_array
        mask = None
    
    h, w = rgb_array.shape[:2]
    
    # Convert RGB to LAB (OpenCV format: L [0, 255], a [0, 255], b [0, 255])
    img_lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Normalize to [0, 1] to match palette normalization
    img_lab = img_lab / 255.0
    
    pixels = img_lab.reshape(-1, 3)
    
    # Normalize transport plan rows to ensure they sum to 1 (prevents darkening)
    transport_normalized = transport_plan / (transport_plan.sum(axis=1, keepdims=True) + 1e-10)
    
    # For each pixel, find the closest color in the original palette
    # and apply the transport plan
    transformed_pixels = pixels.copy()

    if mask is not None:
        mask_flat = mask.reshape(-1)
        masked_indices = np.where(mask_flat)[0]
        for idx in masked_indices:
            pixel = pixels[idx]
            distances = np.linalg.norm(original_colors - pixel, axis=1)
            closest_idx = np.argmin(distances)
            transported_color = (transport_normalized[closest_idx, :, np.newaxis] * target_colors).sum(axis=0)
            transformed_pixels[idx] = pixel * (1 - morph_amount) + transported_color * morph_amount
    else:
        for i, pixel in enumerate(pixels):
            # Find closest color in original palette (in LAB space)
            distances = np.linalg.norm(original_colors - pixel, axis=1)
            closest_idx = np.argmin(distances)
            
            # Apply transport plan - weighted average of target colors
            transported_color = (transport_normalized[closest_idx, :, np.newaxis] * target_colors).sum(axis=0)
            
            # Interpolate between original and transported based on morph_amount
            transformed_pixels[i] = pixel * (1 - morph_amount) + transported_color * morph_amount
    
    # Reshape back to image
    transformed_lab = transformed_pixels.reshape(h, w, 3)
    
    # Denormalize back to [0, 255]
    transformed_lab = (transformed_lab * 255.0).clip(0, 255).astype(np.uint8)
    
    # Convert back to RGB
    transformed_rgb = cv2.cvtColor(transformed_lab, cv2.COLOR_LAB2RGB)

    if alpha is not None:
        transformed_rgba = np.dstack([transformed_rgb, alpha])
        return transformed_rgba
    
    return transformed_rgb


def compute_optimal_transport(colors_a, weights_a, colors_b, weights_b, reg=0.01):
    """
    Compute optimal transport plan between two color palettes in LAB space.
    
    Args:
        colors_a: Color palette A in LAB space (n_colors x 3)
        weights_a: Weights for palette A
        colors_b: Color palette B in LAB space (n_colors x 3)
        weights_b: Weights for palette B
        reg: Regularization parameter for Sinkhorn algorithm
    
    Returns:
        transport_a_to_b: Transport plan from A to B
        transport_b_to_a: Transport plan from B to A
    """
    # Compute pairwise distances between colors (cost matrix)
    M = ot.dist(colors_a, colors_b, metric='euclidean')
    
    # Normalize the cost matrix
    M = M / M.max()
    
    # Compute optimal transport using Sinkhorn algorithm
    transport_a_to_b = ot.sinkhorn(weights_a, weights_b, M, reg=reg)
    transport_b_to_a = ot.sinkhorn(weights_b, weights_a, M.T, reg=reg)
    
    return transport_a_to_b, transport_b_to_a


def color_match_images(image1, image2, morph_amount_1=0.5, morph_amount_2=0.5, 
                       n_colors=512, reg_param=0.01, progress=gr.Progress()):
    """
    Main function to perform color matching between two images.
    
    Args:
        image1: First input image (numpy array, can be RGB or RGBA)
        image2: Second input image (numpy array, can be RGB or RGBA)
        morph_amount_1: How much to morph image1 towards image2 (0-1)
        morph_amount_2: How much to morph image2 towards image1 (0-1)
        n_colors: Number of colors to use in palette extraction
        reg_param: Regularization for optimal transport
        progress: Gradio Progress tracker
    
    Returns:
        transformed_image1: PIL Image with alpha preserved if present
        transformed_image2: PIL Image with alpha preserved if present
    """
    if image1 is None or image2 is None:
        return None, None
    
    # Keep original full-size images
    progress(0.0, desc="Preparing images...")
    original_image1 = image1.copy()
    original_image2 = image2.copy()
    
    # Resize for processing if needed (for palette extraction only)
    progress(0.1, desc="Resizing for palette extraction...")
    resized_1, scale_1 = resize_for_processing(image1, max_size=128)
    resized_2, scale_2 = resize_for_processing(image2, max_size=128)
    
    # Extract color palettes from resized images
    progress(0.2, desc="Extracting color palette from Image 1...")
    colors_1, weights_1 = extract_color_palette(resized_1, n_colors=n_colors)
    
    progress(0.35, desc="Extracting color palette from Image 2...")
    colors_2, weights_2 = extract_color_palette(resized_2, n_colors=n_colors)
    
    # Compute optimal transport plans
    progress(0.5, desc="Computing optimal transport plans...")
    transport_1_to_2, transport_2_to_1 = compute_optimal_transport(
        colors_1, weights_1, colors_2, weights_2, reg=reg_param
    )
    
    # Apply color mapping to FULL-SIZE original images
    progress(0.65, desc="Transforming Image 1...")
    transformed_1 = apply_color_mapping(
        original_image1, colors_1, colors_2, transport_1_to_2, morph_amount=morph_amount_1
    )
    
    progress(0.85, desc="Transforming Image 2...")
    transformed_2 = apply_color_mapping(
        original_image2, colors_2, colors_1, transport_2_to_1, morph_amount=morph_amount_2
    )
    
    # Convert to PIL Images to properly preserve alpha channel in Gradio
    progress(0.95, desc="Converting to PIL Images...")
    def to_pil(arr):
        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, mode='RGBA')
        else:
            return Image.fromarray(arr, mode='RGB')
    
    progress(1.0, desc="Complete!")
    return to_pil(transformed_1), to_pil(transformed_2)


def create_comparison_view(original, transformed):
    """Create a side-by-side comparison image."""
    if original is None or transformed is None:
        return None
    
    orig_array = np.array(original) if isinstance(original, Image.Image) else original
    trans_array = np.array(transformed) if isinstance(transformed, Image.Image) else transformed
    
    # Resize to same height if needed
    h1, h2 = orig_array.shape[0], trans_array.shape[0]
    if h1 != h2:
        scale = h1 / h2
        if scale > 1:
            trans_array = cv2.resize(trans_array, (int(trans_array.shape[1] * scale), h1))
        else:
            orig_array = cv2.resize(orig_array, (int(orig_array.shape[1] / scale), h2))
    
    comparison = np.hstack([orig_array, trans_array])
    return comparison


# Create Gradio interface
with gr.Blocks(title="Optimal Transport Color Matcher") as demo:
    gr.Markdown("""
    # 🎨 Optimal Transport Color Matcher
    
    Upload two images and adjust their color palettes to match each other using Optimal Transport algorithms in perceptually uniform LAB color space.
    
    **How to use:**
    1. Upload two images
    2. Adjust the morph sliders to control how much each image adopts the other's color palette
    3. Fine-tune with advanced settings if needed
    4. Download the results
    """)
    
    with gr.Row():
        with gr.Column():
            image1_input = gr.Image(label="Image 1", type="numpy")
            morph_slider_1 = gr.Slider(
                minimum=0, maximum=1, value=0.5, step=0.01,
                label="Morph Image 1 → Image 2",
                info="0 = Keep original, 1 = Full color match"
            )
        
        with gr.Column():
            image2_input = gr.Image(label="Image 2", type="numpy")
            morph_slider_2 = gr.Slider(
                minimum=0, maximum=1, value=0.5, step=0.01,
                label="Morph Image 2 → Image 1",
                info="0 = Keep original, 1 = Full color match"
            )
    
    with gr.Row():
        reg_slider = gr.Slider(
            minimum=0.001, maximum=0.1, value=0.01, step=0.001,
            label="Regularization",
            info="Controls transport smoothness (lower = more precise but slower)"
        )
    
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            n_colors_slider = gr.Slider(
                minimum=32, maximum=8192, value=512, step=32,
                label="Palette Size",
                info="Number of colors to extract (higher = more detail but slower)"
            )
    
    process_btn = gr.Button("🎨 Match Colors", variant="primary", size="lg")
    
    gr.Markdown("### Results")
    
    with gr.Row():
        output1 = gr.Image(label="Transformed Image 1", type="pil")
        output2 = gr.Image(label="Transformed Image 2", type="pil")
    
    # Connect the processing function
    process_btn.click(
        fn=color_match_images,
        inputs=[
            image1_input, image2_input,
            morph_slider_1, morph_slider_2,
            n_colors_slider, reg_slider
        ],
        outputs=[output1, output2]
    )
    
    gr.Markdown("""
    ---
    ### Tips for Best Results:
    - Start with morph values around 0.5 for both images
    - Lower regularization (0.001-0.01) = more precise color matching
    - Higher regularization (0.05-0.1) = smoother/softer transitions
    - Processing uses LAB color space for perceptually accurate color matching
    - Large images (>128px) are automatically downsized for palette extraction, then applied at full resolution
    - Use higher palette sizes (512-8192) for images with complex colors
    - **Alpha channels are fully preserved** - transparent pixels are ignored during matching
    - Note: K-means clustering is used for computational efficiency (full pixel-level OT on 24-bit images would be intractable)
    """)


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
