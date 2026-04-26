#!/usr/bin/env python3
"""
Painterly Filter Gradio App
Applies artistic painterly effects using structure tensor analysis for brush stroke direction detection.
Optimized for speed with preview mode option.
"""

import numpy as np
import gradio as gr
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter


def compute_structure_tensor(image_gray, sigma=1.5):
    """
    Compute the structure tensor to find local image orientation.
    Returns dominant orientation angle and coherence at each pixel.
    """
    # Compute gradients
    dx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Structure tensor components
    dx2 = gaussian_filter(dx * dx, sigma)
    dy2 = gaussian_filter(dy * dy, sigma)
    dxdy = gaussian_filter(dx * dy, sigma)
    
    # Compute eigenvalues and eigenvectors
    trace = dx2 + dy2
    det = dx2 * dy2 - dxdy * dxdy
    
    # Eigenvalues
    lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4*det + 1e-10))
    lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4*det + 1e-10))
    
    # Orientation angle of dominant eigenvector (perpendicular to edges)
    # For brush strokes to FOLLOW edges, we add π/2 to rotate 90 degrees
    orientation_perpendicular = 0.5 * np.arctan2(2 * dxdy, dx2 - dy2)
    orientation = orientation_perpendicular + np.pi / 2
    
    # Coherence (measure of orientation strength)
    coherence = np.divide(lambda1 - lambda2, lambda1 + lambda2 + 1e-10)
    coherence = np.clip(coherence, 0, 1)
    
    return orientation, coherence


def fast_kuwahara_filter(image, kernel_size=5):
    """
    Fast approximation of Kuwahara filter using OpenCV's edge-preserving filter.
    Much faster than traditional implementation while maintaining quality.
    """
    # Use edge-preserving filter for oil painting effect
    # flags=1: normalized convolution (better for oil painting)
    # sigma_s: spatial smoothing (kernel_size * 6-8 works well)
    # sigma_r: color range threshold (0.05-0.15 for visible oil effect)
    sigma_s = kernel_size * 6
    sigma_r = 0.08  # Lower values preserve edges better and create stronger oil effect
    result = cv2.edgePreservingFilter(image, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    return result


def apply_oriented_brush_strokes_optimized(image, orientation, coherence, brush_length_min=5, 
                                           brush_length_max=30, brush_width_min=1, 
                                           brush_width_max=5, opacity=0.8, color_simplification=0,
                                           stroke_density=1000):
    """
    Optimized brush strokes using vectorized operations with random length/width.
    """
    h, w = image.shape[:2]
    
    # Validate and swap if min > max
    if brush_length_min > brush_length_max:
        brush_length_min, brush_length_max = brush_length_max, brush_length_min
    if brush_width_min > brush_width_max:
        brush_width_min, brush_width_max = brush_width_max, brush_width_min
    
    # Color quantization for painterly effect
    if color_simplification > 0:
        image_quant = (image // color_simplification) * color_simplification
    else:
        image_quant = image
    
    # Create multiple canvases for different stroke directions
    canvas = np.zeros_like(image, dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.float32)
    
    # Calculate spacing based on desired stroke density
    # stroke_density represents approximate number of strokes
    total_pixels = h * w
    spacing = max(1, int(np.sqrt(total_pixels / stroke_density)))
    y_coords = np.arange(0, h, spacing)
    x_coords = np.arange(0, w, spacing)
    
    # Vectorized stroke generation
    for y in y_coords:
        for x in x_coords:
            if y >= h or x >= w:
                continue
            
            # Get local properties
            angle = orientation[y, x]
            coh = coherence[y, x]
            color = image_quant[y, x].astype(np.float32)
            
            # Random brush length within specified range, modulated by coherence
            if brush_length_min == brush_length_max:
                base_length = brush_length_min
            else:
                base_length = np.random.randint(brush_length_min, brush_length_max + 1)
            actual_length = int(base_length * (0.5 + 0.5 * coh))
            
            # Random brush width within specified range
            if brush_width_min == brush_width_max:
                actual_width = brush_width_min
            else:
                actual_width = np.random.randint(brush_width_min, brush_width_max + 1)
            
            # Calculate stroke endpoints
            dx = actual_length * np.cos(angle)
            dy = actual_length * np.sin(angle)
            
            x1 = int(np.clip(x - dx/2, 0, w-1))
            y1 = int(np.clip(y - dy/2, 0, h-1))
            x2 = int(np.clip(x + dx/2, 0, w-1))
            y2 = int(np.clip(y + dy/2, 0, h-1))
            
            # Draw stroke directly on canvas
            cv2.line(canvas, (x1, y1), (x2, y2), tuple(color.tolist()), actual_width)
            cv2.line(coverage, (x1, y1), (x2, y2), 1.0, actual_width)
    
    # Normalize by coverage and smooth
    coverage_3d = np.stack([coverage] * 3, axis=2)
    canvas_normalized = np.divide(canvas, coverage_3d + 1e-10, 
                                  out=image.astype(np.float32), 
                                  where=coverage_3d > 0)
    
    # Light smoothing
    canvas_smooth = cv2.GaussianBlur(canvas_normalized.astype(np.float32), (3, 3), 0.5)
    
    # Blend with original image
    result = opacity * canvas_smooth + (1 - opacity) * image.astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_painterly_filter(image, brush_length_min=5, brush_length_max=30, 
                           brush_width_min=1, brush_width_max=5, structure_sigma=2.0,
                           color_simplification=0, edge_enhancement=0.0, detail_level=1.0,
                           stroke_opacity=0.8, kuwahara_size=5, use_kuwahara=True,
                           bilateral_color=75, bilateral_space=75, preview_mode=False,
                           preview_size=200, stroke_density=1000):
    """
    Optimized painterly filter with preview mode option.
    """
    if image is None:
        return None
    
    # Validate and swap if min > max
    if brush_length_min > brush_length_max:
        brush_length_min, brush_length_max = brush_length_max, brush_length_min
    if brush_width_min > brush_width_max:
        brush_width_min, brush_width_max = brush_width_max, brush_width_min
    
    # Convert to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    original_shape = image.shape[:2]
    
    # Preview mode: resize for faster processing
    if preview_mode:
        h, w = image.shape[:2]
        scale = preview_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_process = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Scale brush parameters
        brush_length_min_scaled = max(1, int(brush_length_min * scale))
        brush_length_max_scaled = max(2, int(brush_length_max * scale))
        brush_width_min_scaled = max(1, int(brush_width_min * scale))
        brush_width_max_scaled = max(1, int(brush_width_max * scale))
        # Scale stroke density proportionally to image size reduction
        stroke_density_scaled = max(10, int(stroke_density * scale * scale))
    else:
        image_process = image
        brush_length_min_scaled = brush_length_min
        brush_length_max_scaled = brush_length_max
        brush_width_min_scaled = brush_width_min
        brush_width_max_scaled = brush_width_max
        stroke_density_scaled = stroke_density
    
    # Step 1: Fast bilateral filter for edge-preserving smoothing
    smoothed = cv2.bilateralFilter(image_process, d=5, 
                                    sigmaColor=bilateral_color, 
                                    sigmaSpace=bilateral_space)
    
    # Step 2: Apply fast Kuwahara-like filter if enabled
    if use_kuwahara:
        smoothed = fast_kuwahara_filter(smoothed, kernel_size=kuwahara_size)
    
    # Step 3: Compute structure tensor for orientation analysis
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    orientation, coherence = compute_structure_tensor(gray, sigma=structure_sigma)
    
    # Step 4: Apply detail level (blend between original and smoothed)
    base_image = (detail_level * image_process.astype(np.float32) + 
                  (1 - detail_level) * smoothed.astype(np.float32))
    base_image = np.clip(base_image, 0, 255).astype(np.uint8)
    
    # Step 5: Apply optimized oriented brush strokes
    result = apply_oriented_brush_strokes_optimized(
        base_image, orientation, coherence,
        brush_length_min=brush_length_min_scaled,
        brush_length_max=brush_length_max_scaled,
        brush_width_min=brush_width_min_scaled,
        brush_width_max=brush_width_max_scaled,
        opacity=stroke_opacity,
        color_simplification=color_simplification,
        stroke_density=stroke_density_scaled
    )
    
    # Step 6: Edge enhancement
    if edge_enhancement > 0:
        # Detect edges
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend edges into result
        result = result.astype(np.float32)
        result = result - edge_enhancement * edges_color.astype(np.float32)
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Resize back to original size if in preview mode
    if preview_mode and result.shape[:2] != original_shape:
        result = cv2.resize(result, (original_shape[1], original_shape[0]), 
                          interpolation=cv2.INTER_LINEAR)
    
    return result


def process_image(image, brush_length_min, brush_length_max, brush_width_min, 
                 brush_width_max, stroke_density, structure_sigma, 
                 color_simplification, edge_enhancement, detail_level, 
                 stroke_opacity, kuwahara_size, use_kuwahara, 
                 bilateral_color, bilateral_space, preview_mode):
    """
    Wrapper function for Gradio interface.
    """
    if image is None:
        return None
    
    result = apply_painterly_filter(
        image=image,
        brush_length_min=brush_length_min,
        brush_length_max=brush_length_max,
        brush_width_min=brush_width_min,
        brush_width_max=brush_width_max,
        structure_sigma=structure_sigma,
        color_simplification=color_simplification,
        edge_enhancement=edge_enhancement,
        detail_level=detail_level,
        stroke_opacity=stroke_opacity,
        kuwahara_size=kuwahara_size,
        use_kuwahara=use_kuwahara,
        bilateral_color=bilateral_color,
        bilateral_space=bilateral_space,
        preview_mode=preview_mode,
        preview_size=200,
        stroke_density=stroke_density
    )
    
    return result


# Create Gradio interface with 4-column layout
with gr.Blocks(title="Painterly Filter App", css="""
    .image-container img {
        max-width: none !important;
        max-height: none !important;
        width: auto !important;
        height: auto !important;
    }
    .image-container {
        overflow: auto !important;
        max-height: 1200px !important;
    }
""") as demo:
    gr.Markdown("""
    # Painterly Filter Application
    
    Upload an image and adjust parameters to create artistic painterly effects.
    Uses structure tensor analysis for SOTA brush stroke direction detection.
    """)
    
    with gr.Row():
        # Column 1: Input Image
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(label="Upload Image", type="numpy")
            preview_mode = gr.Checkbox(value=False, 
                                      label="Quick Preview Mode",
                                      info="Process at 200px for faster preview")
            process_btn = gr.Button("Apply Filter", variant="primary", size="lg")
        
        # Column 2: Primary Parameters
        with gr.Column(scale=1):
            gr.Markdown("### Brush Stroke Parameters")
            brush_length_min = gr.Slider(1, 200, value=5, step=1, 
                                        label="Brush Length Min",
                                        info="Minimum stroke length")
            brush_length_max = gr.Slider(1, 200, value=30, step=1, 
                                        label="Brush Length Max",
                                        info="Maximum stroke length")
            brush_width_min = gr.Slider(1, 40, value=1, step=1,
                                       label="Brush Width Min",
                                       info="Minimum stroke thickness")
            brush_width_max = gr.Slider(1, 40, value=5, step=1,
                                       label="Brush Width Max",
                                       info="Maximum stroke thickness")
            stroke_density = gr.Slider(100, 50000, value=1000, step=100,
                                      label="Stroke Count",
                                      info="Approximate number of brush strokes")
            stroke_opacity = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                      label="Stroke Opacity",
                                      info="Paint dominance")
            
            gr.Markdown("### Color and Style")
            color_simplification = gr.Slider(0, 32, value=0, step=2,
                                            label="Color Simplification",
                                            info="Reduce palette")
            detail_level = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                    label="Detail Preservation",
                                    info="Smooth to detailed")
            edge_enhancement = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                        label="Edge Enhancement",
                                        info="Edge darkness")
        
        # Column 3: Advanced Parameters
        with gr.Column(scale=1):
            gr.Markdown("### Advanced Parameters")
            structure_sigma = gr.Slider(0.5, 5.0, value=2.0, step=0.1,
                                       label="Structure Analysis",
                                       info="Direction detection scale")
            use_kuwahara = gr.Checkbox(value=True, 
                                      label="Oil Painting Filter",
                                      info="Enable oil painting effect")
            kuwahara_size = gr.Slider(3, 21, value=7, step=2,
                                     label="Oil Effect Strength",
                                     info="Higher values = stronger oil effect")
            bilateral_color = gr.Slider(10, 200, value=50, step=5,
                                       label="Color Smoothing",
                                       info="Lower = preserve more color variation")
            bilateral_space = gr.Slider(10, 200, value=50, step=5,
                                       label="Spatial Smoothing",
                                       info="Lower = preserve more spatial detail")
        
        # Column 4: Output Image
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            output_image = gr.Image(label="Filtered Result", type="numpy")
    
    # Connect the main process button
    process_btn.click(
        fn=process_image,
        inputs=[input_image, brush_length_min, brush_length_max, brush_width_min,
                brush_width_max, stroke_density, structure_sigma, 
                color_simplification, edge_enhancement, detail_level, 
                stroke_opacity, kuwahara_size, use_kuwahara, 
                bilateral_color, bilateral_space, preview_mode],
        outputs=output_image
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
