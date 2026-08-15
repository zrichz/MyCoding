"""
Gray-Scott Filter - Image Processor (Gradio Version)

A reaction-diffusion pattern generator that creates artistic effects by:
1. Converting image to grayscale
2. Applying multiple iterations of sharpen + blur filters
3. Optional binarization (50% threshold)
4. Morphological operations (erosion/dilation) on binarized images

The filter creates patterns similar to biological systems like spots and stripes.
"""

from PIL import Image, ImageFilter
import numpy as np
import gradio as gr

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Morphological operations will use basic numpy implementation.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Cardinal snapping will not be available.")


def multiply_blend(base, top, inverse_lines=False):
    """
    Blend two images using multiply mode or inverse line mode
    
    Args:
        base: PIL Image (bottom layer, RGB)
        top: PIL Image (top layer, grayscale)
        inverse_lines: If True, dark lines become inverse color of base, light areas keep base color
        
    Returns:
        PIL Image with blend applied
    """
    # Convert base to RGB if needed
    if base.mode != 'RGB':
        base = base.convert('RGB')
    
    # Ensure same size
    if base.size != top.size:
        top = top.resize(base.size, Image.Resampling.LANCZOS)
    
    base_array = np.array(base).astype(float) / 255.0
    
    if inverse_lines:
        # Inverse color blending mode
        # Dark lines (low values in top) get inverse color of base
        # Light areas (high values in top) keep base color
        
        # Get top as grayscale array
        if top.mode == 'L':
            top_gray = np.array(top).astype(float) / 255.0
        else:
            top_gray = np.array(top.convert('L')).astype(float) / 255.0
        
        # Calculate inverse of base
        inverse_base = 1.0 - base_array
        
        # Use top_gray as weight: 0 (dark/lines) = use inverse, 1 (light/background) = use base
        # Expand top_gray to 3 channels
        weight = np.stack([top_gray] * 3, axis=-1)
        
        # Blend: base * weight + inverse_base * (1 - weight)
        blended = base_array * weight + inverse_base * (1.0 - weight)
    else:
        # Standard multiply blend
        # Convert top to RGB (replicate grayscale across all channels)
        if top.mode == 'L':
            top_rgb = top.convert('RGB')
        else:
            top_rgb = top
        
        top_array = np.array(top_rgb).astype(float) / 255.0
        blended = base_array * top_array
    
    blended = (blended * 255).astype(np.uint8)
    return Image.fromarray(blended)


def pixellate_image(image, pixel_size):
    """
    Apply pixellation effect to image by downsampling and upsampling
    
    Args:
        image: PIL Image
        pixel_size: size of each pixel block (2-32)
        
    Returns:
        PIL Image with pixellation effect
    """
    if pixel_size <= 1:
        return image
    
    # Get original size
    original_size = image.size
    
    # Calculate new size (downsampled)
    small_width = max(1, original_size[0] // pixel_size)
    small_height = max(1, original_size[1] // pixel_size)
    
    # Downsample using NEAREST to create blocky effect
    small = image.resize((small_width, small_height), Image.Resampling.NEAREST)
    
    # Upsample back to original size using NEAREST to preserve blocks
    pixellated = small.resize(original_size, Image.Resampling.NEAREST)
    
    return pixellated


def snap_to_cardinal_directions(image):
    """
    Snap lines to cardinal and diagonal directions (45-degree increments: 0, 45, 90, 135, 180, 225, 270, 315 degrees)
    
    Args:
        image: PIL Image (grayscale)
        
    Returns:
        PIL Image with lines snapped to cardinal and diagonal directions
    """
    if not CV2_AVAILABLE:
        print("Warning: cv2 not available, skipping cardinal snapping")
        return image
    
    try:
        # Convert PIL to cv2 format (numpy array)
        img_array = np.array(image)
        
        # Threshold the image (binary inverse)
        _, th = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours - use LIST retrieval to get all contours
        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create white output image
        out = np.ones_like(img_array) * 255
        
        # Collect all contours with their perimeters for thickness scaling
        contour_data = []
        for c in contours:
            if len(c) < 2:
                continue
            perimeter = cv2.arcLength(c, closed=True)
            contour_data.append((c, perimeter))
        
        # Define thickness range (min 1 pixel, max 8 pixels)
        min_thickness = 1
        max_thickness = 8
        
        # Find min and max perimeters for scaling
        if contour_data:
            perimeters = [p for _, p in contour_data]
            min_perim = min(perimeters)
            max_perim = max(perimeters)
        else:
            min_perim = 0
            max_perim = 0
        
        # Process each contour
        for c, perimeter in contour_data:
            # Calculate thickness proportional to perimeter
            if max_perim > min_perim:
                # Scale thickness from min to max based on perimeter
                thickness_ratio = (perimeter - min_perim) / (max_perim - min_perim)
                thickness = int(min_thickness + thickness_ratio * (max_thickness - min_thickness))
                thickness = max(1, thickness)  # Ensure at least 1 pixel
            else:
                thickness = min_thickness
            
            # Approximate polygon with larger epsilon for smoother results
            epsilon = 3.0
            approx = cv2.approxPolyDP(c, epsilon, closed=True)
            pts = approx.reshape(-1, 2)
            
            if len(pts) < 2:
                continue
            
            # Build snapped path maintaining connectivity
            snapped_path = []
            current_pos = pts[0].astype(float)
            snapped_path.append(current_pos.astype(int))
            
            # Process each segment
            for i in range(len(pts)):
                # Get next point (wrapping around for closed contour)
                next_idx = (i + 1) % len(pts)
                p = current_pos
                q = pts[next_idx].astype(float)
                
                dx, dy = q - p
                
                # Calculate angle
                ang = np.degrees(np.arctan2(dy, dx))
                
                # Snap to nearest 45-degree increment (0, 45, 90, 135, 180, 225, 270, 315)
                snapped = round(ang / 45) * 45
                rad = np.radians(snapped)
                
                # Calculate length
                length = np.hypot(dx, dy)
                
                # Calculate new endpoint with snapped angle, starting from current position
                new_endpoint = p + np.array([np.cos(rad), np.sin(rad)]) * length
                
                # Update current position for next segment
                current_pos = new_endpoint
                snapped_path.append(new_endpoint.astype(int))
            
            # Draw the snapped contour as a polyline with proportional thickness
            if len(snapped_path) > 1:
                pts_array = np.array(snapped_path, dtype=np.int32)
                cv2.polylines(out, [pts_array], isClosed=True, color=0, thickness=thickness)
        
        # Convert back to PIL Image
        return Image.fromarray(out)
        
    except Exception as e:
        print(f"Error in cardinal snapping: {e}")
        import traceback
        traceback.print_exc()
        return image


def sharpen_image(image):
    """Apply sharpening filter"""
    return image.filter(ImageFilter.SHARPEN)


def blur_image(image):
    """Apply Gaussian blur"""
    return image.filter(ImageFilter.GaussianBlur(radius=1))


def gray_scott_filter(image, iterations, use_half_res=False, snap_cardinal=False, multiply_overlay=False, pixellate_amount=1, inverse_line_colors=False):
    """
    Apply Gray-Scott reaction-diffusion filter
    
    Args:
        image: PIL Image
        iterations: Number of sharpen-blur cycles
        use_half_res: Process at half resolution for speed
        snap_cardinal: Snap lines to cardinal directions (0/90/180/270 degrees)
        multiply_overlay: Blend result onto original using multiply mode
        pixellate_amount: Pixellation size for multiply blend base layer (1 = no pixellation)
        inverse_line_colors: Make overlaid lines inverse color of base (only with multiply_overlay)
    
    Returns:
        Processed PIL Image and status message
    """
    if image is None:
        return None, "Please upload an image first"
    
    try:
        iterations = int(iterations)
        if iterations <= 0:
            return None, "Please enter a positive number of iterations"
        
        # Store original for multiply blend
        original_img = image.copy()
        original_size = image.size
        
        # Resize to half resolution if requested (for Gray-Scott processing only)
        if use_half_res:
            new_width = image.size[0] // 2
            new_height = image.size[1] // 2
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to greyscale
        processed = image.convert('L')
        
        # Apply iterations of sharpen + blur
        for i in range(iterations):
            processed = sharpen_image(processed)
            processed = sharpen_image(processed)  # Double sharpen
            processed = blur_image(processed)
        
        # Resize back to original size if we used half resolution
        if use_half_res:
            processed = processed.resize(original_size, Image.Resampling.LANCZOS)
        
        # Apply cardinal direction snapping if requested (before multiply blend)
        if snap_cardinal:
            if CV2_AVAILABLE:
                processed = snap_to_cardinal_directions(processed)
            else:
                status = "Error: cv2 not available for cardinal snapping"
                return None, status
        
        # Apply multiply blend if requested
        if multiply_overlay:
            # Use pixellated base if pixellate_amount > 1
            if pixellate_amount > 1:
                base_img = pixellate_image(original_img, int(pixellate_amount))
            else:
                base_img = original_img
            processed = multiply_blend(base_img, processed, inverse_lines=inverse_line_colors)
        
        status = f"Processing complete - Applied {iterations} iterations of Gray-Scott filter\n"
        status += f"Image size: {processed.width}x{processed.height} pixels"
        if use_half_res:
            status += "\nProcessed at half resolution for speed"
        if snap_cardinal:
            status += "\nLines snapped to cardinal/diagonal directions (45-degree increments)"
        if multiply_overlay:
            blend_mode = "inverse color" if inverse_line_colors else "multiply"
            if pixellate_amount > 1:
                status += f"\n{blend_mode.capitalize()} blend applied with pixellated base ({int(pixellate_amount)}px blocks)"
            else:
                status += f"\n{blend_mode.capitalize()} blend applied with original"
        
        return processed, status
        
    except ValueError:
        return None, "Error: Please enter a valid number of iterations"
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def binarize_image(image):
    """
    Convert image to pure black and white using 50% threshold
    
    Args:
        image: PIL Image (grayscale)
    
    Returns:
        Binarized PIL Image
    """
    if image is None:
        return None, "⚠️ Please process an image first"
    
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            grayscale = image.convert('L')
        else:
            grayscale = image.copy()
        
        # Convert to numpy array for thresholding
        img_array = np.array(grayscale)
        
        # Apply 50% threshold (127.5 for 0-255 range)
        binary_array = (img_array > 127).astype(np.uint8) * 255
        
        # Convert back to PIL Image
        binary_image = Image.fromarray(binary_array, mode='L')
        
        status = f"✓ Image binarized using 50% threshold\n"
        status += f"Image size: {binary_image.width}x{binary_image.height} pixels"
        
        return binary_image, status
        
    except Exception as e:
        return None, f"⚠️ Error binarizing image: {str(e)}"


def erode_image(image):
    """
    Apply erosion morphological operation (3x3 structuring element)
    
    Args:
        image: PIL Image (should be binarized)
    
    Returns:
        Eroded PIL Image
    """
    if image is None:
        return None, "⚠️ Please binarize an image first"
    
    try:
        # Convert to numpy array
        img_array = np.array(image)
        binary_array = img_array > 127
        
        if SCIPY_AVAILABLE:
            # Use scipy for proper morphological operations
            structure = np.ones((3, 3), dtype=bool)
            eroded = ndimage.binary_erosion(binary_array, structure=structure)
        else:
            # Basic erosion implementation using numpy
            eroded = np.zeros_like(binary_array)
            h, w = binary_array.shape
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # Check if all pixels in 3x3 neighborhood are white
                    if np.all(binary_array[i-1:i+2, j-1:j+2]):
                        eroded[i, j] = True
        
        # Convert back to 0-255 range
        eroded_array = (eroded.astype(np.uint8)) * 255
        
        # Convert back to PIL Image
        eroded_image = Image.fromarray(eroded_array, mode='L')
        
        status = f"✓ Erosion applied (3x3 structuring element)\n"
        status += f"Image size: {eroded_image.width}x{eroded_image.height} pixels"
        
        return eroded_image, status
        
    except Exception as e:
        return None, f"⚠️ Error applying erosion: {str(e)}"


def dilate_image(image):
    """
    Apply dilation morphological operation (3x3 structuring element)
    
    Args:
        image: PIL Image (should be binarized)
    
    Returns:
        Dilated PIL Image
    """
    if image is None:
        return None, "⚠️ Please binarize an image first"
    
    try:
        # Convert to numpy array
        img_array = np.array(image)
        binary_array = img_array > 127
        
        if SCIPY_AVAILABLE:
            # Use scipy for proper morphological operations
            structure = np.ones((3, 3), dtype=bool)
            dilated = ndimage.binary_dilation(binary_array, structure=structure)
        else:
            # Basic dilation implementation using numpy
            dilated = np.zeros_like(binary_array)
            h, w = binary_array.shape
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # Check if any pixel in 3x3 neighborhood is white
                    if np.any(binary_array[i-1:i+2, j-1:j+2]):
                        dilated[i, j] = True
        
        # Convert back to 0-255 range
        dilated_array = (dilated.astype(np.uint8)) * 255
        
        # Convert back to PIL Image
        dilated_image = Image.fromarray(dilated_array, mode='L')
        
        status = f"✓ Dilation applied (3x3 structuring element)\n"
        status += f"Image size: {dilated_image.width}x{dilated_image.height} pixels"
        
        return dilated_image, status
        
    except Exception as e:
        return None, f"⚠️ Error applying dilation: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Gray-Scott Filter Faker - Image Processor") as app:
        gr.Markdown("# Gray-Scott Filter Faker - Image Processor")
        gr.Markdown("""
        Imitate reaction-diffusion patterns using iterative sharpen-blur cycles.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input & Controls")
                
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )
                
                iterations_slider = gr.Slider(
                    minimum=10,
                    maximum=1500,
                    value=200,
                    step=10,
                    label="Iterations",
                    info="Number of sharpen-blur cycles (more = stronger effect)"
                )
                
                use_half_res = gr.Checkbox(
                    label="Half Resolution Processing",
                    value=False,
                    info="Process at 50% size for faster results with large images"
                )
                
                snap_cardinal = gr.Checkbox(
                    label="Snap to Cardinal and Diagonal Directions",
                    value=False,
                    info="Straighten organic lines to 45-degree increments (0, 45, 90, 135, 180, 225, 270, 315 degrees)"
                )
                
                multiply_overlay = gr.Checkbox(
                    label="Multiply Blend with Original",
                    value=False,
                    info="Overlay result onto original image using multiply mode"
                )
                
                pixellate_amount = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=1,
                    step=1,
                    label="Pixellate Base Layer",
                    info="Pixellation size for multiply blend base (1 = no pixellation, only applies when multiply blend is enabled)"
                )
                
                inverse_line_colors = gr.Checkbox(
                    label="Inverse Line Colors",
                    value=False,
                    info="Make overlaid lines the inverse color of the underlying base (only with multiply blend)"
                )
                
                with gr.Row():
                    process_btn = gr.Button("Process Gray-Scott Filter", variant="primary", size="lg")
                    reset_btn = gr.Button("Reset", variant="secondary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                
                output_image = gr.Image(
                    label="Processed Image",
                    type="pil",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False
                )
        
        # Event handlers
        
        # Store original image in state
        original_img = gr.State(None)
        
        def process_and_store(img, iters, half_res, snap_card, mult_overlay, pixellate_amt, inverse_colors):
            """Process and store original"""
            result, status = gray_scott_filter(img, iters, half_res, snap_card, mult_overlay, pixellate_amt, inverse_colors)
            return result, status, img  # Store original in state
        
        def reset_to_original(orig_img):
            """Reset to original uploaded image"""
            if orig_img is None:
                return None, "No original image to reset to"
            return orig_img, "Reset to original image"
        
        # Main process button
        process_btn.click(
            fn=process_and_store,
            inputs=[input_image, iterations_slider, use_half_res, snap_cardinal, multiply_overlay, pixellate_amount, inverse_line_colors],
            outputs=[output_image, status_text, original_img]
        )
        
        # Reset button
        reset_btn.click(
            fn=reset_to_original,
            inputs=[original_img],
            outputs=[output_image, status_text]
        )
    
    return app


if __name__ == "__main__":
    print("=" * 70)
    print("Gray-Scott Filter - Image Processor")
    print("=" * 70)
    if SCIPY_AVAILABLE:
        print("✓ scipy available - using optimized morphological operations")
    else:
        print("⚠ scipy not available - using basic numpy implementation")
    if CV2_AVAILABLE:
        print("✓ cv2 available - cardinal direction snapping enabled")
    else:
        print("⚠ cv2 not available - cardinal snapping disabled")
    print("=" * 70)
    
    app = create_interface()
    app.launch(
        inbrowser=True,
        show_error=True,
        share=False
    )
