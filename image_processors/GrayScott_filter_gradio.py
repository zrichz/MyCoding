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


def sharpen_image(image):
    """Apply sharpening filter"""
    return image.filter(ImageFilter.SHARPEN)


def blur_image(image):
    """Apply Gaussian blur"""
    return image.filter(ImageFilter.GaussianBlur(radius=1))


def gray_scott_filter(image, iterations):
    """
    Apply Gray-Scott reaction-diffusion filter
    
    Args:
        image: PIL Image
        iterations: Number of sharpen-blur cycles
    
    Returns:
        Processed PIL Image
    """
    if image is None:
        return None, "⚠️ Please upload an image first"
    
    try:
        iterations = int(iterations)
        if iterations <= 0:
            return None, "⚠️ Please enter a positive number of iterations"
        
        # Convert to greyscale
        processed = image.convert('L')
        
        # Apply iterations of sharpen + blur
        for i in range(iterations):
            processed = sharpen_image(processed)
            processed = sharpen_image(processed)  # Double sharpen
            processed = blur_image(processed)
        
        status = f"✓ Processing complete! Applied {iterations} iterations of Gray-Scott filter\n"
        status += f"Image size: {processed.width}x{processed.height} pixels"
        
        return processed, status
        
    except ValueError:
        return None, "⚠️ Error: Please enter a valid number of iterations"
    except Exception as e:
        return None, f"⚠️ Error processing image: {str(e)}"


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
                
                with gr.Row():
                    process_btn = gr.Button("🔄 Process Gray-Scott Filter", variant="primary", size="lg")
                    reset_btn = gr.Button("↩️ Reset", variant="secondary", size="lg")
            
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
        
        def process_and_store(img, iters):
            """Process and store original"""
            result, status = gray_scott_filter(img, iters)
            return result, status, img  # Store original in state
        
        def reset_to_original(orig_img):
            """Reset to original uploaded image"""
            if orig_img is None:
                return None, "⚠️ No original image to reset to"
            return orig_img, "↩️ Reset to original image"
        
        # Main process button
        process_btn.click(
            fn=process_and_store,
            inputs=[input_image, iterations_slider],
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
    print("=" * 70)
    
    app = create_interface()
    app.launch(
        inbrowser=True,
        show_error=True,
        share=False
    )
