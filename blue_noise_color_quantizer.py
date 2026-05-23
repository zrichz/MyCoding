#!/home/rich/MyCoding/venvmycoding313/bin/python
import gradio as gr
import numpy as np
from PIL import Image
import random
import math

def calculate_blue_noise_energy(positions):
    """Calculate energy based on proximity of same-color pixels (lower is better)"""
    if len(positions) <= 1:
        return 0
    
    # Vectorized distance calculation
    pos_array = np.array(positions)
    if len(pos_array) == 0:
        return 0
    
    # Calculate all pairwise distances at once
    diff = pos_array[:, np.newaxis, :] - pos_array[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask out diagonal (self-distances)
    mask = dist > 0
    energy = np.sum(1.0 / (dist[mask] + 0.1))
    
    return energy / 2  # Divide by 2 since we count each pair twice

def distribute_colors_blue_noise(r_count, g_count, b_count, w_count, k_count, chunk_size=8, iterations=200):
    """Use simulated annealing to distribute colors in blue noise pattern"""
    total_pixels = chunk_size * chunk_size
    
    # Initialize pixel assignments (0=red, 1=green, 2=blue, 3=white, 4=black)
    pixels = ([0] * r_count + [1] * g_count + [2] * b_count + 
              [3] * w_count + [4] * k_count)
    
    # Pad if needed
    if len(pixels) < total_pixels:
        pixels += [4] * (total_pixels - len(pixels))
    elif len(pixels) > total_pixels:
        pixels = pixels[:total_pixels]
    
    random.shuffle(pixels)
    
    # Convert to 2D grid
    grid = np.array(pixels).reshape(chunk_size, chunk_size)
    
    # Separate positions by color
    def get_color_positions(grid, color):
        return [(y, x) for y in range(chunk_size) for x in range(chunk_size) if grid[y, x] == color]
    
    # Calculate initial energy (only for R, G, B - not white or black)
    def total_energy():
        r_pos = get_color_positions(grid, 0)
        g_pos = get_color_positions(grid, 1)
        b_pos = get_color_positions(grid, 2)
        return (calculate_blue_noise_energy(r_pos) + 
                calculate_blue_noise_energy(g_pos) + 
                calculate_blue_noise_energy(b_pos))
    
    current_energy = total_energy()
    
    # Simulated annealing
    temperature = 5.0
    cooling_rate = 0.99
    
    for iteration in range(iterations):
        # Random swap
        y1, x1 = random.randint(0, chunk_size-1), random.randint(0, chunk_size-1)
        y2, x2 = random.randint(0, chunk_size-1), random.randint(0, chunk_size-1)
        
        # Swap
        grid[y1, x1], grid[y2, x2] = grid[y2, x2], grid[y1, x1]
        
        # Calculate new energy
        new_energy = total_energy()
        
        # Accept or reject
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / temperature):
            current_energy = new_energy
        else:
            # Revert swap
            grid[y1, x1], grid[y2, x2] = grid[y2, x2], grid[y1, x1]
        
        # Cool down
        temperature *= cooling_rate
    
    return grid

def process_8x8_chunk(chunk, quality=200):
    """Process a single 8x8 chunk"""
    chunk_array = np.array(chunk)
    
    # Calculate average color per pixel
    avg_r = np.mean(chunk_array[:, :, 0])
    avg_g = np.mean(chunk_array[:, :, 1])
    avg_b = np.mean(chunk_array[:, :, 2])
    
    # Extract white component (common to all channels)
    white_component = min(avg_r, avg_g, avg_b)
    
    # Remaining color components after extracting white
    red_component = avg_r - white_component
    green_component = avg_g - white_component
    blue_component = avg_b - white_component
    
    # Calculate pixel counts based on component values
    # Each pixel contributes 255 to its channel
    w_count = int(round(64 * white_component / 255.0))
    r_count = int(round(64 * red_component / 255.0))
    g_count = int(round(64 * green_component / 255.0))
    b_count = int(round(64 * blue_component / 255.0))
    
    # Black fills the remainder
    k_count = 64 - (r_count + g_count + b_count + w_count)
    
    # Ensure we have exactly 64 pixels
    while k_count < 0:
        # Too many pixels - reduce the largest count
        counts = [(r_count, 'r'), (g_count, 'g'), (b_count, 'b'), (w_count, 'w')]
        counts.sort(reverse=True)
        if counts[0][1] == 'r' and r_count > 0:
            r_count -= 1
        elif counts[0][1] == 'g' and g_count > 0:
            g_count -= 1
        elif counts[0][1] == 'b' and b_count > 0:
            b_count -= 1
        elif counts[0][1] == 'w' and w_count > 0:
            w_count -= 1
        k_count = 64 - (r_count + g_count + b_count + w_count)
    
    # Ensure non-negative
    r_count = max(0, r_count)
    g_count = max(0, g_count)
    b_count = max(0, b_count)
    w_count = max(0, w_count)
    k_count = max(0, k_count)
    
    # Distribute colors using blue noise
    color_grid = distribute_colors_blue_noise(r_count, g_count, b_count, w_count, k_count, iterations=quality)
    
    # Create output chunk
    output_chunk = np.zeros((8, 8, 3), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            if color_grid[y, x] == 0:  # Red
                output_chunk[y, x] = [255, 0, 0]
            elif color_grid[y, x] == 1:  # Green
                output_chunk[y, x] = [0, 255, 0]
            elif color_grid[y, x] == 2:  # Blue
                output_chunk[y, x] = [0, 0, 255]
            elif color_grid[y, x] == 3:  # White
                output_chunk[y, x] = [255, 255, 255]
            else:  # Black
                output_chunk[y, x] = [0, 0, 0]
    
    return output_chunk

def process_image(image, quality):
    """Process the entire image in 8x8 chunks"""
    if image is None:
        yield None, "No image loaded"
        return
    
    # Convert to RGB if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    
    # Get dimensions
    width, height = image.size
    
    # Calculate padded dimensions (multiple of 8)
    padded_width = ((width + 7) // 8) * 8
    padded_height = ((height + 7) // 8) * 8
    
    # Create padded image
    padded_image = Image.new('RGB', (padded_width, padded_height), (0, 0, 0))
    padded_image.paste(image, (0, 0))
    
    # Convert to numpy array
    img_array = np.array(padded_image)
    
    # Create output array with explicit dtype
    output_array = np.zeros_like(img_array, dtype=np.uint8)
    
    # Calculate total chunks
    total_chunks = (padded_height // 8) * (padded_width // 8)
    chunk_count = 0
    
    yield None, f"Starting... (0/{total_chunks} chunks)"
    
    # Process each 8x8 chunk with progress updates
    for y in range(0, padded_height, 8):
        for x in range(0, padded_width, 8):
            chunk = img_array[y:y+8, x:x+8]
            processed_chunk = process_8x8_chunk(chunk, quality)
            output_array[y:y+8, x:x+8] = processed_chunk
            
            chunk_count += 1
            # Update progress and yield intermediate results
            if chunk_count % 10 == 0 or chunk_count == total_chunks:
                # Yield intermediate result for live preview every 10 chunks
                temp_output = Image.fromarray(output_array, mode='RGB')
                temp_output = temp_output.crop((0, 0, width, height))
                percent = int(100 * chunk_count / total_chunks)
                yield temp_output, f"Processing: {percent}% ({chunk_count}/{total_chunks} chunks)"
    
    # Crop back to original size
    output_image = Image.fromarray(output_array, mode='RGB')
    output_image = output_image.crop((0, 0, width, height))
    
    yield output_image, f"✓ Complete! Processed {total_chunks} chunks"

# Create Gradio interface
with gr.Blocks(title="Blue Noise Color Quantizer") as demo:
    gr.Markdown("""
    # Blue Noise Color Quantizer
    
    Processes images in 8×8 pixel chunks:
    - Calculates total R, G, B components per chunk
    - Extracts white (common component) to preserve brightness
    - Uses black pixels for shadows
    - Normalizes to 64 pixels total (Red, Green, Blue, White, Black)
    - Uses simulated annealing to distribute colors in a blue noise pattern
    
    **Quality slider controls blue noise iterations** (higher = better distribution but slower)
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            quality_slider = gr.Slider(
                minimum=5, 
                maximum=500, 
                value=150, 
                step=5, 
                label="Quality (iterations per chunk)",
                info="5=very fast, 150=balanced, 500=best quality"
            )
            with gr.Row():
                process_btn = gr.Button("Process Image", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Processed Image", format="png")
            status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
            gr.Markdown("**⚠️ Important:** Save as PNG (not WebP/JPG) to preserve pure RGB colors without compression artifacts!")
    
    # Process click - show stop button, hide process button
    process_event = process_btn.click(
        fn=process_image,
        inputs=[input_image, quality_slider],
        outputs=[output_image, status_text]
    )
    
    process_btn.click(
        fn=lambda: (gr.Button(visible=False), gr.Button(visible=True)),
        inputs=None,
        outputs=[process_btn, stop_btn],
        queue=False
    )
    
    # Stop button cancels the process
    stop_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[process_event],
        queue=False
    )
    
    # When processing completes, hide stop and show process button
    process_event.then(
        fn=lambda: (gr.Button(visible=True), gr.Button(visible=False)),
        inputs=None,
        outputs=[process_btn, stop_btn],
        queue=False
    )
    
    gr.Examples(
        examples=[],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
