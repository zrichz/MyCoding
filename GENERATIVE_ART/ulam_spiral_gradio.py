"""
Ulam Spiral Visualizer - Gradio Version
Creates a visual representation of the Ulam Spiral where prime numbers are arranged in a spiral pattern.
White dots represent prime numbers, black dots represent composite numbers.
Web-based interface using Gradio.
"""

import gradio as gr
import numpy as np
from math import ceil, sqrt
from PIL import Image as PILImage
import io
import tempfile

class UlamSpiralGenerator:
    def __init__(self):
        self.max_display_size = 600  # Maximum display size for performance
    
    def cell(self, n, x, y, start=1):
        """Calculate the number at position (x, y) in an n×n Ulam spiral"""
        d, y, x = 0, y - n//2, x - (n - 1)//2
        l = 2*max(abs(x), abs(y))
        d = (l*3 + x + y) if y >= x else (l - x - y)
        return (l - 1)**2 + d + start - 1
    
    def sieve_of_eratosthenes(self, limit):
        """Generate prime numbers up to limit using Sieve of Eratosthenes"""
        is_prime = [False, False, True] + [True, False] * (limit // 2)
        
        for x in range(3, 1 + int(sqrt(limit)), 2):
            if not is_prime[x]:
                continue
            for i in range(x*x, limit, x*2):
                is_prime[i] = False
        
        return is_prime
    
    def generate_spiral_data(self, grid_size, start_number):
        """Generate the Ulam spiral data"""
        n = int(grid_size)
        start = int(start_number)
        
        # Calculate the maximum number in the spiral
        max_number = start + n * n - 1
        
        # Generate prime sieve
        is_prime = self.sieve_of_eratosthenes(max_number + 1)
        
        # Generate spiral grid
        spiral_grid = np.zeros((n, n), dtype=int)
        prime_grid = np.zeros((n, n), dtype=bool)
        
        for y in range(n):
            for x in range(n):
                number = self.cell(n, x, y, start)
                spiral_grid[y, x] = number
                if number < len(is_prime):
                    prime_grid[y, x] = is_prime[number]
        
        return spiral_grid, prime_grid
    
    def generate_hexagonal_data(self, grid_size, start_number):
        """Generate prime flags for values arranged in a center-out hexagonal spiral."""
        count = int(grid_size) ** 2
        start = int(start_number)

        radius = 0
        while 1 + 3 * radius * (radius + 1) < count:
            radius += 1

        q_parts = [np.array([0], dtype=np.int32)]
        r_parts = [np.array([0], dtype=np.int32)]
        directions = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

        for ring in range(1, radius + 1):
            q = -ring
            r = ring
            steps = np.arange(ring, dtype=np.int32)
            for dq, dr in directions:
                q_parts.append(q + dq * steps)
                r_parts.append(r + dr * steps)
                q += dq * ring
                r += dr * ring

        q_coords = np.concatenate(q_parts)[:count]
        r_coords = np.concatenate(r_parts)[:count]
        max_number = start + count - 1
        sieve = np.asarray(self.sieve_of_eratosthenes(max_number + 1), dtype=bool)
        numbers = start + np.arange(count)
        prime_flags = sieve[numbers]

        return q_coords, r_coords, prime_flags, radius

    def create_hexagonal_image(self, grid_size, start_number, max_dimension=None):
        """Render the hexagonal spiral as a tessellation of hexagonal cells."""
        q_coords, r_coords, prime_flags, radius = self.generate_hexagonal_data(
            grid_size, start_number
        )

        color_lookup = np.full((2 * radius + 1, 2 * radius + 1), 64, dtype=np.uint8)
        color_lookup[r_coords + radius, q_coords + radius] = np.where(
            prime_flags, 255, 0
        )

        x_coords = sqrt(3) * (q_coords + r_coords / 2)
        y_coords = 1.5 * r_coords
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        span_x = max_x - min_x + 2
        span_y = max_y - min_y + 2

        cell_radius = 1.0
        if max_dimension is not None:
            cell_radius = max_dimension / max(span_x, span_y)

        width = max(1, ceil(span_x * cell_radius))
        height = max(1, ceil(span_y * cell_radius))
        pixel_x = (np.arange(width, dtype=np.float32) + 0.5) / cell_radius
        pixel_y = (np.arange(height, dtype=np.float32) + 0.5) / cell_radius
        world_x = pixel_x[None, :] + min_x - 1
        world_y = pixel_y[:, None] + min_y - 1

        q_float = (sqrt(3) / 3 * world_x) - (world_y / 3)
        r_float = (2 / 3) * world_y
        q_float, r_float = np.broadcast_arrays(q_float, r_float)
        q_round = np.rint(q_float).astype(np.int32)
        r_round = np.rint(r_float).astype(np.int32)
        s_float = -q_float - r_float
        s_round = np.rint(s_float).astype(np.int32)

        q_diff = np.abs(q_round - q_float)
        r_diff = np.abs(r_round - r_float)
        s_diff = np.abs(s_round - s_float)
        q_largest = (q_diff > r_diff) & (q_diff > s_diff)
        r_largest = (r_diff > s_diff) & ~q_largest
        q_round[q_largest] = -r_round[q_largest] - s_round[q_largest]
        r_round[r_largest] = -q_round[r_largest] - s_round[r_largest]

        valid = (
            (q_round >= -radius)
            & (q_round <= radius)
            & (r_round >= -radius)
            & (r_round <= radius)
        )
        pixel_colors = np.full((height, width), 64, dtype=np.uint8)
        safe_q = np.clip(q_round, -radius, radius) + radius
        safe_r = np.clip(r_round, -radius, radius) + radius
        sampled_colors = color_lookup[safe_r, safe_q]
        pixel_colors[valid] = sampled_colors[valid]

        return PILImage.fromarray(pixel_colors, mode="L").convert("RGB"), int(
            prime_flags.sum()
        )

    def create_display_image(self, grid_size, start_number, hexagonal=False):
        """Create display image for the web interface"""
        try:
            if hexagonal:
                pil_image, prime_count = self.create_hexagonal_image(
                    grid_size, start_number, self.max_display_size
                )
            else:
                _, prime_data = self.generate_spiral_data(grid_size, start_number)

                display_size = min(self.max_display_size, int(grid_size))
                pixel_scale = display_size / int(grid_size)
                display_array = np.full(
                    (display_size, display_size, 3), 64, dtype=np.uint8
                )

                for grid_y in range(int(grid_size)):
                    for grid_x in range(int(grid_size)):
                        start_x = int(grid_x * pixel_scale)
                        end_x = int((grid_x + 1) * pixel_scale)
                        start_y = int(grid_y * pixel_scale)
                        end_y = int((grid_y + 1) * pixel_scale)
                        color = (
                            [255, 255, 255]
                            if prime_data[grid_y, grid_x]
                            else [0, 0, 0]
                        )
                        display_array[start_y:end_y, start_x:end_x] = color

                pil_image = PILImage.fromarray(display_array, mode="RGB")
                prime_count = np.sum(prime_data)
            
            # Calculate statistics
            total_numbers = int(grid_size) * int(grid_size)
            prime_percentage = (prime_count / total_numbers) * 100 if total_numbers > 0 else 0
            arrangement = "Hexagonal" if hexagonal else "Square"
            
            stats_text = f"""**Spiral Statistics:**
- **Arrangement:** {arrangement}
- **Grid Size:** {grid_size}×{grid_size}
- **Start Number:** {start_number}
- **Total Numbers:** {total_numbers:,}
- **Prime Numbers:** {prime_count:,}
- **Prime Percentage:** {prime_percentage:.2f}%
- **Range:** {start_number} to {start_number + total_numbers - 1}"""
            
            return pil_image, stats_text
            
        except Exception as e:
            # Return error image
            error_img = PILImage.new('RGB', (400, 300), color=(64, 64, 64))
            return error_img, f"**Error:** {str(e)}"
    
    def create_high_res_image(self, grid_size, start_number, hexagonal=False):
        """Create a high-resolution image for download."""
        if hexagonal:
            pil_image, _ = self.create_hexagonal_image(grid_size, start_number)
        else:
            _, prime_data = self.generate_spiral_data(grid_size, start_number)

            save_width = int(grid_size) * 2
            save_height = int(grid_size) * 2
            image_array = np.full(
                (save_height, save_width, 3), 64, dtype=np.uint8
            )

            for grid_y in range(int(grid_size)):
                for grid_x in range(int(grid_size)):
                    pixel_x = grid_x * 2
                    pixel_y = grid_y * 2
                    color = (
                        [255, 255, 255]
                        if prime_data[grid_y, grid_x]
                        else [0, 0, 0]
                    )
                    image_array[pixel_y:pixel_y + 2, pixel_x:pixel_x + 2] = color

            pil_image = PILImage.fromarray(image_array, mode="RGB")

        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=False)
        img_buffer.seek(0)

        arrangement = "_hexagonal" if hexagonal else ""
        filename = (
            f"ulam_spiral_{grid_size}x{grid_size}_start{start_number}"
            f"{arrangement}.png"
        )

        return filename, img_buffer.getvalue()

# Initialize generator
generator = UlamSpiralGenerator()

def generate_spiral_interface(
    grid_size, start_number, hexagonal, progress=gr.Progress()
):
    """Interface function for Gradio"""
    progress(0, desc="Starting spiral generation...")
    
    # Update progress
    progress(0.3, desc="Generating prime sieve...")
    
    # Generate display image and stats
    display_image, stats_text = generator.create_display_image(
        grid_size, start_number, hexagonal
    )
    
    progress(0.8, desc="Creating visualization...")
    
    progress(1.0, desc="Complete.")
    
    return display_image, stats_text

def download_high_res(grid_size, start_number, hexagonal):
    """Generate high-resolution image for download"""
    filename, img_data = generator.create_high_res_image(
        grid_size, start_number, hexagonal
    )
    if not img_data:
        raise RuntimeError("PNG generation returned no image data.")

    with tempfile.NamedTemporaryFile(
        prefix=f"{filename[:-4]}_", suffix=".png", delete=False
    ) as png_file:
        png_file.write(img_data)

    return gr.update(value=png_file.name, visible=True)

# Create Gradio interface
with gr.Blocks(title="Ulam Spiral Visualizer", theme=gr.themes.Soft()) as app:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>Ulam Spiral Visualizer</h1>
        <p style="font-size: 18px; color: #666;">
            Explore prime number patterns in the famous Ulam Spiral<br>
            <strong>White dots = Prime numbers</strong> • <strong>Black dots = Composite numbers</strong>
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>Controls</h3>")
            
            grid_size = gr.Slider(
                minimum=50,
                maximum=1000,
                step=10,
                value=200,
                label="Grid Size",
                info="Square grid dimensions; hex mode keeps the same total number of values"
            )
            
            start_number = gr.Number(
                value=1,
                label="Start Number",
                info="Starting number for the spiral (minimum: 1)",
                precision=0
            )

            hexagonal = gr.Checkbox(
                label="Hexagonal arrangement",
                value=False,
                info="Arrange the same number of values in a center-out hexagonal spiral"
            )
            
            generate_btn = gr.Button(
                "Generate Spiral",
                variant="primary",
                size="lg"
            )
            
            gr.HTML("<h3>Statistics</h3>")
            stats_display = gr.Markdown(
                """**Spiral Statistics:**
- **Arrangement:** Not generated
- **Grid Size:** Not generated
- **Start Number:** Not generated
- **Total Numbers:** Not generated
- **Prime Numbers:** Not generated
- **Prime Percentage:** Not generated
- **Range:** Not generated"""
            )
            
            gr.HTML("<h3>Download</h3>")
            gr.HTML("""
            <p style="font-size: 14px; color: #666;">
                Download a high-resolution PNG of the current arrangement
            </p>
            """)
            
            download_btn = gr.Button(
                "Download High-Res PNG",
                variant="secondary"
            )
            
            download_file = gr.File(
                label="Download File",
                visible=False
            )
        
        with gr.Column(scale=2):
            gr.HTML("<h3>Visualization</h3>")
            
            spiral_image = gr.Image(
                label="Ulam Spiral",
                type="pil",
                interactive=False,
                height=600
            )
            
            gr.HTML("""
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin-top: 10px;">
                <h4>How to Use:</h4>
                <ul style="margin: 10px 0;">
                    <li><strong>Adjust Grid Size:</strong> Use the slider to change spiral dimensions</li>
                    <li><strong>Change Start Number:</strong> Enter a different starting number to explore ranges</li>
                    <li><strong>Hexagonal Arrangement:</strong> Enable the toggle to arrange values in a hexagonal spiral</li>
                    <li><strong>Generate:</strong> Click "Generate Spiral" to create the visualization</li>
                    <li><strong>Download:</strong> Get pixel-perfect PNG files for detailed analysis</li>
                </ul>
                <p><strong>Note:</strong> Display is capped at 600×600 pixels for performance. Download for full resolution.</p>
            </div>
            """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_spiral_interface,
        inputs=[grid_size, start_number, hexagonal],
        outputs=[spiral_image, stats_display],
        show_progress=True
    )
    
    download_btn.click(
        fn=download_high_res,
        inputs=[grid_size, start_number, hexagonal],
        outputs=[download_file]
    )
    
    # Auto-generate on startup
    app.load(
        fn=generate_spiral_interface,
        inputs=[grid_size, start_number, hexagonal],
        outputs=[spiral_image, stats_display]
    )

def main():
    """Launch the Gradio app"""
    print("Ulam Spiral Visualizer - Gradio Version")
    print("=" * 50)
    print("Features:")
    print("- Interactive web interface")
    print("- Real-time spiral generation")
    print("- Adjustable grid size (50×50 to 1000×1000)")
    print("- Custom start numbers")
    print("- High-resolution PNG downloads")
    print("- Prime number statistics")
    print("=" * 50)
    
    # Launch with sharing enabled for remote access
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
        quiet=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
