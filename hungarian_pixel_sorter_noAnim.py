"""
Hungarian Pixel Sorter (no animation).

This script rearranges square blocks of a "start" image so that, when viewed
as a coarse grid, its block colors best match the block colors of an "end"
image. The matching is solved as an assignment problem: each start-image grid
cell is assigned to a destination grid cell such that the total cost (color
difference plus a weighted distance penalty) is minimized. The actual pixel
content placed at each destination is always taken from the start image, so
the result looks like the start image's texture reshuffled into the end
image's color layout.

Workflow:
1. Both images are center-cropped to squares.
2. Each image is downsampled to a small grid_size x grid_size grid to get an
   average color per cell.
3. A cost matrix combining color distance and grid-position distance is built
   and solved with the Hungarian algorithm (linear_sum_assignment) to find the
   optimal one-to-one mapping from start cells to end cells.
4. The start image is re-sliced into that many textured patches, and each
   patch is placed at its assigned destination cell to build the final image.
5. The final image is resized to a fixed 1024x1024 output and saved as a PNG.

A Gradio UI wraps this logic, with an optional mode that first turns the end
image into a diamond-pixelated version to use as the start image.
"""

import gradio as gr
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def center_crop_square(image):
    # Crop the largest possible centered square out of the source image.
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def make_diamond_pixelated_start(image, diamond_size=32):
    # Builds an alternate "start" image by averaging pixels within diamond-
    # shaped (45-degree rotated square) cells, producing a pixelated look
    # whose cell boundaries run diagonally instead of on the usual grid axes.
    if image is None:
        raise gr.Error("Load an end image first.")

    image = center_crop_square(image).convert("RGB")
    pixels = np.asarray(image, dtype=np.uint8)
    height, width = pixels.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    # Rotate coordinates by 45 degrees (via sum/difference) so that grouping
    # by integer bins carves the image into diamonds rather than squares.
    diagonal_a = np.floor((x_coords + y_coords) / diamond_size + 0.5).astype(np.int64)
    diagonal_b = np.floor((x_coords - y_coords) / diamond_size + 0.5).astype(np.int64)
    # Combine the two diagonal bins into a single 1D id (much faster to sort
    # and unique than 2D row tuples) before mapping each pixel to its cell.
    diagonal_a -= diagonal_a.min()
    diagonal_b -= diagonal_b.min()
    combined_id = (diagonal_a * (diagonal_b.max() + 1) + diagonal_b).ravel()
    # `inverse` maps each pixel back to the index of its cell so per-cell
    # stats can be scattered/gathered.
    _, inverse = np.unique(combined_id, return_inverse=True)

    flattened_pixels = pixels.reshape(-1, 3)
    cell_counts = np.bincount(inverse)
    pixelated = np.empty_like(flattened_pixels)
    for channel in range(3):
        # Sum each channel's values per cell, then divide by cell size to get
        # the per-cell average, broadcast back out to every pixel in the cell.
        channel_totals = np.bincount(inverse, weights=flattened_pixels[:, channel])
        pixelated[:, channel] = np.rint(channel_totals[inverse] / cell_counts[inverse])

    return Image.fromarray(pixelated.reshape(height, width, 3))


def generate_final_image(start_image, end_image, grid_size=24, position_distance_factor=0.25):
    # Core algorithm: match start-image grid cells to end-image grid cells by
    # color similarity (with a positional penalty), then rebuild the image by
    # placing each start-image texture patch at its matched destination cell.
    if start_image is None or end_image is None:
        raise gr.Error("Load both images.")

    start_image = center_crop_square(start_image)
    end_image = center_crop_square(end_image)

    grid_size = int(grid_size)
    position_distance_factor = float(position_distance_factor)
    num_pixels = grid_size * grid_size

    # Downsample both images to grid_size x grid_size so each pixel represents
    # the average color of one grid cell.
    start_small = start_image.convert("RGB").resize(
        (grid_size, grid_size), Image.Resampling.LANCZOS
    )
    end_small = end_image.convert("RGB").resize(
        (grid_size, grid_size), Image.Resampling.LANCZOS
    )

    # Flatten grid colors to normalized [0, 1] RGB vectors, one row per cell.
    # float32 halves the memory/compute of the pairwise distance matrices
    # below versus float64, with no meaningful precision loss for this use.
    start_colors = np.asarray(start_small, dtype=np.float32).reshape(-1, 3) / 255.0
    end_colors = np.asarray(end_small, dtype=np.float32).reshape(-1, 3) / 255.0

    # Normalized (x, y) coordinate for every grid cell, used to penalize
    # matches that would move content too far from its original position.
    x_coords, y_coords = np.meshgrid(
        np.linspace(0, 1, grid_size, dtype=np.float32),
        np.linspace(0, 1, grid_size, dtype=np.float32),
    )
    grid_positions = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

    # Pairwise cost between every start cell and every end cell: color
    # distance plus a weighted positional distance, combined into one matrix.
    # cdist computes this in optimized C rather than broadcasting full 3D
    # intermediate arrays in Python, which is significantly faster at larger
    # grid sizes.
    color_cost = cdist(start_colors, end_colors, metric="euclidean")
    position_cost = cdist(grid_positions, grid_positions, metric="euclidean")
    cost_matrix = color_cost + position_distance_factor * position_cost
    # Solve the assignment problem for the minimum-cost one-to-one mapping.
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_order = col_indices[np.argsort(row_indices)]

    # Choose a texture resolution per cell (canvas_scale) so the working
    # canvas stays close to 640px, then upscale the start image to match.
    canvas_scale = min(16, max(1, 640 // grid_size))
    image_size = grid_size * canvas_scale
    start_texture = start_image.convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.LANCZOS,
    )
    # Slice the upscaled start image into grid_size x grid_size textured
    # patches (one canvas_scale x canvas_scale block per grid cell).
    texture_patches = np.asarray(start_texture).reshape(
        grid_size, canvas_scale, grid_size, canvas_scale, 3
    ).transpose(0, 2, 1, 3, 4).reshape(num_pixels, canvas_scale, canvas_scale, 3)

    # Paste each start-image patch into its Hungarian-assigned destination
    # cell to build the final rearranged image.
    output_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for source_index, destination_index in enumerate(matched_order):
        x = (destination_index % grid_size) * canvas_scale
        y = (destination_index // grid_size) * canvas_scale
        output_array[y:y + canvas_scale, x:x + canvas_scale] = texture_patches[source_index]

    # Always output a fixed 1024x1024 PNG regardless of the working canvas size.
    output_image_final = Image.fromarray(output_array).resize(
        (1024, 1024), Image.Resampling.LANCZOS
    )
    output_path = "hungarian_pixel_sorter_final.png"
    output_image_final.save(output_path, format="PNG")
    return output_path


def generate_with_diamond_start(
    end_image,
    grid_size=24,
    position_distance_factor=0.25,
    diamond_size=32,
):
    # Convenience wrapper: derive the start image from the end image itself
    # (via diamond pixelation) instead of requiring a separate uploaded start.
    diamond_start = make_diamond_pixelated_start(end_image, diamond_size)
    output_path = generate_final_image(
        diamond_start,
        end_image,
        grid_size,
        position_distance_factor,
    )
    return diamond_start, output_path


# --- Gradio UI wiring ---
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Source images: "Start" supplies the texture, "End" supplies the
            # target color layout that the start-image cells get matched to.
            start_image_input = gr.Image(label="Start", type="pil", height=180, format="png")
            end_image_input = gr.Image(label="End", type="pil", height=180, format="png")
            grid_size_slider = gr.Slider(
                minimum=8,
                maximum=64,
                value=20,
                step=4,
                label="Grid size (Res)",
            )
            position_distance_factor_slider = gr.Slider(
                minimum=0,
                maximum=2,
                value=0.25,
                step=0.01,
                label="Position distance factor",
            )
            diamond_size_slider = gr.Slider(
                minimum=4,
                maximum=128,
                value=32,
                step=4,
                label="Diamond size (px)",
            )
            run_button = gr.Button("Run", variant="primary")
            diamond_start_button = gr.Button("Run with diamond-pixelated end as start")

        with gr.Column(scale=2):
            # Output is always rendered/saved at a fixed 1024x1024 resolution.
            output_image = gr.Image(
                label="Final chunk positions",
                type="filepath",
                format="png",
                width=1024,
                height=1024,
            )

    # "Run" uses the user-provided Start and End images directly.
    run_button.click(
        fn=generate_final_image,
        inputs=[
            start_image_input,
            end_image_input,
            grid_size_slider,
            position_distance_factor_slider,
        ],
        outputs=output_image,
    )

    # "Run with diamond-pixelated end as start" derives the start image from
    # the end image and also refreshes the Start preview with it.
    diamond_start_button.click(
        fn=generate_with_diamond_start,
        inputs=[
            end_image_input,
            grid_size_slider,
            position_distance_factor_slider,
            diamond_size_slider,
        ],
        outputs=[start_image_input, output_image],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)