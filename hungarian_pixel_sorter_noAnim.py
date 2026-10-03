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
5. Native-resolution start-image tiles are rearranged and saved as a PNG.

A Gradio UI wraps this logic, with options to use a generated cross-hatch
image or turn the end image into a diamond-pixelated start image.
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


def make_builtin_crosshatch_start():
    tile_count = 64
    tile_size = 16
    max_total_lines = 32
    supersample = 16
    image = np.full((tile_count * tile_size, tile_count * tile_size, 3), 18, dtype=np.uint8)
    coordinates = (
        (np.arange(tile_size * supersample, dtype=np.float32) + 0.5)
        / supersample
        - tile_size / 2
    )
    x_coords, y_coords = np.meshgrid(coordinates, coordinates)
    random_generator = np.random.default_rng(0)

    for row in range(tile_count):
        for col in range(tile_count):
            diagonal_progress = (row + col) / (2 * (tile_count - 1))
            lines_per_direction = round((max_total_lines // 2) * diagonal_progress)
            if lines_per_direction == 0:
                continue

            angle = random_generator.uniform(0, np.pi)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            first_projection = -x_coords * sin_angle + y_coords * cos_angle
            second_projection = x_coords * cos_angle + y_coords * sin_angle
            first_extent = tile_size / 2 * (abs(sin_angle) + abs(cos_angle))
            second_extent = tile_size / 2 * (abs(cos_angle) + abs(sin_angle))

            def near_parallel_line(projection, extent):
                spacing = 2 * extent / lines_per_direction
                distance = np.abs(
                    np.mod(projection + extent + spacing / 2, spacing) - spacing / 2
                )
                half_width = min(0.25, spacing / 4)
                return distance <= half_width

            line_mask = near_parallel_line(first_projection, first_extent)
            line_mask |= near_parallel_line(second_projection, second_extent)
            coverage = line_mask.reshape(
                tile_size, supersample, tile_size, supersample
            ).mean(axis=(1, 3))
            tile = image[
                row * tile_size:(row + 1) * tile_size,
                col * tile_size:(col + 1) * tile_size,
            ]
            tile[:] = np.rint(18 + coverage[:, :, None] * 220).astype(np.uint8)

    return Image.fromarray(image)


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


def generate_final_image(
    start_image,
    end_image,
    grid_size=24,
    position_distance_factor=0.25,
    use_builtin_start=False,
):
    # Core algorithm: match start-image grid cells to end-image grid cells by
    # color similarity (with a positional penalty), then rebuild the image by
    # placing each start-image texture patch at its matched destination cell.
    if end_image is None or (start_image is None and not use_builtin_start):
        raise gr.Error("Load an end image and either a start image or the built-in image.")

    if use_builtin_start:
        start_image = make_builtin_crosshatch_start()
    start_image = center_crop_square(start_image)
    end_image = center_crop_square(end_image)

    grid_size = int(grid_size)
    position_distance_factor = float(position_distance_factor)
    num_pixels = grid_size * grid_size

    tile_size = start_image.width // grid_size
    if tile_size < 1:
        raise gr.Error("The start image must be at least as large as the grid size.")
    image_size = tile_size * grid_size
    crop_left = (start_image.width - image_size) // 2
    crop_top = (start_image.height - image_size) // 2
    start_image = start_image.crop(
        (crop_left, crop_top, crop_left + image_size, crop_top + image_size)
    )

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

    # Slice native-resolution tiles from the grid-aligned start image.
    texture_patches = np.asarray(start_image.convert("RGB")).reshape(
        grid_size, tile_size, grid_size, tile_size, 3
    ).transpose(0, 2, 1, 3, 4).reshape(num_pixels, tile_size, tile_size, 3)

    # Paste each start-image patch into its Hungarian-assigned destination
    # cell to build the final rearranged image.
    output_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for source_index, destination_index in enumerate(matched_order):
        x = (destination_index % grid_size) * tile_size
        y = (destination_index // grid_size) * tile_size
        output_array[y:y + tile_size, x:x + tile_size] = texture_patches[source_index]

    output_image_final = Image.fromarray(output_array)
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
            builtin_start_checkbox = gr.Checkbox(
                value=False,
                label="Use built-in cross-hatch start image",
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
            builtin_start_preview = gr.Image(
                value=make_builtin_crosshatch_start(),
                label="Built-in cross-hatch source (debug preview)",
                type="pil",
                format="png",
                width=512,
                height=512,
            )
            # The saved image keeps the native dimensions of the extracted tiles.
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
            builtin_start_checkbox,
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