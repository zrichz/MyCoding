import gradio as gr
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def center_crop_square(image):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def make_diamond_pixelated_start(image, diamond_size=32):
    if image is None:
        raise gr.Error("Load an end image first.")

    image = center_crop_square(image).convert("RGB")
    pixels = np.asarray(image, dtype=np.uint8)
    height, width = pixels.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    diagonal_a = np.floor((x_coords + y_coords) / diamond_size + 0.5).astype(np.int32)
    diagonal_b = np.floor((x_coords - y_coords) / diamond_size + 0.5).astype(np.int32)
    cell_ids, inverse = np.unique(
        np.stack((diagonal_a, diagonal_b), axis=-1).reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    del cell_ids

    flattened_pixels = pixels.reshape(-1, 3)
    cell_counts = np.bincount(inverse)
    pixelated = np.empty_like(flattened_pixels)
    for channel in range(3):
        channel_totals = np.bincount(inverse, weights=flattened_pixels[:, channel])
        pixelated[:, channel] = np.rint(channel_totals[inverse] / cell_counts[inverse])

    return Image.fromarray(pixelated.reshape(height, width, 3))


def generate_final_image(start_image, end_image, grid_size=24, position_distance_factor=0.25):
    if start_image is None or end_image is None:
        raise gr.Error("Load both images.")

    start_image = center_crop_square(start_image)
    end_image = center_crop_square(end_image)

    grid_size = int(grid_size)
    position_distance_factor = float(position_distance_factor)
    num_pixels = grid_size * grid_size

    start_small = start_image.convert("RGB").resize(
        (grid_size, grid_size), Image.Resampling.LANCZOS
    )
    end_small = end_image.convert("RGB").resize(
        (grid_size, grid_size), Image.Resampling.LANCZOS
    )

    start_colors = np.asarray(start_small, dtype=np.float64).reshape(-1, 3) / 255.0
    end_colors = np.asarray(end_small, dtype=np.float64).reshape(-1, 3) / 255.0

    x_coords, y_coords = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size),
    )
    grid_positions = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

    color_cost = np.linalg.norm(start_colors[:, None, :] - end_colors[None, :, :], axis=2)
    position_cost = np.linalg.norm(grid_positions[:, None, :] - grid_positions[None, :, :], axis=2)
    cost_matrix = color_cost + position_distance_factor * position_cost
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_order = col_indices[np.argsort(row_indices)]

    canvas_scale = min(16, max(1, 640 // grid_size))
    image_size = grid_size * canvas_scale
    start_texture = start_image.convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.LANCZOS,
    )
    texture_patches = np.asarray(start_texture).reshape(
        grid_size, canvas_scale, grid_size, canvas_scale, 3
    ).transpose(0, 2, 1, 3, 4).reshape(num_pixels, canvas_scale, canvas_scale, 3)

    output_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for source_index, destination_index in enumerate(matched_order):
        x = (destination_index % grid_size) * canvas_scale
        y = (destination_index // grid_size) * canvas_scale
        output_array[y:y + canvas_scale, x:x + canvas_scale] = texture_patches[source_index]

    output_path = "hungarian_pixel_sorter_final.png"
    Image.fromarray(output_array).save(output_path, format="PNG")
    return output_path


def generate_with_diamond_start(
    end_image,
    grid_size=24,
    position_distance_factor=0.25,
    diamond_size=32,
):
    diamond_start = make_diamond_pixelated_start(end_image, diamond_size)
    output_path = generate_final_image(
        diamond_start,
        end_image,
        grid_size,
        position_distance_factor,
    )
    return diamond_start, output_path


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
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
            output_image = gr.Image(
                label="Final chunk positions",
                type="filepath",
                format="png",
            )

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