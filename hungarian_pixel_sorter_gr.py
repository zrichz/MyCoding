import io

import gradio as gr
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def generate_anim(start_image, end_image, num_frames=96, grid_size=24, position_distance_factor=0.25):
    if start_image is None or end_image is None:
        raise gr.Error("load 2 imgaes")

    grid_size = int(grid_size)
    num_frames = int(num_frames)
    position_distance_factor = float(position_distance_factor)
    num_pixels = grid_size * grid_size

    # downscale images as Hungarian algo is order(n^3)
    start_small = start_image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)
    end_small = end_image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)

    start_colors = np.asarray(start_small, dtype=np.float64).reshape(-1, 3) / 255.0
    end_colors = np.asarray(end_small, dtype=np.float64).reshape(-1, 3) / 255.0

    # Grid coords (normalized), in raster order
    x_coords, y_coords = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    grid_positions = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

    # Pair pixels by color while penalizing assignments that move far across the grid.
    color_cost = np.linalg.norm(start_colors[:, None, :] - end_colors[None, :, :], axis=2)
    position_cost = np.linalg.norm(grid_positions[:, None, :] - grid_positions[None, :, :], axis=2)
    cost_matrix = color_cost + position_distance_factor * position_cost
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_order = col_indices[np.argsort(row_indices)]

    start_positions = grid_positions
    final_positions = grid_positions[matched_order]

    color_frames = []
    patchwork_frames = []
    canvas_scale = min(16, max(1, 640 // grid_size))
    img_dim = grid_size * canvas_scale
    start_texture = start_image.convert("RGB").resize((img_dim, img_dim), Image.LANCZOS)
    texture_patches = np.asarray(start_texture).reshape(
        grid_size, canvas_scale, grid_size, canvas_scale, 3
    ).transpose(0, 2, 1, 3, 4).reshape(num_pixels, canvas_scale, canvas_scale, 3)
    color_tiles = (start_colors * 255).astype(np.uint8)

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        t_smooth = t*t * (3-2*t)  # smoothstep

        current_positions = (1 - t_smooth) * start_positions + t_smooth * final_positions

        pixel_positions = np.rint(current_positions * (grid_size - 1) * canvas_scale).astype(int)
        color_frame_array = np.zeros((img_dim, img_dim, 3), dtype=np.uint8)
        patchwork_frame_array = np.zeros((img_dim, img_dim, 3), dtype=np.uint8)

        for i in range(num_pixels):
            x, y = pixel_positions[i]
            color_frame_array[y:y + canvas_scale, x:x + canvas_scale] = color_tiles[i]
            patchwork_frame_array[y:y + canvas_scale, x:x + canvas_scale] = texture_patches[i]

        color_frames.append(Image.fromarray(color_frame_array))
        patchwork_frames.append(Image.fromarray(patchwork_frame_array))

    # Hold first and last frames for 1 sec, ping-pong
    frame_ms = 100
    hold_ms = 1000
    forward_durations = [frame_ms] * num_frames
    forward_durations[0] = hold_ms
    forward_durations[-1] = hold_ms

    reverse_frames = color_frames[-2:0:-1] # exclude first and last frames to avoid duplication
    reverse_patchwork_frames = patchwork_frames[-2:0:-1]
    reverse_durations = [frame_ms] * len(reverse_frames)

    full_frames = color_frames + reverse_frames # combine fwd+rev frames
    full_patchwork_frames = patchwork_frames + reverse_patchwork_frames
    full_durations = forward_durations + reverse_durations

    color_palette = color_frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    patchwork_palette = patchwork_frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    full_frames = [frame.quantize(palette=color_palette, dither=Image.Dither.NONE) for frame in full_frames]
    full_patchwork_frames = [
        frame.quantize(palette=patchwork_palette, dither=Image.Dither.NONE)
        for frame in full_patchwork_frames
    ]

    output_path = "hungarian_morph.gif"
    full_frames[0].save(output_path, format="GIF", save_all=True,
        append_images=full_frames[1:],
        duration=full_durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    patchwork_output_path = "hungarian_patchwork_morph.gif"
    full_patchwork_frames[0].save(patchwork_output_path, format="GIF", save_all=True,
        append_images=full_patchwork_frames[1:],
        duration=full_durations,
        loop=0,
        optimize=True,
        disposal=2,
    )

    return output_path, patchwork_output_path


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            start_image_input = gr.Image(label="Start", type="pil", height=180)
            end_image_input = gr.Image(label="End", type="pil", height=180)
            grid_size_slider = gr.Slider(minimum=8, maximum=64, value=20, step=4, label="Grid size (Res)")
            frames_slider = gr.Slider(minimum=12, maximum=120, value=96, step=4, label="no of frames")
            position_distance_factor_slider = gr.Slider(
                minimum=0,
                maximum=2,
                value=0.25,
                step=0.01,
                label="Position distance factor",
            )
            morph_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Color block animation", type="filepath")
            patchwork_output_image = gr.Image(label="Patchwork animation", type="filepath")

    morph_btn.click(
        fn=generate_anim,
        inputs=[
            start_image_input,
            end_image_input,
            frames_slider,
            grid_size_slider,
            position_distance_factor_slider,
        ],
        outputs=[output_image, patchwork_output_image],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
