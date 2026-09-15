import io

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


def generate_anim(start_image, end_image, num_frames=96, grid_size=24):
    if start_image is None or end_image is None:
        raise gr.Error("load 2 imgaes")

    grid_size = int(grid_size)
    num_frames = int(num_frames)
    num_pixels = grid_size * grid_size

    # downscale images as Hungarian algo is order(n^3)
    start_small = start_image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)
    end_small = end_image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)

    start_colors = np.asarray(start_small, dtype=np.float64).reshape(-1, 3) / 255.0
    end_colors = np.asarray(end_small, dtype=np.float64).reshape(-1, 3) / 255.0

    # Grid coords (normalized), in raster order
    x_coords, y_coords = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    grid_positions = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

    # HUNGARIAN ALGORITHM: pair each start pixel with end pixel that minimizes total color-dist (no colors are changed, only positions)
    cost_matrix = np.linalg.norm(start_colors[:, None, :] - end_colors[None, :, :], axis=2)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_order = col_indices[np.argsort(row_indices)]

    start_positions = grid_positions
    final_positions = grid_positions[matched_order]

    frames = []
    canvas_scale = 16
    img_dim = grid_size * canvas_scale

    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1)
        t_smooth = t*t * (3-2*t)  # smoothstep

        current_positions = (1 - t_smooth) * start_positions + t_smooth * final_positions

        frame_img = Image.new("RGB", (img_dim, img_dim), "black")
        draw = ImageDraw.Draw(frame_img)

        for i in range(num_pixels):
            x = current_positions[i, 0] * (grid_size - 1) * canvas_scale
            y = current_positions[i, 1] * (grid_size - 1) * canvas_scale
            color_tuple = tuple((start_colors[i] * 255).astype(int))
            draw.rectangle(
                [x, y, x + canvas_scale - 1, y + canvas_scale - 1],
                fill=color_tuple,
            )

        frames.append(frame_img)

    # Hold first and last frames for 1 sec, ping-pong
    frame_ms = 100
    hold_ms = 1000
    forward_durations = [frame_ms] * num_frames
    forward_durations[0] = hold_ms
    forward_durations[-1] = hold_ms

    reverse_frames = frames[-2:0:-1] # exclude first and last frames to avoid duplication
    reverse_durations = [frame_ms] * len(reverse_frames)

    full_frames = frames + reverse_frames # combine fwd+rev frames
    full_durations = forward_durations + reverse_durations

    gif_bytes = io.BytesIO()
    full_frames[0].save(gif_bytes,format="GIF",save_all=True,
        append_images=full_frames[1:],
        duration=full_durations,
        loop=0,
    )
    gif_bytes.seek(0)

    output_path = "hungarian_morph.gif"
    with open(output_path, "wb") as f:
        f.write(gif_bytes.read())

    return output_path


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            start_image_input = gr.Image(label="Start", type="pil", height=180)
            end_image_input = gr.Image(label="End", type="pil", height=180)
            grid_size_slider = gr.Slider(minimum=8, maximum=64, value=20, step=4, label="Grid size (Res)")
            frames_slider = gr.Slider(minimum=12, maximum=120, value=96, step=4, label="no of frames")
            morph_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2):
            output_image = gr.Image(label="output", type="filepath")

    morph_btn.click(
        fn=generate_anim,
        inputs=[start_image_input, end_image_input, frames_slider, grid_size_slider],
        outputs=[output_image],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
