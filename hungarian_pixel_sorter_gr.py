"""Create a pixel-morphing animation between two input images.

The script provides a Gradio interface where users upload a start and end
image and choose the grid resolution, number of animation frames, and the
penalty for moving pixels across the grid. It downsamples both images,
matches their pixels with the Hungarian algorithm using perceptual color and
position costs, and smoothly animates the matched pixels from the start layout
to the end layout.

When run, it creates two looping GIF files: a color-block animation and a
patchwork animation that uses tiles from the original start image. Both GIFs
are displayed in the Gradio interface.

Technical details
-----------------
The animation operates on a square grid containing ``grid_size ** 2`` cells.
Each input is first converted to RGB, center-cropped to a square with
``PIL.ImageOps.fit``, and resized with Pillow's Lanczos filter. The resulting
small images provide one RGB color vector per grid cell. The original start
image is also resized to a grid-aligned texture so that each source cell can
later be represented by a small image patch in the patchwork output.

Pixel correspondence is solved as a linear assignment problem. For every
start cell and end cell, the cost is the Euclidean distance between their
OKLab color vectors plus ``position_distance_factor`` times the Euclidean
distance between their normalized grid coordinates. OKLab is used instead of
raw sRGB because its axes are designed to make Euclidean distances more
closely reflect perceived lightness and chromatic differences. This produces
a ``(grid_size ** 2) x (grid_size ** 2)`` dense cost matrix. SciPy's
``linear_sum_assignment`` implements the Hungarian/Jonker-Volgenant-style
minimum-cost assignment solver and returns a one-to-one permutation of the
end cells. The cubic worst-case complexity in the number of cells is why the
images are reduced before matching; increasing the grid resolution increases
both the matrix memory requirement and solver time rapidly.

For each animation frame, matched cell coordinates are interpolated from
their start positions to their assigned end positions. The interpolation
parameter uses the cubic smoothstep function ``t*t*(3-2*t)`` to ease in and
out. Positions are rounded to the output cell lattice, then rendered as
constant-color blocks for the color animation or as extracted texture tiles
for the patchwork animation. Frames are assembled forward and backward to
create a ping-pong loop, with one-second holds on the endpoints.

NumPy performs the vectorized color, coordinate, and frame-array operations.
Pillow creates the frames, enlarges them with nearest-neighbor sampling to
preserve block edges, reduces each animation to a 128-color palette, and
writes the looping GIFs. Gradio's ``Blocks`` layout, image inputs, sliders,
button callback, and filepath outputs provide the interactive front end.
"""

import io

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment


def srgb_to_oklab(rgb_colors):
    """Convert an array of normalized sRGB colors to OKLab coordinates.

    The input is expected to have shape ``(..., 3)`` and values in the
    normalized sRGB range ``[0, 1]``. The output has the same leading shape,
    with the final three values representing OKLab ``L``, ``a``, and ``b``.

    sRGB values are gamma-encoded, so their numeric distances do not directly
    correspond to distances in emitted light. The conversion first removes
    that encoding, transforms linear-light RGB into the intermediate LMS cone
    response space, applies the perceptual cube-root compression used by
    OKLab, and finally rotates the result into the OKLab axes. All operations
    are vectorized with NumPy so the function can process every grid cell in
    one call.
    """
    # Decode gamma-encoded sRGB into linear-light RGB. The piecewise threshold
    # is the standard sRGB transfer function breakpoint.
    linear_rgb = np.where(
        rgb_colors <= 0.04045,
        rgb_colors / 12.92,
        ((rgb_colors + 0.055) / 1.055) ** 2.4,
    )

    # Convert linear RGB into an LMS-like cone response space. The final axis
    # contains R, G, and B, so matrix multiplication applies to every pixel.
    lms = linear_rgb @ np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        dtype=np.float64,
    ).T

    # OKLab uses a cube-root compression of the LMS responses. np.cbrt also
    # handles negative values correctly, which is useful for general inputs.
    lms_cuberoot = np.cbrt(lms)

    # Rotate the compressed cone responses into perceptual lightness and two
    # opponent-color axes. For colors originating in sRGB, L is approximately
    # in [0, 1], while a and b describe green/red and blue/yellow differences.
    return lms_cuberoot @ np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        dtype=np.float64,
    ).T


def generate_anim(start_image, end_image, num_frames=48, grid_size=24, position_distance_factor=0.005):
    if start_image is None or end_image is None:
        raise gr.Error("load 2 imgaes")

    start_image = ImageOps.fit(start_image.convert("RGB"), (min(start_image.size),) * 2, method=Image.LANCZOS)
    end_image = ImageOps.fit(end_image.convert("RGB"), (min(end_image.size),) * 2, method=Image.LANCZOS)
    grid_size = int(grid_size)
    num_frames = int(num_frames)
    position_distance_factor = float(position_distance_factor)
    num_pixels = grid_size * grid_size

    # downscale images as Hungarian algo is order(n^3)
    start_small = start_image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)
    end_small = end_image.convert("RGB").resize((grid_size, grid_size), Image.LANCZOS)

    start_colors = np.asarray(start_small, dtype=np.float64).reshape(-1, 3) / 255.0
    end_colors = np.asarray(end_small, dtype=np.float64).reshape(-1, 3) / 255.0

    # Keep the normalized sRGB values for rendering the color-block output,
    # but compare colors in OKLab for a more perceptually meaningful match.
    # Converting after downsampling keeps the expensive pairwise assignment
    # matrix at the selected grid resolution.
    start_oklab = srgb_to_oklab(start_colors)
    end_oklab = srgb_to_oklab(end_colors)

    # Grid coords (normalized), in raster order
    x_coords, y_coords = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    grid_positions = np.vstack([x_coords.ravel(), y_coords.ravel()]).T

    # Pair pixels by perceptual color while penalizing assignments that move
    # far across the grid. The broadcasted arrays compare every start cell to
    # every end cell, producing the dense color-cost matrix required by the
    # linear assignment solver.
    color_cost = np.linalg.norm(start_oklab[:, None, :] - end_oklab[None, :, :], axis=2)
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

    output_size = (512, 512)
    full_frames = [frame.resize(output_size, Image.NEAREST) for frame in full_frames]
    full_patchwork_frames = [
        frame.resize(output_size, Image.NEAREST) for frame in full_patchwork_frames
    ]
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
            frames_slider = gr.Slider(minimum=12, maximum=120, value=48, step=4, label="no of frames")
            position_distance_factor_slider = gr.Slider(
                minimum=0,
                maximum=0.02,
                value=0.005,
                step=0.001,
                label="Position distance factor",
            )
            morph_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Color block animation", type="filepath", width=512, height=512)
            patchwork_output_image = gr.Image(label="Patchwork animation", type="filepath", width=512, height=512)

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
