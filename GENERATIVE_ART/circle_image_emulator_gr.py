#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Circle-Packing Color Image Emulator

Recreates a target image with a circle-packing style: at each of several
size levels (from a large initial radius down to a small final radius), the
image is sampled at circular sections, the average color of each section is
computed in perceptual LAB space, and a circle of that average color is
painted onto the output. Smaller circles are only added where they still
noticeably improve the local match, so packing naturally concentrates finer
circles in detailed areas while flat areas stay covered by larger circles.
"""

import gradio as gr
import numpy as np
from PIL import Image
import random


# --- Perceptual color space conversion (sRGB <-> CIE LAB, D65 white point) ---

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _f_forward(t):
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)


def _f_inverse(t):
    return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb):
    """Convert an array of sRGB values (0-255, shape (...,3)) to CIE LAB."""
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T
    xyz = xyz / _WHITE
    fx = _f_forward(xyz[..., 0])
    fy = _f_forward(xyz[..., 1])
    fz = _f_forward(xyz[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """Convert an array of CIE LAB values (shape (...,3)) back to sRGB (0-255)."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = np.stack([
        _f_inverse(fx) * _WHITE[0],
        _f_inverse(fy) * _WHITE[1],
        _f_inverse(fz) * _WHITE[2],
    ], axis=-1)
    linear = xyz @ _XYZ_TO_RGB.T
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255)


# --- Image loading / canvas helpers ---

def load_color(image_path, max_dimension):
    """Load an image as RGB and resize so its largest side is max_dimension."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = max_dimension / max(w, h)
    if scale < 1.0:
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def get_background_rgb(target_rgb, mode, rng):
    if mode == "Mean of target":
        return target_rgb.reshape(-1, 3).mean(axis=0)
    if mode == "White":
        return np.array([255.0, 255.0, 255.0])
    if mode == "Black":
        return np.array([0.0, 0.0, 0.0])
    return np.array([rng.uniform(0, 255) for _ in range(3)])  # Random


def build_radius_levels(max_radius, min_radius, step_factor):
    """Build a descending sequence of circle radii from max_radius to min_radius."""
    radii = []
    r = float(max_radius)
    min_radius = float(min_radius)
    while r > min_radius:
        radii.append(r)
        r *= step_factor
    radii.append(min_radius)
    return radii


def draw_circle(canvas, x, y, r, color, width, height):
    """Paint a filled circle of the given color onto canvas, clipped to bounds."""
    x0 = max(0, int(np.floor(x - r)))
    x1 = min(width, int(np.ceil(x + r)) + 1)
    y0 = max(0, int(np.floor(y - r)))
    y1 = min(height, int(np.ceil(y + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    canvas[y0:y1, x0:x1][mask] = color


def circle_slice_and_mask(x, y, r, width, height):
    """Return the bounding-box slice and a circular boolean mask within it."""
    x0 = max(0, int(np.floor(x - r)))
    x1 = min(width, int(np.ceil(x + r)) + 1)
    y0 = max(0, int(np.floor(y - r)))
    y1 = min(height, int(np.ceil(y + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None, None, None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
    return (slice(y0, y1), slice(x0, x1)), mask, mask.sum()


def pack_circles(
    image_path,
    initial_size,
    final_size,
    radius_step_factor,
    spacing_factor,
    jitter_amount,
    detail_threshold,
    working_resolution,
    background_mode,
    seed,
):
    if image_path is None:
        yield None, "Please upload a target image.", None
        return

    rng = random.Random(int(seed))

    orig_img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = orig_img.size
    target_img = load_color(image_path, int(working_resolution))
    width, height = target_img.size
    target_rgb = np.asarray(target_img, dtype=np.float64)
    target_lab = rgb_to_lab(target_rgb)

    background_rgb = get_background_rgb(target_rgb, background_mode, rng)
    canvas = np.tile(background_rgb, (height, width, 1))
    covered = np.zeros((height, width), dtype=bool)  # pixels already claimed by a placed circle

    radii = build_radius_levels(float(initial_size), float(final_size), float(radius_step_factor))
    detail_threshold = float(detail_threshold)
    jitter_amount = float(jitter_amount)
    spacing_factor = float(spacing_factor)

    all_circles = []  # (x_frac, y_frac, radius_frac, color) as fractions of width/height

    for level_index, r in enumerate(radii):
        spacing = max(1.0, r * spacing_factor)
        is_first_level = level_index == 0
        row_y = -r
        row_index = 0
        while row_y < height + r:
            row_offset = (spacing / 2.0) if (row_index % 2 == 1) else 0.0
            col_x = -r + row_offset
            while col_x < width + r:
                x = col_x + rng.uniform(-spacing, spacing) * jitter_amount
                y = row_y + rng.uniform(-spacing, spacing) * jitter_amount

                bbox, mask, count = circle_slice_and_mask(x, y, r, width, height)
                # Skip if this circle would overlap any pixel already claimed by a placed circle
                if bbox is not None and count > 0 and not covered[bbox][mask].any():
                    target_patch = target_lab[bbox][mask]
                    avg_lab = target_patch.mean(axis=0)

                    if is_first_level:
                        should_draw = True
                    else:
                        canvas_patch = canvas[bbox][mask]
                        canvas_patch_lab = rgb_to_lab(canvas_patch)
                        diff = canvas_patch_lab - target_patch
                        local_error = float(np.mean(np.sum(diff * diff, axis=-1)))
                        should_draw = local_error > detail_threshold

                    if should_draw:
                        color_rgb = lab_to_rgb(avg_lab)
                        draw_circle(canvas, x, y, r, color_rgb, width, height)
                        covered[bbox][mask] = True
                        all_circles.append((x / width, y / height, r / min(width, height), color_rgb))

                col_x += spacing
            row_y += spacing
            row_index += 1

        preview = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
        message = (
            f"Level {level_index + 1}/{len(radii)}, radius {r:.2f}px\n"
            f"Circles drawn so far: {len(all_circles)}\n"
            f"Canvas: {width}x{height}"
        )
        yield preview, message, gr.skip()

    final_preview = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")

    # Re-render at the original resolution using the fractional circle positions/radii
    full_canvas = np.tile(background_rgb, (orig_height, orig_width, 1))
    min_dim = min(orig_width, orig_height)
    for x_frac, y_frac, r_frac, color_rgb in all_circles:
        draw_circle(
            full_canvas,
            x_frac * orig_width,
            y_frac * orig_height,
            r_frac * min_dim,
            color_rgb,
            orig_width,
            orig_height,
        )
    full_preview = Image.fromarray(np.clip(full_canvas, 0, 255).astype(np.uint8), mode="RGB")

    final_message = (
        f"Finished all {len(radii)} levels (radius {initial_size} down to {final_size}).\n"
        f"Total circles drawn: {len(all_circles)}, Canvas: {width}x{height}\n"
        f"Full-size render: {orig_width}x{orig_height}"
    )
    yield final_preview, final_message, full_preview


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("Circle-Packing Color Image Emulator")
        gr.Markdown(
            "For each circle size, from the initial (largest) down to the final "
            "(smallest), circular sections of the target image are averaged in "
            "perceptual LAB color space and painted as circles. Smaller circles "
            "are only added where they still meaningfully improve the local match."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Target", type="filepath", format="png", height=400)
                initial_size = gr.Slider(label="Initial (largest) circle size", minimum=5, maximum=150, value=40, step=1)
                final_size = gr.Slider(label="Final (smallest) circle size", minimum=1, maximum=30, value=2, step=1)
                radius_step_factor = gr.Slider(label="Radius step factor (per level)", minimum=0.5, maximum=0.95, value=0.75, step=0.01)

            with gr.Column():
                working_resolution = gr.Slider(label="Working resolution", minimum=32, maximum=400, value=160, step=8)
                spacing_factor = gr.Slider(label="Candidate spacing (x radius)", minimum=1.0, maximum=3.0, value=1.6, step=0.05)
                jitter_amount = gr.Slider(label="Position jitter", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                detail_threshold = gr.Slider(label="Detail threshold (LAB error to refine)", minimum=0.0, maximum=50.0, value=4.0, step=0.5)

            with gr.Column():
                background_mode = gr.Radio(
                    label="Background",
                    choices=["Mean of target", "White", "Black", "Random"],
                    value="Mean of target",
                )
                seed = gr.Number(label="Random seed", value=42, precision=0)
                run_btn = gr.Button("GO!", variant="primary")

        with gr.Row():
            output_image = gr.Image(label="Preview", format="png", height=400, width=400)
            output_message = gr.Textbox(label="Progress", lines=6)
            final_output_image = gr.Image(label="Final Full-Size Render", format="png")

        run_btn.click(
            fn=pack_circles,
            inputs=[
                input_image,
                initial_size,
                final_size,
                radius_step_factor,
                spacing_factor,
                jitter_amount,
                detail_threshold,
                working_resolution,
                background_mode,
                seed,
            ],
            outputs=[output_image, output_message, final_output_image],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(inbrowser=True)
