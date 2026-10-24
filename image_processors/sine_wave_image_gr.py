#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Sine Wave Image Recreator

Recreates an image as horizontal sine waves on a white background.
Waves are spaced at a fixed vertical pixel spacing. Local frequency follows
ink density: dark areas get a high frequency, bright areas a low one.

Modes:
  Black and white      - single layer of black waves from luminance.
  CMYK                 - four wave layers (C, M, Y, K), each at its own angle,
                         combined subtractively like print halftones.
  Line colour from image - luminance-driven black-and-white wave structure,
                         with each line coloured by the local image colour.
"""

import cv2
import gradio as gr
import numpy as np
from PIL import Image

MIN_FREQ = 0.02  # cycles per pixel for the lightest areas
MAX_FREQ = 0.30  # cycles per pixel for the darkest areas

# Subtractive ink colours as RGB (0..1)
INKS = {
    "C": np.array([0.0, 1.0, 1.0], np.float32),
    "M": np.array([1.0, 0.0, 1.0], np.float32),
    "Y": np.array([1.0, 1.0, 0.0], np.float32),
    "K": np.array([0.0, 0.0, 0.0], np.float32),
}


def draw_layer(density, spacing, amplitude, mult, thickness, angle,
               row_offset=0, fade=False):
    """Draw one wave layer. density is 0..1 (1 = most ink). Returns coverage 0..1."""
    h, w = density.shape
    size = int(np.ceil(np.hypot(h, w))) + 2
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    M[0, 2] += size / 2.0 - w / 2.0
    M[1, 2] += size / 2.0 - h / 2.0
    frame = cv2.warpAffine(density, M, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    k = max(3, spacing | 1)
    smooth = cv2.blur(frame, (k, k))

    canvas = np.zeros((size, size), dtype=np.uint8)
    xs = np.arange(size)
    start = spacing // 2 + int(row_offset) % spacing
    for yc in range(start, size, spacing):
        d = smooth[yc, :]
        freq = (MIN_FREQ + d * (MAX_FREQ - MIN_FREQ)) * mult
        # Integrate frequency so the wave stays continuous where it changes
        phase = 2.0 * np.pi * np.cumsum(freq)
        ys = yc + amplitude * np.sin(phase)
        pts = np.round(np.stack([xs, ys], axis=1) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, 255, thickness, cv2.LINE_AA, shift=4)

    cov = canvas.astype(np.float32) / 255.0
    if fade:
        # Suppress lines where there is almost no ink
        cov *= np.clip(smooth / 0.12, 0.0, 1.0)

    inv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(cov, inv, (w, h), flags=cv2.INTER_LINEAR)


def rgb_to_cmyk(rgb):
    k = 1.0 - rgb.max(axis=2)
    denom = np.maximum(1.0 - k, 1e-6)
    c = (1.0 - rgb[..., 0] - k) / denom
    m = (1.0 - rgb[..., 1] - k) / denom
    y = (1.0 - rgb[..., 2] - k) / denom
    return {"C": c, "M": m, "Y": y, "K": k}


def render_waves(image, mode, spacing, amplitude, mult, thickness, invert,
                 ang_c, ang_m, ang_y, ang_k):
    if image is None:
        return None, "Please load an image."

    spacing = int(spacing)
    thickness = int(thickness)
    rgb = np.asarray(Image.fromarray(image).convert("RGB"), dtype=np.float32) / 255.0
    if invert:
        rgb = 1.0 - rgb
    h, w = rgb.shape[:2]

    if mode == "CMYK":
        channels = rgb_to_cmyk(rgb)
        angles = {"C": ang_c, "M": ang_m, "Y": ang_y, "K": ang_k}
        out = np.ones((h, w, 3), np.float32)
        for i, name in enumerate("CMYK"):
            d = np.clip(channels[name], 0, 1).astype(np.float32)
            cov = draw_layer(d, spacing, amplitude, mult, thickness,
                             angles[name], row_offset=spacing * i // 4, fade=True)
            out *= 1.0 - cov[..., None] * (1.0 - INKS[name])
    else:
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        cov = draw_layer((1.0 - gray).astype(np.float32), spacing, amplitude,
                         mult, thickness, 0.0)
        if mode == "Line colour from image":
            k = max(3, spacing | 1)
            col = cv2.blur(rgb, (k, k))
            # Normalise so lines stay visible on white while keeping hue
            col = col * np.minimum(1.0, 0.6 / np.maximum(col.max(axis=2, keepdims=True), 1e-6))
            out = 1.0 - cov[..., None] * (1.0 - col)
        else:
            out = np.repeat(1.0 - cov[..., None], 3, axis=2)

    result = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    msg = (f"Mode {mode}: {w}x{h} canvas, spacing {spacing}px, "
           f"amplitude {amplitude}px, frequency multiplier {mult:.2f}.")
    return result, msg


def create_demo():
    with gr.Blocks(title="Sine Wave Image Recreator") as demo:
        gr.Markdown("# Sine Wave Image Recreator")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="Input image", type="numpy")
                mode = gr.Radio(["Black and white", "CMYK", "Line colour from image"],
                                value="Black and white", label="Mode")
                spacing = gr.Slider(2, 128, value=16, step=1, label="Vertical spacing (px)")
                amplitude = gr.Slider(0, 64, value=6, step=0.5, label="Amplitude (px)")
                mult = gr.Slider(0.1, 4.0, value=1.0, step=0.05,
                                 label="Overall brightness (frequency multiplier)")
                thickness = gr.Slider(1, 6, value=1, step=1, label="Line thickness (px)")
                invert = gr.Checkbox(value=False, label="Invert image")
                with gr.Accordion("CMYK wave angles (degrees)", open=False):
                    ang_c = gr.Slider(-90, 90, value=15, step=1, label="Cyan")
                    ang_m = gr.Slider(-90, 90, value=75, step=1, label="Magenta")
                    ang_y = gr.Slider(-90, 90, value=0, step=1, label="Yellow")
                    ang_k = gr.Slider(-90, 90, value=45, step=1, label="Black")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Image(label="Sine wave output", type="numpy", format="png")
                msg = gr.Textbox(label="Info", interactive=False)

        inputs = [inp, mode, spacing, amplitude, mult, thickness, invert,
                  ang_c, ang_m, ang_y, ang_k]
        btn.click(render_waves, inputs, [out, msg])
    return demo


if __name__ == "__main__":
    create_demo().launch(inbrowser=True)
