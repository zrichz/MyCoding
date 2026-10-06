import numpy as np
import gradio as gr
from PIL import Image
from scipy.ndimage import gaussian_filter


def to_ycc(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([y, (b - y) * 0.564, (r - y) * 0.713], axis=-1)


def to_rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    r = y + cr / 0.713
    b = y + cb / 0.564
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.stack([r, g, b], axis=-1)


def to_u8(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def remove_mottling(image, radius, chroma_strength, luma_strength, scale_ratio):
    if image is None:
        return None, None, None

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    ycc = to_ycc(rgb)

    # Frequency split: Gaussian low band, residual high band
    low = gaussian_filter(ycc, sigma=(radius, radius, 0), mode="reflect")
    high = ycc - low

    # Mottling is the mid band between radius and radius*ratio; very-large-scale colour is kept
    very_low = gaussian_filter(ycc, sigma=(radius * scale_ratio, radius * scale_ratio, 0), mode="reflect")
    strength = np.array([luma_strength, chroma_strength, chroma_strength], dtype=np.float32)
    new_low = very_low + (low - very_low) * (1 - strength)

    result = to_rgb(new_low + high)

    low_vis = to_rgb(low)
    high_vis = to_rgb(np.stack([high[..., 0] + 0.5, high[..., 1], high[..., 2]], axis=-1))
    return to_u8(result), to_u8(low_vis), to_u8(high_vis)


with gr.Blocks(title="Color Mottling Remover") as demo:
    gr.Markdown(
        "Splits the image into low and high frequency bands in YCbCr space. "
        "Color blotches in the low band are flattened, fine detail is kept. "
        "Larger radius treats larger blotches as low frequency."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input", format="png")
            radius = gr.Slider(1, 200, value=20, step=0.5, label="Split radius (Gaussian sigma, pixels)")
            chroma = gr.Slider(0, 1, value=1.0, step=0.01, label="Color mottling removal strength")
            luma = gr.Slider(0, 1, value=0.0, step=0.01, label="Brightness blotch removal strength")
            ratio = gr.Slider(1.5, 20, value=4, step=0.5, label="Outer scale ratio (colour larger than radius x ratio is kept)")
            btn = gr.Button("Apply")
        with gr.Column():
            out = gr.Image(type="pil", label="Result", format="png")
            with gr.Row():
                low_out = gr.Image(type="pil", label="Low frequency band", format="png")
                high_out = gr.Image(type="pil", label="High frequency band", format="png")

    inputs = [inp, radius, chroma, luma, ratio]
    outputs = [out, low_out, high_out]
    btn.click(remove_mottling, inputs, outputs)
    for c in (radius, chroma, luma, ratio):
        c.release(remove_mottling, inputs, outputs)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
