#!/home/rich/MyCoding/venvMyCoding/bin/python
"""Slide a small circular-windowed FFT across an image to map local radial frequency content."""

import gradio as gr
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


MAX_SIDE_LENGTH = 1024


def resize_if_needed(image_array):
    """Downscale large images so the sliding FFT stays fast."""
    height, width = image_array.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_SIDE_LENGTH:
        return image_array
    scale = MAX_SIDE_LENGTH / longest_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    pil_image = Image.fromarray(image_array)
    return np.asarray(pil_image.resize(new_size, Image.LANCZOS))


def make_circular_window(size):
    """Build an isotropic (circular) Hann-style taper for the sliding window patch."""
    center = (size - 1) / 2.0
    y_indices, x_indices = np.indices((size, size))
    distances = np.sqrt((y_indices - center) ** 2 + (x_indices - center) ** 2)
    radius = center if center > 0 else 1.0
    normalized_distance = np.clip(distances / radius, 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * normalized_distance))


def make_gaussian_falloff(size, sigma_fraction):
    """Build a radial Gaussian weighting mask over a patch's FFT magnitude."""
    center = (size - 1) / 2.0
    y_indices, x_indices = np.indices((size, size))
    distances = np.sqrt((y_indices - center) ** 2 + (x_indices - center) ** 2)
    max_radius = center if center > 0 else 1.0
    sigma = max(sigma_fraction, 1e-4) * max_radius
    return np.exp(-(distances ** 2) / (2 * sigma ** 2))


def local_radial_frequency_map(
    image, window_size, stride, gaussian_sigma_fraction, exclude_dc, log_scale
):
    """Slide a circular-windowed FFT patch across the image and map radial frequency energy."""
    if image is None:
        raise gr.Error("Please upload an image.")

    window_size = int(window_size)
    stride = int(stride)

    image_array = np.asarray(image)
    original_height, original_width = image_array.shape[:2]
    if image_array.ndim == 3:
        grayscale = np.dot(image_array[:, :, :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    else:
        grayscale = image_array.astype(np.float64)

    grayscale = resize_if_needed(grayscale.astype(np.float32)).astype(np.float64)
    height, width = grayscale.shape

    if window_size > height or window_size > width:
        raise gr.Error("Window size is larger than the (resized) image dimensions.")

    circular_window = make_circular_window(window_size)
    falloff_mask = make_gaussian_falloff(window_size, gaussian_sigma_fraction)
    if exclude_dc:
        center = window_size // 2
        falloff_mask[center, center] = 0.0

    patches = sliding_window_view(grayscale, (window_size, window_size))
    patches = patches[::stride, ::stride]

    windowed_patches = patches * circular_window
    spectra = np.fft.fftshift(np.fft.fft2(windowed_patches, axes=(-2, -1)), axes=(-2, -1))
    magnitude = np.abs(spectra)

    if log_scale:
        magnitude = np.log1p(magnitude)

    weighted_energy = np.sum(magnitude * falloff_mask, axis=(-2, -1))

    energy_min, energy_max = weighted_energy.min(), weighted_energy.max()
    if energy_max > energy_min:
        normalized = (weighted_energy - energy_min) / (energy_max - energy_min)
    else:
        normalized = np.zeros_like(weighted_energy)

    output_small = (normalized * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_small, mode="L").resize(
        (original_width, original_height), Image.NEAREST
    )
    return output_image


with gr.Blocks(title="Local Radial Frequency Map") as demo:
    gr.Markdown("# Local Radial Frequency Map")
    gr.Markdown(
        "Slides a small circularly-windowed FFT patch across the image and maps the "
        "radial frequency magnitude found at each position, producing a spatially "
        "varying greyscale map of local frequency content."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            window_size_slider = gr.Slider(
                minimum=4, maximum=32, value=8, step=1,
                label="Sliding Window Size (pixels)"
            )
            stride_slider = gr.Slider(
                minimum=1, maximum=16, value=2, step=1,
                label="Stride (step between window positions)"
            )
            gaussian_sigma_slider = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.5, step=0.05,
                label="Gaussian Falloff Sigma (fraction of window radius)"
            )
            exclude_dc_checkbox = gr.Checkbox(value=True, label="Exclude DC (mean brightness) Component")
            log_scale_checkbox = gr.Checkbox(value=True, label="Use Log Scale")
            analyze_button = gr.Button("Analyze")

        with gr.Column():
            output_image = gr.Image(label="Local Radial Frequency Map", type="pil")

    analyze_inputs = [
        input_image,
        window_size_slider,
        stride_slider,
        gaussian_sigma_slider,
        exclude_dc_checkbox,
        log_scale_checkbox,
    ]

    analyze_button.click(fn=local_radial_frequency_map, inputs=analyze_inputs, outputs=output_image)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
