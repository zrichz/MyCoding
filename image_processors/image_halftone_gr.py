"""
Halftone Pattern Generator (Gradio)
Convert an image into a classic dot-based halftone pattern
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image


def darkness_to_radius(darkness, sample_size, contrast):
    """Map a 0-255 darkness value to a dot radius, area-correct and contrast-adjusted"""
    ratio = darkness / 255.0
    # Apply gamma so midtones grow faster (contrast < 1) or shrink (contrast > 1)
    ratio = ratio ** contrast
    # Dot area (not radius) should scale linearly with darkness, so use sqrt.
    # max_radius allows slight overlap into neighboring cells so full darkness prints solid black.
    max_radius = sample_size * 0.75
    return int(np.sqrt(ratio) * max_radius)


def dot_halftone_channel(channel, sample_size, invert, angle, contrast):
    """Generate a halftone pattern for a single grayscale channel"""
    h, w = channel.shape
    output = np.ones((h, w), dtype=np.uint8) * 255

    # Rotate the sampling grid by working in a larger, rotated canvas
    if angle != 0:
        diagonal = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
        pad_h = (diagonal - h) // 2 + sample_size
        pad_w = (diagonal - w) // 2 + sample_size
        padded = cv2.copyMakeBorder(
            channel, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE
        )
        rot_h, rot_w = padded.shape
        rot_matrix = cv2.getRotationMatrix2D((rot_w / 2, rot_h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            padded, rot_matrix, (rot_w, rot_h), borderMode=cv2.BORDER_REPLICATE
        )

        rot_output = np.ones((rot_h, rot_w), dtype=np.uint8) * 255

        for y in range(0, rot_h, sample_size):
            for x in range(0, rot_w, sample_size):
                block = rotated[y:y + sample_size, x:x + sample_size]
                if block.size == 0:
                    continue
                mean_intensity = np.mean(block)
                darkness = mean_intensity if invert else 255 - mean_intensity
                radius = darkness_to_radius(darkness, sample_size, contrast)
                if radius > 0:
                    center_x = x + sample_size // 2
                    center_y = y + sample_size // 2
                    cv2.circle(rot_output, (center_x, center_y), radius, 0, -1)

        # Rotate back and crop to the original size
        inv_matrix = cv2.getRotationMatrix2D((rot_w / 2, rot_h / 2), -angle, 1.0)
        unrotated = cv2.warpAffine(
            rot_output, inv_matrix, (rot_w, rot_h), borderValue=255
        )
        start_y = (rot_h - h) // 2
        start_x = (rot_w - w) // 2
        output = unrotated[start_y:start_y + h, start_x:start_x + w]
        return output

    for y in range(0, h, sample_size):
        for x in range(0, w, sample_size):
            block = channel[y:y + sample_size, x:x + sample_size]
            mean_intensity = np.mean(block)
            darkness = mean_intensity if invert else 255 - mean_intensity
            radius = darkness_to_radius(darkness, sample_size, contrast)

            center_x = x + sample_size // 2
            center_y = y + sample_size // 2
            if radius > 0:
                cv2.circle(output, (center_x, center_y), radius, 0, -1)

    return output


def generate_halftone(input_image, sample_size, invert, color_mode, angle, contrast):
    """Generate a halftone pattern from an uploaded image"""
    if input_image is None:
        return None, "load an image first"

    try:
        sample_size = max(2, int(sample_size))
        img_array = np.array(input_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if color_mode:
            # CMY-style color halftone: process each channel with a different screen angle
            b, g, r = cv2.split(img_bgr)
            channel_angles = [angle, angle + 15, angle + 30]

            cyan = dot_halftone_channel(255 - r, sample_size, not invert, channel_angles[0], contrast)
            magenta = dot_halftone_channel(255 - g, sample_size, not invert, channel_angles[1], contrast)
            yellow = dot_halftone_channel(255 - b, sample_size, not invert, channel_angles[2], contrast)

            # Subtractive combine: each channel's "ink" darkens the final color
            ink_c = (255 - cyan).astype(np.int32)
            ink_m = (255 - magenta).astype(np.int32)
            ink_y = (255 - yellow).astype(np.int32)

            out_r = np.clip(255 - ink_c, 0, 255).astype(np.uint8)
            out_g = np.clip(255 - ink_m, 0, 255).astype(np.uint8)
            out_b = np.clip(255 - ink_y, 0, 255).astype(np.uint8)

            output_rgb = cv2.merge([out_r, out_g, out_b])
            output_image = Image.fromarray(output_rgb)
        else:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            output = dot_halftone_channel(gray, sample_size, invert, angle, contrast)
            output_image = Image.fromarray(output)

        return output_image, "Halftone pattern generated."

    except Exception as e:
        return None, f"Error generating halftone: {e}"


with gr.Blocks(theme=gr.themes.Soft(), title="Halftone Pattern Generator") as demo:
    gr.Markdown("# Halftone Pattern Generator")
    gr.Markdown("Convert an image into a classic dot-based halftone pattern.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            sample_size = gr.Slider(
                minimum=2, maximum=40, value=10, step=1,
                label="Dot Cell Size (pixels)"
            )
            angle = gr.Slider(
                minimum=0, maximum=90, value=15, step=1,
                label="Screen Angle (degrees)"
            )
            contrast = gr.Slider(
                minimum=0.3, maximum=2.0, value=0.7, step=0.05,
                label="Contrast / Dot Gain (lower = darker, bigger dots)"
            )
            invert = gr.Checkbox(label="Invert (light areas get larger dots)", value=False)
            color_mode = gr.Checkbox(label="Color halftone (CMY channels)", value=False)
            generate_btn = gr.Button("Generate Halftone", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Halftone Result", type="pil")
            status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_halftone,
        inputs=[input_image, sample_size, invert, color_mode, angle, contrast],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
