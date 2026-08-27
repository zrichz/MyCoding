#!/home/rich/MyCoding/venvMyCoding/bin/python
"""Gradio app: 2x expansion with magenta fill and no interpolation.

Rules:
- Horizontal: even output columns copy source, odd columns are magenta.
- Vertical: even output rows copy source, odd rows are magenta.
- Both: source lands on even rows and even columns; all other pixels are magenta.
"""

import gradio as gr
import numpy as np
from PIL import Image
import tempfile


MAGENTA_RGB = np.array([255, 0, 255], dtype=np.uint8)


def expand_with_magenta(image: Image.Image, mode: str):
    """Return a 2x-expanded image using strict copy-and-magenta fill."""
    if image is None:
        return None, None, "Please upload an image file"

    rgb_image = image.convert("RGB")
    src = np.array(rgb_image, dtype=np.uint8)

    height, width, _ = src.shape

    if mode == "Horizontal 2x":
        out = np.empty((height, width * 2, 3), dtype=np.uint8)
        out[:, :, :] = MAGENTA_RGB
        out[:, 0::2, :] = src
    elif mode == "Vertical 2x":
        out = np.empty((height * 2, width, 3), dtype=np.uint8)
        out[:, :, :] = MAGENTA_RGB
        out[0::2, :, :] = src
    else:
        out = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
        out[:, :, :] = MAGENTA_RGB
        out[0::2, 0::2, :] = src

    result = Image.fromarray(out, mode="RGB")

    png_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png_path = png_file.name
    png_file.close()
    result.save(png_path, format="PNG")

    out_h, out_w, _ = out.shape
    message = f"Processed {width}x{height} to {out_w}x{out_h} and saved PNG output"
    return result, png_path, message


with gr.Blocks(title="2x Magenta Fill Expander") as demo:
    gr.Markdown("## 2x Magenta Fill Expander")
    gr.Markdown(
        "Upload an image and choose expansion mode. "
        "The app uses direct copy plus pure magenta fill with no interpolation."
    )

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image", sources=["upload"])
        output_image = gr.Image(type="pil", label="Output Image")

    mode_choice = gr.Radio(
        choices=["Horizontal 2x", "Vertical 2x", "Horizontal and Vertical 2x"],
        value="Horizontal 2x",
        label="Expansion Mode",
    )

    png_output = gr.File(label="Download PNG", file_types=[".png"])

    status_text = gr.Textbox(label="Status", interactive=False)
    run_button = gr.Button("Process")

    run_button.click(
        fn=expand_with_magenta,
        inputs=[input_image, mode_choice],
        outputs=[output_image, png_output, status_text],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
