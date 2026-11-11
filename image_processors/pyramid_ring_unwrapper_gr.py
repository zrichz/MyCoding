import numpy as np
import gradio as gr
from PIL import Image

GREY = 128
MAX_WIDTH = 4096


def max_radius(w, h):
    # Smallest integer radius whose circle encloses the corner pixels
    return int(np.ceil(np.hypot((h - 1) / 2.0, (w - 1) / 2.0)))


def mosaic_rows(img, n):
    """Average each row in horizontal blocks of n pixels."""
    size = img.shape[1]
    starts = np.arange(0, size, n)
    counts = np.diff(np.append(starts, size))
    sums = np.add.reduceat(img.astype(np.float32), starts, axis=1)
    means = np.rint(sums / counts[None, :, None]).astype(np.uint8)
    return np.repeat(means, counts, axis=1)


def make_pyramid(image, use_mosaic, mosaic_exp):
    if image is None:
        return None, "Please upload an image."

    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    max_r = max_radius(w, h)
    size = max(1, min(MAX_WIDTH, int(round(2 * np.pi * max_r))))

    out = np.zeros((size, size, 3), dtype=np.uint8)

    for i in range(size):
        r = max_r * i / (size - 1) if size > 1 else 0.0
        n = max(1, int(round(2 * np.pi * r)))
        theta = 2 * np.pi * np.arange(n) / n  # clockwise from top
        xs = np.rint(cx + r * np.sin(theta)).astype(int)
        ys = np.rint(cy - r * np.cos(theta)).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        row = np.full((n, 3), GREY, dtype=np.uint8)
        row[inside] = arr[ys[inside], xs[inside]]

        if n == size:
            out[i] = row
        else:
            row_img = Image.fromarray(row[None, :, :], "RGB")
            out[i] = np.array(row_img.resize((size, 1), Image.LANCZOS))[0]

    msg = f"Output size {size} x {size} from {w} x {h} image, outer radius {max_r} pixels."
    if use_mosaic:
        n_block = 2 ** int(mosaic_exp)
        out = mosaic_rows(out, n_block)
        msg += f" Mosaic applied with {n_block} x 1 blocks."
    return Image.fromarray(out, "RGB"), msg


def reverse_projection(image, out_w, out_h):
    if image is None:
        return None, "Please provide an unwrapped image."

    src = np.array(image.convert("RGB"))
    size_y, size_x = src.shape[:2]
    out_w, out_h = int(out_w), int(out_h)

    max_r = max_radius(out_w, out_h)
    cy, cx = (out_h - 1) / 2.0, (out_w - 1) / 2.0

    yy, xx = np.mgrid[0:out_h, 0:out_w]
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    theta = np.mod(np.arctan2(dx, -dy), 2 * np.pi)  # clockwise from top

    row = np.clip(np.rint(r / max_r * (size_y - 1)), 0, size_y - 1).astype(int)
    col = np.floor(theta / (2 * np.pi) * size_x).astype(int) % size_x

    result = src[row, col]
    return Image.fromarray(result, "RGB"), f"Reconstructed {out_w} x {out_h} image."


with gr.Blocks(title="Pyramid Ring Unwrapper") as demo:
    gr.Markdown(
        "Row 0 is the centre pixel stretched across the full width. Each following row is a circle "
        "around the centre, unrolled clockwise from the top and Lanczos-resampled to the full output "
        "width. The last row is the circle enclosing the image corners. Output is square and at most "
        "4096 pixels wide. Samples outside the image are mid-grey."
    )
    with gr.Tab("Unwrap"):
        with gr.Row():
            inp = gr.Image(type="pil", label="Input image")
            out = gr.Image(type="pil", label="Unwrapped", format="png")
        with gr.Row():
            use_mosaic = gr.Checkbox(label="Apply horizontal mosaic", value=False)
            mosaic_exp = gr.Slider(
                1, 7, value=3, step=1,
                label="Mosaic block width exponent (N = 2^value, from 2 to 128)",
            )
        status = gr.Textbox(label="Status", interactive=False)
        gr.Button("Generate").click(make_pyramid, [inp, use_mosaic, mosaic_exp], [out, status])

    with gr.Tab("Reverse"):
        gr.Markdown(
            "Provide an unwrapped image (mosaiced or not) and the width and height of the original "
            "image to reconstruct a pixellated version of it."
        )
        with gr.Row():
            rev_in = gr.Image(type="pil", label="Unwrapped image")
            rev_out = gr.Image(type="pil", label="Reconstruction", format="png")
        with gr.Row():
            rev_w = gr.Number(value=1024, precision=0, minimum=1, label="Reconstruction width")
            rev_h = gr.Number(value=1024, precision=0, minimum=1, label="Reconstruction height")
        rev_status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            gr.Button("Use unwrapped result from first tab").click(lambda x: x, out, rev_in)
            gr.Button("Reconstruct").click(reverse_projection, [rev_in, rev_w, rev_h], [rev_out, rev_status])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
