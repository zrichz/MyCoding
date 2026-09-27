#!/home/rich/MyCoding/venvMyCoding/bin/python
"""Convert an uploaded image into mean-colour Voronoi regions."""

import gradio as gr
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw


MAX_SIDE_LENGTH = 1024


def random_seed_pixels(height, width, num_points, rng):
    """Choose unique seed pixels with a uniform distribution."""
    return rng.choice(height * width, size=num_points, replace=False)


def voronoi_labels(height, width, seed_pixels):
    """Assign each pixel to its nearest seed and return region labels."""
    seed_mask = np.ones((height, width), dtype=bool)
    seed_ys, seed_xs = np.unravel_index(seed_pixels, (height, width))
    seed_mask[seed_ys, seed_xs] = False
    nearest_indices = distance_transform_edt(seed_mask, return_distances=False, return_indices=True)
    nearest_pixels = nearest_indices[0] * width + nearest_indices[1]
    seed_lookup = {pixel: region_id for region_id, pixel in enumerate(seed_pixels, start=1)}
    return np.vectorize(seed_lookup.get)(nearest_pixels)


def relax_seed_pixels(height, width, seed_pixels, iterations):
    """Move seeds to their region centroids using Lloyd's algorithm."""
    for _ in range(iterations):
        labels = voronoi_labels(height, width, seed_pixels)
        y_coordinates, x_coordinates = np.indices((height, width))
        relaxed_pixels = []
        occupied_pixels = set()
        for region_id in range(1, len(seed_pixels) + 1):
            region = labels == region_id
            candidate_y = int(np.rint(y_coordinates[region].mean()))
            candidate_x = int(np.rint(x_coordinates[region].mean()))
            candidate_pixel = candidate_y * width + candidate_x
            if candidate_pixel in occupied_pixels:
                candidate_pixel = seed_pixels[region_id - 1]
            occupied_pixels.add(candidate_pixel)
            relaxed_pixels.append(candidate_pixel)
        seed_pixels = np.asarray(relaxed_pixels, dtype=np.int64)
    return seed_pixels


def colorize_voronoi(
    image, num_points, seed, relaxation_iterations, circle_radius_scale
):
    """Draw relaxed, mean-colour circles over the original image."""
    if image is None:
        raise gr.Error("Please upload an image.")

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    height, width = image_array.shape[:2]
    max_side = max(height, width)
    if max_side > MAX_SIDE_LENGTH:
        scale = MAX_SIDE_LENGTH / max_side
        resized_size = (round(width * scale), round(height * scale))
        image_array = np.asarray(
            Image.fromarray(image_array).resize(resized_size, Image.Resampling.LANCZOS)
        )

    height, width = image_array.shape[:2]
    num_points = min(int(num_points), height * width)
    rng = np.random.default_rng(None if seed is None else int(seed))

    seed_pixels = random_seed_pixels(height, width, num_points, rng)
    seed_pixels = relax_seed_pixels(
        height, width, seed_pixels, int(relaxation_iterations)
    )
    labels = voronoi_labels(height, width, seed_pixels)

    output_image = Image.fromarray(image_array.copy())
    drawing = ImageDraw.Draw(output_image)
    seed_ys, seed_xs = np.unravel_index(seed_pixels, (height, width))
    for region_id in range(1, num_points + 1):
        region = labels == region_id
        average_color = np.clip(image_array[region].mean(axis=0), 0, 255).astype(
            np.uint8
        )
        radius = max(
            1, int(np.sqrt(region.sum() / np.pi) * float(circle_radius_scale))
        )
        center_x = int(seed_xs[region_id - 1])
        center_y = int(seed_ys[region_id - 1])
        drawing.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=tuple(average_color),
        )

    return np.asarray(output_image)


with gr.Blocks(theme=gr.themes.Soft(), title="Voronoi Image Colorizer") as demo:
    gr.Markdown("# Voronoi Image Colorizer")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Image", type="numpy")
            num_points = gr.Slider(
                minimum=10,
                maximum=3000,
                value=500,
                step=1,
                label="Voronoi Seed Points",
            )
            relaxation_iterations = gr.Slider(
                minimum=0,
                maximum=20,
                value=5,
                step=1,
                label="Lloyd Relaxation Iterations",
            )
            circle_radius_scale = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.4,
                step=0.05,
                label="Circle Radius",
            )
            seed = gr.Number(value=0, precision=0, label="Random Seed")
            convert_button = gr.Button("Create Voronoi Image", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Voronoi Result", type="numpy")

    convert_button.click(
        fn=colorize_voronoi,
        inputs=[
            input_image,
            num_points,
            seed,
            relaxation_iterations,
            circle_radius_scale,
        ],
        outputs=output_image,
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)