# minimal random Voronoi edges

import random
import gradio as gr
import numpy as np
from PIL import Image

GRID_SIZE = 1024
NUM_POINTS = 200 # note 65k max for uint16

def voronoi():
    # draws boundary pixels
    rng = random.Random()
    points = np.array(
        [
            (rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
            for _ in range(NUM_POINTS)
        ],
        dtype=np.int32,
    )
    grid_y, grid_x = np.indices((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    nearest_seed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint16)
    dist_min = np.full((GRID_SIZE, GRID_SIZE), np.iinfo(np.int64).max)

    for index, (point_x, point_y) in enumerate(points):
        dist_squared = (grid_x - point_x) ** 2 + (grid_y - point_y) ** 2
        closer = dist_squared < dist_min
        nearest_seed[closer] = index
        dist_min[closer] = dist_squared[closer]

    edges = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    edges[:, 1:] |= nearest_seed[:, 1:] != nearest_seed[:, :-1] # mark vertical edges
    edges[1:, :] |= nearest_seed[1:, :] != nearest_seed[:-1, :] # mark horizontal edges

    image_array = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    image_array[edges] = 255 # set edge pixels to white
    return Image.fromarray(image_array, mode="L") # return the Voronoi edge image

def build_interface():
    with gr.Blocks() as demo:
        output_image = gr.Image(type="pil", format="png", show_label=False)
        regenerate_button = gr.Button("Regen", variant="primary")

        regenerate_button.click(fn=voronoi, outputs=output_image)
        demo.load(fn=voronoi, outputs=output_image)
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(inbrowser=True)
