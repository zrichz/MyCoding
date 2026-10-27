"""
Pinwheel Tiling - Gradio Version
Recursively subdivides a pinwheel triangle into 5 smaller triangles and plots the result.
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation


# Children as (right angle, sharp angle, medium angle) in a frame where the parent
# has right angle (0,0), sharp angle (10,0) and medium angle (0,5). Some children
# are mirrored relative to the parent, which is what makes the tiling exact.
CHILDREN = [
    ((1, 2), (5, 0), (2, 4)),
    ((1, 2), (5, 0), (0, 0)),
    ((2, 4), (0, 0), (0, 5)),
    ((6, 2), (10, 0), (5, 0)),
    ((6, 2), (2, 4), (5, 0)),
]


def subdivide(triangle):
    """Split a pinwheel triangle (v0 right, v1 sharp, v2 medium) into 5 similar triangles."""
    v0, v1, v2 = (np.array(v, dtype=float) for v in triangle)
    e1 = (v1 - v0) / 10.0
    e2 = (v2 - v0) / 5.0

    def pt(p):
        return v0 + p[0] * e1 + p[1] * e2

    return [tuple(pt(p) for p in child) for child in CHILDREN]


def generate_pinwheel(depth):
    triangles = [((2.0, 0.0), (0.0, 0.0), (2.0, 1.0))]
    for _ in range(depth):
        next_gen = []
        for t in triangles:
            next_gen.extend(subdivide(t))
        triangles = next_gen
    return triangles


RED_WHITE = LinearSegmentedColormap.from_list("red_white", ["#000000", "#ffffff"])
BACKGROUND = "#808080"


def render(depth, show_edges):
    depth = int(depth)
    triangles = generate_pinwheel(depth)
    n = len(triangles)

    # Vertex order is (right angle, sharp angle, medium angle). Red at the sharp
    # vertex, white on the opposite (shortest) side, interpolated linearly.
    pts = np.array(triangles, dtype=float)
    x = pts[:, :, 0].ravel()
    y = pts[:, :, 1].ravel()
    values = np.tile([1.0, 0.0, 1.0], n)
    tri = Triangulation(x, y, np.arange(3 * n).reshape(n, 3))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.tripcolor(tri, values, shading="gouraud", cmap=RED_WHITE, vmin=0, vmax=1)
    if show_edges:
        ax.add_collection(PolyCollection(pts, facecolors="none", edgecolors="black",
                                         linewidths=0.3 / max(depth, 1)))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Pinwheel Tiling Fractal (Depth {depth}, {n} triangles)",
                 fontsize=14, color="white")
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    return fig


with gr.Blocks(title="Pinwheel Tiling") as demo:
    gr.Markdown("# Pinwheel Tiling")
    with gr.Row():
        with gr.Column(scale=1):
            depth = gr.Slider(0, 7, value=5, step=1, label="Recursion depth (5 triangles per level)")
            edges = gr.Checkbox(value=True, label="Draw edges")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            plot = gr.Plot(label="Pinwheel tiling")

    inputs = [depth, edges]
    btn.click(render, inputs, plot)
    demo.load(render, inputs, plot)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
