"""Generate a 1024x1024 atlas containing every 4x4 binary pattern."""

import tempfile

import gradio as gr
import numpy as np
from PIL import Image


def create_atlas(distinguish_symmetries=False):
    pattern_count = 1 << 16
    patterns_per_side = 1 << 8
    tile_size = 4

    pattern_ids = np.arange(pattern_count, dtype=np.uint16)[:, None]
    bit_positions = np.arange(15, -1, -1, dtype=np.uint16)[None, :]
    patterns = ((pattern_ids >> bit_positions) & 1).astype(np.uint8)
    patterns = patterns.reshape(pattern_count, tile_size, tile_size)

    if distinguish_symmetries:
        rotational = np.zeros(pattern_count, dtype=bool)
        for turns in (1, 2, 3):
            rotated = np.rot90(patterns, k=turns, axes=(1, 2))
            rotational |= np.all(patterns == rotated, axis=(1, 2))

        transposed = np.swapaxes(patterns, 1, 2)
        reflections = (
            np.flip(patterns, axis=1),
            np.flip(patterns, axis=2),
            transposed,
            np.flip(transposed, axis=(1, 2)),
        )
        mirrored = np.logical_or.reduce(
            [np.all(patterns == reflected, axis=(1, 2)) for reflected in reflections]
        )

        foreground_colors = np.full((pattern_count, 3), 255, dtype=np.uint8)
        foreground_colors[rotational & ~mirrored] = (255, 64, 64)
        foreground_colors[mirrored & ~rotational] = (64, 160, 255)
        foreground_colors[rotational & mirrored] = (255, 200, 64)
        colored_patterns = patterns[..., None] * foreground_colors[:, None, None, :]
        atlas = colored_patterns.reshape(
            patterns_per_side, patterns_per_side, tile_size, tile_size, 3
        ).transpose(0, 2, 1, 3, 4).reshape(
            patterns_per_side * tile_size,
            patterns_per_side * tile_size,
            3,
        )
        return Image.fromarray(atlas, mode="RGB")

    patterns = patterns.reshape(patterns_per_side, patterns_per_side, tile_size, tile_size)

    atlas = patterns.transpose(0, 2, 1, 3).reshape(
        patterns_per_side * tile_size,
        patterns_per_side * tile_size,
    )
    return Image.fromarray(atlas * 255, mode="L")


def create_atlas_file(distinguish_symmetries):
    atlas = create_atlas(distinguish_symmetries)
    with tempfile.NamedTemporaryFile(prefix="binary_4x4_atlas_", suffix=".png", delete=False) as png_file:
        atlas.save(png_file, format="PNG")
        png_path = png_file.name
    return atlas, png_path


demo = gr.Interface(
    fn=create_atlas_file,
    inputs=gr.Checkbox(label="Distinguish rotational and mirror symmetries", value=False),
    outputs=[
        gr.Image(label="All 4x4 binary patterns", type="pil", format="png"),
        gr.File(label="Download PNG"),
    ],
    title="4x4 Binary Pattern Atlas",
)


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())