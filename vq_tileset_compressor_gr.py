#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Vector Quantization Tileset Compressor

Based on a notebook by Pekka Vaananen | 30fps.net | April 30th, 2023


compress a single image to a tilemap and its tileset (its "codebook"). We will:

1. split an image into BxB pixel blocks (or "tiles"),
2. reorganize those blocks into a big list of 1D vectors,
3. cluster those using the K-means algorithm, and
4. then assign each image block to a cluster.

This is lossy image compression because we can then transmit the image as a set of
clusters and block-to-cluster assignments in a smaller size. Note that we are not
quantizing the colors to a palette, even though VQ is often used for that purpose.

Making it better

- Use some other colorspace instead of RGB. YCbCr worked OK in Pekka's tests.
- Use a smarter clustering method:
    - For large images, Mini-Batch K-Means with random initialization runs much faster.
    - The Generalized Lloyd's Algorithm (GLA) is very close to K-Means but the centroids
      are computed from voronoi region bounds, not as a mean of vectors in that cluster.
- Pack the codebook tighter, for example with a 16-bit pixel format.
- Add delta coding and some general compression like Zstandard.
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


def zoom(image_array, factor=2):
    """Nearest-neighbour upscale of an image array by an integer factor, for display."""
    return np.kron(image_array, np.ones((factor, factor, 1)))


def image_to_blocks(im, block_size):
    """
    Split the image into blocks and convert it to a big data array X.

    We wish to convert the image into an [N x D] array X of N rows with D-dimensional
    row vectors. This is the common input format for machine learning algorithms such
    as scikit-learn's KMeans class. Each block becomes a row in the array.

    The input image is an RGB bitmap of shape [h x w x 3] and is converted to an array
    of vectors with some numpy trickery:

    1. Reshape so an extra dimension of size B is added after the y and x axes.
    2. Reorder the dimensions to [bh x bw x B x B x 3].
    3. Reshape into an [N x D] array of row vectors.

    This avoids making a copy of the data and doesn't need any for loops.
    """
    h, w, _ = im.shape
    B = block_size

    # Image size measured in blocks instead of pixels
    bh, bw = h // B, w // B

    # Round image size down to a multiple of block size and crop
    h = bh * B
    w = bw * B
    im = im[:h, :w]

    # im shape is                              [h x w x 3]
    blocks = im.reshape(bh, B, bw, B, 3)     # [bh x B x bw x B x 3]
    blocks = blocks.transpose(0, 2, 1, 3, 4)  # [bh x bw x B x B x 3]

    # Length of each data vector, e.g. 8*8*3 = 192 components
    D = B * B * 3

    X = blocks.reshape(-1, D)  # [N x D]

    return X, bh, bw, im


def vectors_to_pixels(x, h, w, block_size):
    """
    The inverse operation of image_to_blocks.
    Converts an [N x D] array of block vectors back to an [h x w x 3] image.
    """
    B = block_size
    #                                 [N, D]            input shape
    x = x.reshape(h, w, -1)         # [bh, bw, D]       a 2D array of vectors
    x = x.reshape(h, w, B, B, 3)    # [bh, bw, B, B, 3] a 2D array of RGB blocks
    x = x.transpose(0, 2, 1, 3, 4)  # [bh, B, bw, B, 3] reorder block axes
    x = x.reshape(h * B, w * B, 3)  # [h, w, 3]         collapse back to pixels
    return x


def build_codebook_preview(codebook, num_codes, block_size):
    """Arrange the codebook (the "tileset") into a grid image, ordered by brightness."""
    # Pick a grid width that divides num_codes reasonably well
    cols = min(20, num_codes)
    while num_codes % cols != 0 and cols > 1:
        cols -= 1
    rows = num_codes // cols

    # Order by median Y (luma) value, since that's the channel humans read as brightness
    brightness_order = np.argsort(np.median(codebook[:, 0::3], axis=1))
    codebook_im = vectors_to_pixels(codebook[brightness_order], rows, cols, block_size)
    return zoom(codebook_im, 3)


def ycbcr_to_rgb_array(ycbcr_array):
    """Convert a float [h x w x 3] YCbCr array back to a uint8 RGB array via PIL."""
    ycbcr_u8 = np.clip(ycbcr_array, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(ycbcr_u8, mode="YCbCr").convert("RGB"))


def find_representative_blocks(distances_to_centers):
    """
    For each cluster, find the single original block closest to that cluster's center
    (its "medoid"). These actual blocks - not the averaged cluster centers - become
    the codebook, so the codebook is always made of real, unaltered image tiles and
    no color/brightness blending (quantization) is introduced.
    """
    return np.argmin(distances_to_centers, axis=0)


def draw_representative_tiles(original_rgb, representative_blocks, bw, block_size):
    """Outline, on the original image, which blocks were chosen to form the tileset."""
    debug_im = Image.fromarray(original_rgb).convert("RGB")
    draw = ImageDraw.Draw(debug_im)
    outline_width = max(1, block_size // 8)

    for block_index in representative_blocks:
        row, col = divmod(int(block_index), bw)
        x0, y0 = col * block_size, row * block_size
        x1, y1 = x0 + block_size - 1, y0 + block_size - 1
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=outline_width)

    return debug_im


def compress_image(input_image, block_size, num_codes, luma_weight, show_debug_tiles, progress=gr.Progress()):
    """
    Run the full vector quantization pipeline on an uploaded image and return:
    - the decoded (lossy, reconstructed) image
    - a preview of the codebook ("tileset")
    - a text summary including the compression ratio
    """
    if input_image is None:
        return None, None, "Please load an image first.", gr.update(value=None, visible=False)

    block_size = int(block_size)
    num_codes = int(num_codes)
    luma_weight = float(luma_weight)

    # Convert to YCbCr so brightness (Y) and color (Cb/Cr) can be weighted separately.
    # Human vision is far more sensitive to luma detail than chroma detail, which is
    # also why JPEG and video codecs subsample the chroma channels.
    im = np.asarray(input_image.convert("YCbCr")).astype(np.float32)
    h, w, _ = im.shape

    if h < block_size or w < block_size:
        message = (
            f"Image is smaller than the block size ({block_size}x{block_size}). "
            "Choose a smaller block size or a larger image."
        )
        return None, None, message, gr.update(value=None, visible=False)

    progress(0.1, desc="Splitting image into blocks")
    X, bh, bw, cropped_im = image_to_blocks(im, block_size)

    max_codes = X.shape[0]
    if num_codes > max_codes:
        num_codes = max_codes

    # Within each flattened block vector, Y/Cb/Cr channels repeat every 3 entries.
    # Scaling the Y column up makes K-means distance care more about luma error
    # than chroma error, which better matches how humans perceive block artifacts.
    X_weighted = X.copy()
    X_weighted[:, 0::3] *= luma_weight

    progress(0.3, desc="Clustering blocks (K-means)")
    # K-means is only used to group similar blocks together. The resulting cluster
    # centers are averaged, blended colors, so they are NOT used directly as the
    # codebook - that would quantize/blend the colors. Instead, for each cluster we
    # pick the one actual original block closest to its center (see below), so the
    # codebook only ever contains real, unaltered tiles copied from the source image.
    kmeans = KMeans(n_clusters=num_codes, random_state=0, n_init="auto").fit(X_weighted)

    # Assign each image block to its nearest cluster.
    # This is the "tilemap": one codebook index per block.
    codes = kmeans.predict(X_weighted)

    progress(0.5, desc="Selecting representative tiles")
    # Distance from every block to every cluster center, in the same weighted space
    # used for clustering, so the closest actual block is picked consistently.
    distances = kmeans.transform(X_weighted)
    representative_blocks = find_representative_blocks(distances)

    # The codebook (the "tileset") is built from real blocks taken from the original
    # image (unweighted, original pixel data) - never from averaged/blended colors.
    codebook = X[representative_blocks]

    progress(0.7, desc="Decoding compressed image")
    # 'decoded' has the same shape as X but is degraded due to lossy compression:
    # each block has been replaced by its assigned codebook tile.
    decoded = codebook[codes]

    # Convert the big list of codes back to pixels, then out of YCbCr into RGB.
    result = vectors_to_pixels(decoded, bh, bw, block_size)
    result_im = ycbcr_to_rgb_array(result)

    progress(0.9, desc="Building previews")
    codebook_preview = ycbcr_to_rgb_array(build_codebook_preview(codebook, num_codes, block_size))

    debug_image = None
    if show_debug_tiles:
        original_rgb = ycbcr_to_rgb_array(cropped_im)
        debug_image = draw_representative_tiles(original_rgb, representative_blocks, bw, block_size)

    # Compression ratio is a bit too optimistic: it's missing the size of the
    # 'codes' array (the tilemap itself also needs to be stored/transmitted).
    
    summary = (
        f"Orig image cropped to: {cropped_im.shape[1]} x {cropped_im.shape[0]} pixels\n"
        f"Image size in blocks: {bw} x {bh} ({X.shape[0]} blocks total)\n"
        f"Block size: {block_size} x {block_size} pixels\n"
        f"no of unique tiles: {num_codes}\n"
    )

    return (
        Image.fromarray(result_im),
        Image.fromarray(codebook_preview),
        summary,
        gr.update(value=debug_image, visible=show_debug_tiles),
    )


with gr.Blocks(title="Tileset img Compressor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        ### Vector Quantization Tileset Compressor

        lossy image compression:
        turn a pic into a tileset, cluster these into a smaller tileset "codebook" using K-means
        replace each block with its closest codebook entry. fewer codebook entries = smaller/blockier.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input", type="pil", height=800)

            block_size = gr.Slider(
                minimum=2, maximum=64, value=16, step=2,
                label="tile size (w/h in px)"
            )
            num_codes = gr.Slider(
                minimum=2, maximum=512, value=180, step=1,
                label="no. of unique tiles)"
            )
            luma_weight = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.5, step=0.1,
                label="luma weight (higher = prioritize brightness over color)"
            )
            show_debug_tiles = gr.Checkbox(
                label="Debug: outline representative tiles on original image", value=False
            )

            compress_btn = gr.Button("Compress Image", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Decoded", type="pil", height=800)
            summary_text = gr.Textbox(label="Summary", lines=6, interactive=False)

    with gr.Row():
        codebook_image = gr.Image(label="Tileset, ordered by brightness")
        debug_image = gr.Image(label="Representative tiles (debug)", visible=False)

    show_debug_tiles.change(
        fn=lambda visible: gr.update(visible=visible),
        inputs=[show_debug_tiles],
        outputs=[debug_image],
    )

    compress_btn.click(
        fn=compress_image,
        inputs=[input_image, block_size, num_codes, luma_weight, show_debug_tiles],
        outputs=[output_image, codebook_image, summary_text, debug_image],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
