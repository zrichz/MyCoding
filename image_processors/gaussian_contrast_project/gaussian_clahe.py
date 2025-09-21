"""
Gaussian-weighted CLAHE-style local contrast enhancement (Lab luminance only)

- Operates in perceptual Lab color space to preserve hue and saturation
- Splits luminance channel into overlapping tiles
- Applies histogram equalization with Gaussian weighting in each tile
- Blends tiles smoothly using Hanning window to prevent seams
"""

import numpy as np
from skimage import color
from scipy.ndimage import convolve
from PIL import Image
import os
import sys

def generate_gaussian_kernel(kernel_size):
    sigma = kernel_size / 6.0
    center = kernel_size // 2
    x, y = np.meshgrid(np.arange(kernel_size) - center, np.arange(kernel_size) - center)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def equalize_tile(tile, kernel):
    smoothed = convolve(tile, kernel, mode='reflect')
    hist, bins = np.histogram(smoothed.flatten(), 256, [0, 256], density=True)
    cdf = 255 * hist.cumsum() / hist.sum()
    return np.interp(tile.flatten(), bins[:-1], cdf).reshape(tile.shape)

def process_luminance_channel_blend(L_channel, tile_size, stride, kernel, progress_callback=None):
    h, w = L_channel.shape
    result = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    blend_window = np.outer(np.hanning(tile_size), np.hanning(tile_size))

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))
    total_tiles = len(y_steps) * len(x_steps)
    tile_count = 0

    for y in y_steps:
        for x in x_steps:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = L_channel[y:y_end, x:x_end]

            kernel_resized = kernel[:tile.shape[0], :tile.shape[1]]
            blend_resized = blend_window[:tile.shape[0], :tile.shape[1]]
            eq_tile = equalize_tile(tile, kernel_resized)

            result[y:y_end, x:x_end] += eq_tile * blend_resized
            weight_sum[y:y_end, x:x_end] += blend_resized

            # Print progress
            tile_count += 1
            if progress_callback:
                percent = int((tile_count / total_tiles) * 100)
                progress_callback(percent)


    return np.divide(result, weight_sum, out=np.zeros_like(result), where=weight_sum != 0).astype(np.uint8)

def gaussian_clahe_color(image_path, tile_size=64, kernel_size=21, stride=None, progress_callback=None):
    if stride is None:
        stride = tile_size // 2

    image = Image.open(image_path).convert('RGB')
    img_np = np.asarray(image, dtype=np.float32) / 255.0
    lab = color.rgb2lab(img_np)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    L_scaled = (L / 100.0 * 255).astype(np.uint8)

    kernel = generate_gaussian_kernel(kernel_size)
    L_eq = process_luminance_channel_blend(L_scaled, tile_size, stride, kernel, progress_callback)
    L_eq = (L_eq.astype(np.float32) / 255.0) * 100

    lab_eq = np.stack([L_eq, A, B], axis=2)
    rgb_eq = np.clip(color.lab2rgb(lab_eq), 0, 1)
    return Image.fromarray((rgb_eq * 255).astype(np.uint8))
