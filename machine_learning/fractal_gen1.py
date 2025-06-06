import numpy as np
import random

import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_mandelbrot_image(xmin, xmax, ymin, ymax, width, height, max_iter):
    image = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            real = xmin + (x / width) * (xmax - xmin)
            imag = ymin + (y / height) * (ymax - ymin)
            c = complex(real, imag)
            image[y, x] = mandelbrot(c, max_iter)
    return image

def save_mandelbrot_image(filename, xmin, xmax, ymin, ymax, width, height, max_iter, cmap):
    image = generate_mandelbrot_image(xmin, xmax, ymin, ymax, width, height, max_iter)
    plt.imshow(image, extent=(xmin, xmax, ymin, ymax), cmap=cmap)
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# List of available matplotlib colormaps (excluding 'hot' if you want more variety)
colormaps = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
    'cool', 'Wistia', 'spring', 'summer', 'autumn', 'hot','winter', 'bone', 'copper', 'pink',
    'jet', 'turbo', 'cubehelix', 'gnuplot', 'gnuplot2', 'ocean', 'rainbow', 'terrain'
]

# Generate and save 3 Mandelbrot fractal images with random colormaps and at least 50% non-black pixels
for i in range(2):
    while True:
        side = random.uniform(0.1, 0.101)  # side length of the square region
        xmin = random.uniform(0.35, 0.351)
        ymin = random.uniform(0.293, 0.294)
        xmax = xmin + side
        ymax = ymin + side
        image = generate_mandelbrot_image(xmin, xmax, ymin, ymax, 512, 512, 500)
        # Count non-black pixels (assuming black is value 0)
        non_black = np.count_nonzero(image)
        if non_black >= 0.5 * image.size:
            break  # Accept this image

    cmap = random.choice(colormaps)
    print(f"Generating Mandelbrot image {i+1} with bounds: xmin={xmin:.2f}, xmax={xmax:.2f}, ymin={ymin:.2f}, ymax={ymax:.2f}, cmap={cmap}")
    save_mandelbrot_image(f'mandelbrot_{i+1}.png', xmin, xmax, ymin, ymax, 512, 512, 100, cmap)