import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_kernel(kernel_size, sigma):
    center = kernel_size // 2
    x, y = np.meshgrid(np.arange(kernel_size) - center, np.arange(kernel_size) - center)
    gauss_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gauss_kernel / gauss_kernel.sum()

kernel_size = 51
sigmas = [1.0, 2.0, 4.0, 6.0]

plt.figure(figsize=(12, 3))
for i, sigma in enumerate(sigmas, 1):
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    plt.subplot(1, len(sigmas), i)
    plt.imshow(kernel, cmap='hot')
    plt.title(f"σ = {sigma}")
    plt.axis('off')
plt.suptitle("Effect of Sigma on Gaussian Kernel")
plt.tight_layout()
plt.show()
