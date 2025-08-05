"""
Utility functions for image loading, saving, and basic operations.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format
    """
    try:
        # Try with PIL first for better format support
        image = Image.open(image_path)
        image = image.convert('RGB')
        return np.array(image)
    except Exception:
        # Fallback to OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save the image
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    pil_image.save(output_path)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Args:
        image: RGB image array
        
    Returns:
        Grayscale image array
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to 0-255 range.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
    """
    image = image.astype(np.float64)
    image = (image - image.min()) / (image.max() - image.min())
    return (image * 255).astype(np.uint8)


def show_comparison(original: np.ndarray, processed: np.ndarray, 
                   titles: Tuple[str, str] = ("Original", "Processed")) -> None:
    """
    Display before/after comparison of images.
    
    Args:
        original: Original image
        processed: Processed image
        titles: Tuple of titles for the images
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
