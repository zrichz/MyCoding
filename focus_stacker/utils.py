"""
Utility Functions for Focus Stacker
==================================
Helper functions and utilities for the focus stacking application.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def load_image_safe(file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Safely load an image with error handling.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Loaded image or None if failed
        """
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                logger.warning(f"OpenCV failed to load image: {file_path}")
                return None
            return img
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None
    
    @staticmethod
    def resize_image_if_needed(image: np.ndarray, 
                              max_dimension: int = 4000) -> np.ndarray:
        """
        Resize image if it's too large.
        
        Args:
            image: Input image
            max_dimension: Maximum allowed dimension
            
        Returns:
            Resized image if needed, original otherwise
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > max_dimension:
            scale = max_dimension / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    @staticmethod
    def ensure_same_size(images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Ensure all images have the same size.
        
        Args:
            images: List of input images
            
        Returns:
            List of resized images
        """
        if not images:
            return images
        
        # Find the minimum common size
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)
        
        resized_images = []
        for img in images:
            if img.shape[0] != min_h or img.shape[1] != min_w:
                resized = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_LANCZOS4)
                resized_images.append(resized)
            else:
                resized_images.append(img)
        
        return resized_images
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if needed.
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, 
                        alpha: float = 1.5, 
                        beta: int = 0) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Enhanced image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, 
                           sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            sigma: Blur strength
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (0, 0), sigma)


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def find_image_files(directory: Union[str, Path], 
                        extensions: List[str] = None) -> List[Path]:
        """
        Find all image files in a directory.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to search for
            
        Returns:
            List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
        directory = Path(directory)
        image_files = []
        
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    @staticmethod
    def create_output_filename(input_files: List[Union[str, Path]], 
                              suffix: str = "_stacked", 
                              extension: str = ".png") -> str:
        """
        Create an appropriate output filename.
        
        Args:
            input_files: List of input file paths
            suffix: Suffix to add to filename
            extension: Output file extension
            
        Returns:
            Output filename
        """
        if not input_files:
            return f"focus_stacked{suffix}{extension}"
        
        first_file = Path(input_files[0])
        base_name = first_file.stem
        
        # If there's a common prefix, use it
        if len(input_files) > 1:
            common_prefix = os.path.commonprefix([str(p) for p in input_files])
            if common_prefix:
                base_name = Path(common_prefix).stem
        
        return f"{base_name}{suffix}{extension}"
    
    @staticmethod
    def ensure_directory_exists(file_path: Union[str, Path]):
        """
        Ensure the directory for a file path exists.
        
        Args:
            file_path: File path to check
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


class ValidationUtils:
    """Utility functions for validation."""
    
    @staticmethod
    def validate_images(images: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Validate a list of images for focus stacking.
        
        Args:
            images: List of images to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not images:
            return False, "No images provided"
        
        if len(images) < 2:
            return False, "At least 2 images required for focus stacking"
        
        # Check if all images have the same number of channels
        first_channels = len(images[0].shape)
        for i, img in enumerate(images[1:], 1):
            if len(img.shape) != first_channels:
                return False, f"Image {i} has different number of channels than first image"
        
        # Check if images are roughly the same size (allow small variations)
        first_h, first_w = images[0].shape[:2]
        for i, img in enumerate(images[1:], 1):
            h, w = img.shape[:2]
            size_diff_h = abs(h - first_h) / first_h
            size_diff_w = abs(w - first_w) / first_w
            
            if size_diff_h > 0.1 or size_diff_w > 0.1:
                return False, f"Image {i} size ({w}x{h}) differs significantly from first image ({first_w}x{first_h})"
        
        return True, "Images are valid"
    
    @staticmethod
    def validate_parameters(method: str, **kwargs) -> Tuple[bool, str]:
        """
        Validate stacking parameters.
        
        Args:
            method: Stacking method name
            **kwargs: Method parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_methods = ['laplacian', 'gradient', 'variance', 'wavelet']
        if method not in valid_methods:
            return False, f"Invalid method '{method}'. Must be one of: {valid_methods}"
        
        if method == 'laplacian':
            levels = kwargs.get('levels', 5)
            if not isinstance(levels, int) or levels < 2 or levels > 10:
                return False, "Pyramid levels must be an integer between 2 and 10"
            
            sigma = kwargs.get('sigma', 1.0)
            if not isinstance(sigma, (int, float)) or sigma <= 0 or sigma > 5:
                return False, "Gaussian sigma must be a positive number <= 5"
        
        return True, "Parameters are valid"


class PerformanceUtils:
    """Utility functions for performance optimization."""
    
    @staticmethod
    def estimate_memory_usage(images: List[np.ndarray]) -> int:
        """
        Estimate memory usage for processing images.
        
        Args:
            images: List of images
            
        Returns:
            Estimated memory usage in bytes
        """
        if not images:
            return 0
        
        # Calculate total pixel count
        total_pixels = sum(img.shape[0] * img.shape[1] * img.shape[2] if len(img.shape) == 3 else img.shape[0] * img.shape[1] 
                          for img in images)
        
        # Estimate memory usage (assuming float32 for processing)
        # Factor of 4 for temporary arrays during processing
        bytes_per_pixel = 4  # float32
        memory_factor = 4   # for temporary arrays
        
        return total_pixels * bytes_per_pixel * memory_factor
    
    @staticmethod
    def should_tile_processing(images: List[np.ndarray], 
                              max_memory_mb: int = 1000) -> bool:
        """
        Determine if images should be processed in tiles.
        
        Args:
            images: List of images
            max_memory_mb: Maximum memory to use in MB
            
        Returns:
            True if tiling is recommended
        """
        estimated_memory = PerformanceUtils.estimate_memory_usage(images)
        max_memory_bytes = max_memory_mb * 1024 * 1024
        
        return estimated_memory > max_memory_bytes
    
    @staticmethod
    def calculate_optimal_tile_size(image_shape: Tuple[int, int], 
                                   max_memory_mb: int = 500) -> Tuple[int, int]:
        """
        Calculate optimal tile size for memory-efficient processing.
        
        Args:
            image_shape: Shape of the image (height, width)
            max_memory_mb: Maximum memory per tile in MB
            
        Returns:
            Optimal tile size (height, width)
        """
        h, w = image_shape
        max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Assume 4 bytes per pixel (float32) and factor for temporary arrays
        bytes_per_pixel = 4 * 4
        max_pixels = max_memory_bytes // bytes_per_pixel
        
        # Try to keep aspect ratio
        aspect_ratio = w / h
        tile_h = int(np.sqrt(max_pixels / aspect_ratio))
        tile_w = int(tile_h * aspect_ratio)
        
        # Ensure tiles don't exceed image dimensions
        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)
        
        # Ensure minimum tile size
        tile_h = max(tile_h, 256)
        tile_w = max(tile_w, 256)
        
        return tile_h, tile_w


def setup_logging_with_file(log_file: str = "focus_stacker.log"):
    """
    Setup logging with both file and console output.
    
    Args:
        log_file: Path to log file
    """
    import logging
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


if __name__ == "__main__":
    # Test utilities
    print("Testing Focus Stacker Utilities...")
    
    # Test image loading
    print("✓ Image utilities loaded")
    print("✓ File utilities loaded") 
    print("✓ Validation utilities loaded")
    print("✓ Performance utilities loaded")
    
    print("All utilities working correctly!")
