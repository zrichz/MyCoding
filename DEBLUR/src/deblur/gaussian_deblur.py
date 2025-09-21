"""
Gaussian blur removal using Richardson-Lucy deconvolution and other methods.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import restoration, filters
from typing import Optional, Tuple
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.image_utils import rgb_to_grayscale, normalize_image


class GaussianDeblur:
    """
    Gaussian blur removal using various deconvolution techniques.
    """
    
    def __init__(self):
        self.name = "Gaussian Deblur"
    
    def create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """
        Create a Gaussian kernel for deconvolution.
        
        Args:
            size: Kernel size (should be odd)
            sigma: Standard deviation of Gaussian
            
        Returns:
            Gaussian kernel as numpy array
        """
        if size % 2 == 0:
            size += 1
        
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        return kernel / kernel.sum()
    
    def richardson_lucy_deconvolution(self, image: np.ndarray, kernel: np.ndarray, 
                                    iterations: int = 30, show_progress: bool = True) -> np.ndarray:
        """
        Apply Richardson-Lucy deconvolution for deblurring.
        
        Args:
            image: Blurred input image
            kernel: Point spread function (PSF) kernel
            iterations: Number of iterations
            show_progress: Whether to show iteration progress
            
        Returns:
            Deblurred image
        """
        # Convert to float and normalize
        image = image.astype(np.float64)
        kernel = kernel.astype(np.float64)
        
        # Initialize estimate
        estimate = np.copy(image)
        
        # Flip kernel for correlation
        kernel_flipped = np.flip(kernel)
        
        if show_progress:
            print(f"Richardson-Lucy deconvolution: {iterations} iterations")
        
        for i in range(iterations):
            if show_progress and (i + 1) % max(1, iterations // 10) == 0:
                print(f"  Iteration {i + 1}/{iterations} ({((i + 1) / iterations * 100):.1f}%)")
            
            # Convolve estimate with kernel
            convolved = ndimage.convolve(estimate, kernel, mode='reflect')
            
            # Avoid division by zero
            convolved = np.maximum(convolved, 1e-10)
            
            # Calculate ratio
            ratio = image / convolved
            
            # Correlate ratio with flipped kernel
            correlated = ndimage.convolve(ratio, kernel_flipped, mode='reflect')
            
            # Update estimate
            estimate = estimate * correlated
            
            # Ensure non-negative values
            estimate = np.maximum(estimate, 0)
        
        if show_progress:
            print("  Richardson-Lucy deconvolution completed!")
        
        return normalize_image(estimate)
    
    def wiener_deconvolution(self, image: np.ndarray, kernel: np.ndarray, 
                           noise_power: float = 0.01) -> np.ndarray:
        """
        Apply Wiener deconvolution for deblurring.
        
        Args:
            image: Blurred input image
            kernel: Point spread function (PSF) kernel
            noise_power: Estimated noise power
            
        Returns:
            Deblurred image
        """
        # Convert to frequency domain
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        
        # Wiener filter
        kernel_conj = np.conj(kernel_fft)
        kernel_mag_sq = np.abs(kernel_fft) ** 2
        
        wiener_filter = kernel_conj / (kernel_mag_sq + noise_power)
        
        # Apply filter and convert back
        result_fft = image_fft * wiener_filter
        result = np.fft.ifft2(result_fft).real
        
        return normalize_image(result)
    
    def _downsample_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample an image by a given factor using anti-aliasing.
        
        Args:
            image: Input image
            factor: Downsampling factor (2 or 4)
            
        Returns:
            Downsampled image
        """
        if factor == 1:
            return image
        
        if len(image.shape) == 3:
            # Color image
            h, w, c = image.shape
            new_h, new_w = h // factor, w // factor
            downsampled = np.zeros((new_h, new_w, c), dtype=image.dtype)
            
            for channel in range(c):
                # Apply Gaussian filter before downsampling to prevent aliasing
                sigma = factor / 2.0
                filtered = ndimage.gaussian_filter(image[:, :, channel].astype(np.float64), sigma)
                downsampled[:, :, channel] = filtered[::factor, ::factor]
        else:
            # Grayscale image
            h, w = image.shape
            new_h, new_w = h // factor, w // factor
            
            # Apply Gaussian filter before downsampling to prevent aliasing
            sigma = factor / 2.0
            filtered = ndimage.gaussian_filter(image.astype(np.float64), sigma)
            downsampled = filtered[::factor, ::factor]
        
        return downsampled.astype(image.dtype)
    
    def _upsample_image(self, image: np.ndarray, target_shape: tuple, factor: int) -> np.ndarray:
        """
        Upsample an image to target shape using interpolation.
        
        Args:
            image: Downsampled image
            target_shape: Target shape (original image shape)
            factor: Upsampling factor (2 or 4)
            
        Returns:
            Upsampled image
        """
        if factor == 1:
            return image
        
        from scipy import ndimage as ndi
        
        if len(image.shape) == 3:
            # Color image
            h, w, c = target_shape
            upsampled = np.zeros((h, w, c), dtype=image.dtype)
            
            for channel in range(c):
                # Use cubic interpolation for better quality
                zoom_factors = (h / image.shape[0], w / image.shape[1])
                upsampled[:, :, channel] = ndi.zoom(image[:, :, channel].astype(np.float64), 
                                                  zoom_factors, order=3, mode='reflect')
        else:
            # Grayscale image
            h, w = target_shape
            zoom_factors = (h / image.shape[0], w / image.shape[1])
            upsampled = ndi.zoom(image.astype(np.float64), zoom_factors, order=3, mode='reflect')
        
        return upsampled.astype(image.dtype)
    
    def _should_downsample(self, image: np.ndarray, threshold: int = 768) -> tuple:
        """
        Determine if image should be downsampled and by what factor.
        
        Args:
            image: Input image
            threshold: Size threshold in pixels
            
        Returns:
            Tuple of (should_downsample, recommended_factor)
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= threshold:
            return False, 1
        elif max_dim <= threshold * 2:
            return True, 2
        else:
            return True, 4
    def deblur_image(self, image: np.ndarray, kernel_size: int = 15, 
                    iterations: int = 30, method: str = 'richardson_lucy',
                    sigma: Optional[float] = None, auto_downsample: bool = True,
                    downsample_factor: Optional[int] = None, show_progress: bool = True) -> np.ndarray:
        """
        Main deblurring function.
        
        Args:
            image: Input blurred image
            kernel_size: Size of the blur kernel
            iterations: Number of iterations for iterative methods
            method: Deblurring method ('richardson_lucy' or 'wiener')
            sigma: Gaussian sigma, estimated if None
            auto_downsample: Whether to automatically downsample large images
            downsample_factor: Manual downsample factor (1, 2, or 4). Overrides auto_downsample
            show_progress: Whether to show progress information
            
        Returns:
            Deblurred image
        """
        start_time = time.time()
        original_shape = image.shape
        
        # Determine downsampling
        if downsample_factor is not None:
            if downsample_factor not in [1, 2, 4]:
                raise ValueError("Downsample factor must be 1, 2, or 4")
            use_downsample = downsample_factor > 1
            factor = downsample_factor
        elif auto_downsample:
            use_downsample, factor = self._should_downsample(image)
        else:
            use_downsample = False
            factor = 1
        
        # Show processing information
        if show_progress:
            h, w = image.shape[:2]
            print(f"Processing image: {w}x{h} pixels")
            if use_downsample:
                new_h, new_w = h // factor, w // factor
                print(f"Using {factor}x downsampling: {new_w}x{new_h} pixels for processing")
        
        # Downsample if needed
        if use_downsample:
            working_image = self._downsample_image(image, factor)
            # Adjust kernel size proportionally
            working_kernel_size = max(3, kernel_size // factor)
            if working_kernel_size % 2 == 0:
                working_kernel_size += 1
        else:
            working_image = image.copy()
            working_kernel_size = kernel_size
        
        # Convert to grayscale if needed
        if len(working_image.shape) == 3:
            is_color = True
            gray_image = rgb_to_grayscale(working_image)
        else:
            is_color = False
            gray_image = working_image.copy()
        
        # Estimate sigma if not provided
        if sigma is None:
            working_sigma = working_kernel_size / 6.0  # Rule of thumb
        else:
            working_sigma = sigma / factor if use_downsample else sigma
        
        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(working_kernel_size, working_sigma)
        
        if show_progress:
            print(f"Using kernel size: {working_kernel_size}, sigma: {working_sigma:.2f}")
        
        # Apply deblurring
        if method == 'richardson_lucy':
            result = self.richardson_lucy_deconvolution(gray_image, kernel, iterations, show_progress)
        elif method == 'wiener':
            if show_progress:
                print("Applying Wiener deconvolution...")
            result = self.wiener_deconvolution(gray_image, kernel)
            if show_progress:
                print("Wiener deconvolution completed!")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert back to color if needed
        if is_color:
            # Apply the same processing to each channel
            result_color = np.zeros_like(working_image)
            if show_progress:
                print("Processing color channels...")
            for i in range(3):
                if method == 'richardson_lucy':
                    result_color[:, :, i] = self.richardson_lucy_deconvolution(
                        working_image[:, :, i], kernel, iterations, show_progress=False)
                else:
                    result_color[:, :, i] = self.wiener_deconvolution(
                        working_image[:, :, i], kernel)
            result = result_color
        
        # Upsample if we downsampled
        if use_downsample:
            if show_progress:
                print(f"Upsampling result back to original size...")
            result = self._upsample_image(result, original_shape, factor)
        
        processing_time = time.time() - start_time
        if show_progress:
            print(f"Total processing time: {processing_time:.2f} seconds")
        
        return result
    
    def estimate_blur_kernel(self, blurred_image: np.ndarray, 
                           sharp_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate the blur kernel from a blurred image.
        
        Args:
            blurred_image: Blurred input image
            sharp_image: Optional sharp reference image
            
        Returns:
            Estimated blur kernel
        """
        if sharp_image is not None:
            # If we have a sharp reference, use division in frequency domain
            blurred_fft = np.fft.fft2(blurred_image)
            sharp_fft = np.fft.fft2(sharp_image)
            
            # Avoid division by zero
            sharp_fft = np.where(np.abs(sharp_fft) < 1e-10, 1e-10, sharp_fft)
            
            kernel_fft = blurred_fft / sharp_fft
            kernel = np.fft.ifft2(kernel_fft).real
            
            # Extract central portion
            h, w = kernel.shape
            center_h, center_w = h // 2, w // 2
            size = min(31, min(h, w) // 4)  # Reasonable kernel size
            
            kernel = kernel[center_h - size//2:center_h + size//2 + 1,
                          center_w - size//2:center_w + size//2 + 1]
        else:
            # Blind deconvolution - simple edge-based estimation
            edges = filters.sobel(blurred_image)
            kernel_size = 15
            kernel = self.create_gaussian_kernel(kernel_size, kernel_size / 6.0)
        
        return normalize_image(kernel)
