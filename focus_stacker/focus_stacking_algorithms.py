"""
Advanced Focus Stacking Algorithms
==================================
This module contains multiple state-of-the-art focus stacking algorithms.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, feature, measure
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class FocusStackingAlgorithms:
    """Collection of advanced focus stacking algorithms."""
    
    @staticmethod
    def average_stack(images: List[np.ndarray], 
                     weights: Optional[List[float]] = None,
                     progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Simple average stacking - averages all aligned images.
        
        Args:
            images: List of aligned images
            weights: Optional weights for each image (default: equal weights)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Averaged image
        """
        if not images:
            raise ValueError("No images provided")
        
        if progress_callback:
            progress_callback("Starting average stacking...")
        
        # Convert to float for precision
        images_float = []
        for i, img in enumerate(images):
            if progress_callback:
                progress_callback(f"Converting image {i+1}/{len(images)} to float...")
            images_float.append(img.astype(np.float64))
        
        if progress_callback:
            progress_callback("Computing weighted average...")
        
        # Initialize result
        result = np.zeros_like(images_float[0], dtype=np.float64)
        
        # Set default equal weights
        if weights is None:
            weights = np.ones(len(images)) / len(images)
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted average
        for i, (img, weight) in enumerate(zip(images_float, weights)):
            if progress_callback:
                progress_callback(f"Adding image {i+1}/{len(images)} (weight: {weight:.3f})...")
            result += img * weight
        
        if progress_callback:
            progress_callback("Converting result to uint8...")
        
        # Convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if progress_callback:
            progress_callback("Average stacking complete!")
        
        return result

    @staticmethod
    def laplacian_pyramid_stack(images: List[np.ndarray], 
                               levels: int = 5,
                               sigma: float = 1.0,
                               progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Advanced Laplacian pyramid focus stacking with multi-scale analysis.
        
        Args:
            images: List of input images (same size)
            levels: Number of pyramid levels
            sigma: Gaussian blur sigma for pyramid construction
            progress_callback: Optional callback for progress updates
            
        Returns:
            Focus stacked image
        """
        if not images:
            raise ValueError("No images provided")
        
        if progress_callback:
            progress_callback("Starting Laplacian pyramid stacking...")
        
        # Convert to float32 for better precision
        float_images = [img.astype(np.float32) / 255.0 for img in images]
        
        # Build Laplacian pyramids for each image
        pyramids = []
        for img in float_images:
            pyramid = FocusStackingAlgorithms._build_laplacian_pyramid(img, levels, sigma)
            pyramids.append(pyramid)
        
        if progress_callback:
            progress_callback("Combining pyramid levels...")
        
        # Combine pyramids using focus measure
        combined_pyramid = []
        for level in range(levels):
            level_images = [pyramid[level] for pyramid in pyramids]
            combined_level = FocusStackingAlgorithms._combine_pyramid_level(level_images)
            combined_pyramid.append(combined_level)
        
        if progress_callback:
            progress_callback("Reconstructing image from pyramid...")
        
        # Reconstruct image from combined pyramid
        result = FocusStackingAlgorithms._reconstruct_from_pyramid(combined_pyramid)
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        if progress_callback:
            progress_callback("Laplacian pyramid stacking complete!")
        
        return result
    
    @staticmethod
    def _build_laplacian_pyramid(image: np.ndarray, levels: int, sigma: float) -> List[np.ndarray]:
        """Build Laplacian pyramid for an image."""
        # Ensure image is float32
        image = image.astype(np.float32)
        gaussian_pyramid = [image.copy()]
        
        # Build Gaussian pyramid
        current = image.copy()
        for i in range(levels - 1):
            current = cv2.GaussianBlur(current, (0, 0), sigma)
            current = cv2.resize(current, (current.shape[1]//2, current.shape[0]//2), 
                               interpolation=cv2.INTER_LINEAR)
            gaussian_pyramid.append(current)
        
        # Build Laplacian pyramid
        laplacian_pyramid = []
        for i in range(levels - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.resize(gaussian_pyramid[i + 1], size, interpolation=cv2.INTER_LINEAR)
            expanded = cv2.GaussianBlur(expanded, (0, 0), sigma)
            # Ensure same data type for subtraction
            gaussian_current = gaussian_pyramid[i].astype(np.float32)
            expanded = expanded.astype(np.float32)
            laplacian = gaussian_current - expanded
            laplacian_pyramid.append(laplacian)
        
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
    
    @staticmethod
    def _combine_pyramid_level(level_images: List[np.ndarray]) -> np.ndarray:
        """Combine pyramid level using focus measure."""
        if len(level_images) == 1:
            return level_images[0]
        
        # Calculate focus measure for each image
        focus_measures = []
        for img in level_images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = (img * 255).astype(np.uint8)
            
            # Use variance of Laplacian as focus measure
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_measure = np.abs(laplacian).astype(np.float32)
            focus_measures.append(focus_measure)
        
        # Stack focus measures and find best focused pixels
        focus_stack = np.stack(focus_measures, axis=-1)
        best_focus_indices = np.argmax(focus_stack, axis=-1)
        
        # Combine images based on best focus
        result = np.zeros_like(level_images[0])
        for i, img in enumerate(level_images):
            mask = (best_focus_indices == i)
            if len(img.shape) == 3:
                mask = np.stack([mask] * img.shape[2], axis=-1)
            result = np.where(mask, img, result)
        
        return result
    
    @staticmethod
    def _reconstruct_from_pyramid(pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        result = pyramid[-1].astype(np.float32)
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            result = cv2.resize(result, size, interpolation=cv2.INTER_LINEAR)
            pyramid_level = pyramid[i].astype(np.float32)
            result = result + pyramid_level
        return result
    
    @staticmethod
    def gradient_based_stack(images: List[np.ndarray], 
                           kernel_size: int = 5,
                           threshold: float = 0.1,
                           smooth_radius: int = 0,
                           blend_sigma: float = 1.0,
                           progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Gradient-based focus stacking using Sobel operators.
        
        Args:
            images: List of input images
            kernel_size: Size of Sobel kernel
            threshold: Threshold for gradient magnitude
            smooth_radius: Radius for smoothing focus regions (0 = no smoothing, 5+ = smooth)
            blend_sigma: Gaussian sigma for blending between images
            progress_callback: Optional callback for progress updates
            
        Returns:
            Focus stacked image
        """
        if not images:
            raise ValueError("No images provided")
        
        if progress_callback:
            progress_callback("Starting gradient-based stacking...")
        
        # Calculate gradient magnitude for each image
        gradient_maps = []
        for i, img in enumerate(images):
            if progress_callback:
                progress_callback(f"Computing gradients for image {i+1}/{len(images)}...")
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Apply threshold
            gradient_mag = np.where(gradient_mag > threshold, gradient_mag, 0)
            gradient_maps.append(gradient_mag)
        
        if progress_callback:
            progress_callback("Finding best focused pixels...")
        
        # Find best focused pixels
        gradient_stack = np.stack(gradient_maps, axis=-1)
        best_focus_indices = np.argmax(gradient_stack, axis=-1)
        
        if smooth_radius > 0:
            if progress_callback:
                progress_callback(f"Smoothing focus regions (radius: {smooth_radius})...")
            
            # Create smooth weight maps instead of hard binary masks
            return FocusStackingAlgorithms._gradient_smooth_blend(
                images, gradient_maps, smooth_radius, blend_sigma, progress_callback)
        else:
            if progress_callback:
                progress_callback("Combining images with hard selection...")
            
            # Original hard selection method
            result = np.zeros_like(images[0])
            for i, img in enumerate(images):
                mask = (best_focus_indices == i)
                if len(img.shape) == 3:
                    mask = np.stack([mask] * img.shape[2], axis=-1)
                result = np.where(mask, img, result)
        
        if progress_callback:
            progress_callback("Gradient-based stacking complete!")
        
        return result
    
    @staticmethod
    def _gradient_smooth_blend(images: List[np.ndarray], 
                              gradient_maps: List[np.ndarray],
                              smooth_radius: int,
                              blend_sigma: float,
                              progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Smooth blending for gradient-based stacking to reduce noise.
        
        Args:
            images: List of input images
            gradient_maps: Pre-computed gradient magnitude maps
            smooth_radius: Radius for morphological smoothing
            blend_sigma: Gaussian sigma for weight smoothing
            progress_callback: Optional progress callback
            
        Returns:
            Smoothly blended result
        """
        if progress_callback:
            progress_callback("Creating smooth weight maps...")
        
        # Create normalized weight maps from gradient magnitudes
        gradient_stack = np.stack(gradient_maps, axis=-1)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-6
        weight_sum = np.sum(gradient_stack, axis=-1, keepdims=True) + epsilon
        weights = gradient_stack / weight_sum
        
        # Smooth the weight maps to reduce hard transitions
        if progress_callback:
            progress_callback("Smoothing weight maps...")
        
        smooth_weights = []
        for i in range(len(images)):
            weight = weights[:, :, i]
            
            # Apply morphological operations to smooth regions
            if smooth_radius > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (smooth_radius*2+1, smooth_radius*2+1))
                weight = cv2.morphologyEx(weight, cv2.MORPH_CLOSE, kernel)
                weight = cv2.morphologyEx(weight, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian smoothing for soft transitions
            if blend_sigma > 0:
                weight = cv2.GaussianBlur(weight, (0, 0), blend_sigma)
            
            smooth_weights.append(weight)
        
        # Re-normalize weights
        if progress_callback:
            progress_callback("Re-normalizing smooth weights...")
        
        smooth_weight_stack = np.stack(smooth_weights, axis=-1)
        weight_sum = np.sum(smooth_weight_stack, axis=-1, keepdims=True) + epsilon
        normalized_weights = smooth_weight_stack / weight_sum
        
        # Blend images using smooth weights
        if progress_callback:
            progress_callback("Blending images with smooth weights...")
        
        result = np.zeros_like(images[0], dtype=np.float64)
        
        for i, img in enumerate(images):
            weight = normalized_weights[:, :, i]
            if len(img.shape) == 3:
                # Expand weight to match color channels
                weight = np.stack([weight] * img.shape[2], axis=-1)
            
            result += img.astype(np.float64) * weight
        
        # Convert back to original data type
        result = np.clip(result, 0, 255).astype(images[0].dtype)
        
        return result
    
    @staticmethod
    def variance_based_stack(images: List[np.ndarray], 
                           window_size: int = 15,
                           smooth_sigma: float = 1.0,
                           progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Variance-based focus stacking with smoothing.
        
        Args:
            images: List of input images
            window_size: Size of local variance window
            smooth_sigma: Gaussian smoothing sigma for focus map
            
        Returns:
            Focus stacked image
        """
        if not images:
            raise ValueError("No images provided")
        
        if progress_callback:
            progress_callback("Starting variance-based stacking...")
        
        # Calculate local variance for each image
        variance_maps = []
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = img.astype(np.float32)
            
            # Calculate local variance
            mean = cv2.filter2D(gray, -1, kernel)
            sqr_mean = cv2.filter2D(gray * gray, -1, kernel)
            variance = sqr_mean - mean * mean
            
            # Smooth variance map
            variance = cv2.GaussianBlur(variance, (0, 0), smooth_sigma)
            variance_maps.append(variance)
        
        if progress_callback:
            progress_callback("Finding best focused pixels...")
        
        # Find best focused pixels
        variance_stack = np.stack(variance_maps, axis=-1)
        best_focus_indices = np.argmax(variance_stack, axis=-1)
        
        if progress_callback:
            progress_callback("Combining images with feathering...")
        
        # Combine images with feathering for smoother transitions
        result = FocusStackingAlgorithms._feathered_combine(images, best_focus_indices)
        
        if progress_callback:
            progress_callback("Variance-based stacking complete!")
        
        return result
    
    @staticmethod
    def _feathered_combine(images: List[np.ndarray], 
                          indices: np.ndarray,
                          feather_radius: int = 5) -> np.ndarray:
        """Combine images with feathered transitions."""
        result = np.zeros_like(images[0], dtype=np.float32)
        total_weight = np.zeros(indices.shape, dtype=np.float32)
        
        for i, img in enumerate(images):
            # Create mask for this image
            mask = (indices == i).astype(np.float32)
            
            # Apply feathering (distance transform + gaussian)
            if feather_radius > 0:
                mask = cv2.GaussianBlur(mask, (feather_radius*2+1, feather_radius*2+1), feather_radius/3)
            
            # Accumulate weighted result
            if len(img.shape) == 3:
                mask_3d = np.stack([mask] * img.shape[2], axis=-1)
                result += img.astype(np.float32) * mask_3d
                total_weight += mask
            else:
                result += img.astype(np.float32) * mask
                total_weight += mask
        
        # Normalize by total weight
        total_weight = np.maximum(total_weight, 1e-10)
        if len(result.shape) == 3:
            total_weight = np.stack([total_weight] * result.shape[2], axis=-1)
        
        result = result / total_weight
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def wavelet_based_stack(images: List[np.ndarray], 
                           wavelet: str = 'db4',
                           levels: int = 4,
                           progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Wavelet-based focus stacking using discrete wavelet transform.
        
        Args:
            images: List of input images
            wavelet: Wavelet type
            levels: Number of decomposition levels
            
        Returns:
            Focus stacked image
        """
        try:
            import pywt
        except ImportError:
            logger.warning("PyWavelets not available, falling back to Laplacian pyramid")
            return FocusStackingAlgorithms.laplacian_pyramid_stack(images)
        
        if not images:
            raise ValueError("No images provided")
        
        if progress_callback:
            progress_callback("Starting wavelet-based stacking...")
        
        # Convert to float
        float_images = [img.astype(np.float32) / 255.0 for img in images]
        
        if progress_callback:
            progress_callback("Processing each channel separately...")
        
        # Process each channel separately for color images
        if len(float_images[0].shape) == 3:
            channels = []
            for c in range(float_images[0].shape[2]):
                channel_images = [img[:, :, c] for img in float_images]
                stacked_channel = FocusStackingAlgorithms._wavelet_stack_channel(
                    channel_images, wavelet, levels)
                channels.append(stacked_channel)
            result = np.stack(channels, axis=-1)
        else:
            result = FocusStackingAlgorithms._wavelet_stack_channel(
                float_images, wavelet, levels)
        
        if progress_callback:
            progress_callback("Converting result to uint8...")
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _wavelet_stack_channel(images: List[np.ndarray], 
                              wavelet: str, 
                              levels: int) -> np.ndarray:
        """Stack single channel using wavelets."""
        import pywt
        
        # Decompose each image
        coeffs_list = []
        for img in images:
            coeffs = pywt.wavedec2(img, wavelet, level=levels)
            coeffs_list.append(coeffs)
        
        # Combine coefficients based on energy
        combined_coeffs = []
        for level in range(len(coeffs_list[0])):
            if level == 0:  # Approximation coefficients
                level_coeffs = [coeffs[level] for coeffs in coeffs_list]
                combined = np.mean(level_coeffs, axis=0)
                combined_coeffs.append(combined)
            else:  # Detail coefficients (tuple of 3)
                detail_combined = []
                for detail_idx in range(3):  # LH, HL, HH
                    detail_coeffs = [coeffs[level][detail_idx] for coeffs in coeffs_list]
                    # Select coefficients with highest absolute value
                    detail_stack = np.stack(detail_coeffs, axis=-1)
                    abs_stack = np.abs(detail_stack)
                    best_indices = np.argmax(abs_stack, axis=-1)
                    
                    combined_detail = np.zeros_like(detail_coeffs[0])
                    for i, detail in enumerate(detail_coeffs):
                        mask = (best_indices == i)
                        combined_detail = np.where(mask, detail, combined_detail)
                    
                    detail_combined.append(combined_detail)
                combined_coeffs.append(tuple(detail_combined))
        
        # Reconstruct image
        result = pywt.waverec2(combined_coeffs, wavelet)
        return result
