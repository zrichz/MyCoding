"""
Motion blur removal using various deconvolution and filtering techniques.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Optional
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.image_utils import rgb_to_grayscale, normalize_image


class MotionDeblur:
    """
    Motion blur removal using various techniques.
    """
    
    def __init__(self):
        self.name = "Motion Deblur"
    
    def create_motion_kernel(self, length: int, angle: float) -> np.ndarray:
        """
        Create a motion blur kernel.
        
        Args:
            length: Length of motion blur
            angle: Angle of motion in degrees
            
        Returns:
            Motion blur kernel
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Create kernel
        kernel_size = length * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Calculate line coordinates
        center = kernel_size // 2
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        for i in range(-length, length + 1):
            x = int(center + i * cos_angle)
            y = int(center + i * sin_angle)
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # Normalize kernel
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        
        return kernel
    
    def wiener_filter_motion(self, image: np.ndarray, kernel: np.ndarray, 
                           noise_power: float = 0.01) -> np.ndarray:
        """
        Apply Wiener filter for motion deblurring.
        
        Args:
            image: Blurred input image
            kernel: Motion blur kernel
            noise_power: Estimated noise power
            
        Returns:
            Deblurred image
        """
        # Pad kernel to image size
        h, w = image.shape
        kernel_padded = np.zeros((h, w))
        kh, kw = kernel.shape
        start_h, start_w = (h - kh) // 2, (w - kw) // 2
        kernel_padded[start_h:start_h + kh, start_w:start_w + kw] = kernel
        
        # Convert to frequency domain
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel_padded)
        
        # Wiener filter
        kernel_conj = np.conj(kernel_fft)
        kernel_mag_sq = np.abs(kernel_fft) ** 2
        
        # Avoid division by zero
        denominator = kernel_mag_sq + noise_power
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        
        wiener_filter = kernel_conj / denominator
        
        # Apply filter and convert back
        result_fft = image_fft * wiener_filter
        result = np.fft.ifft2(result_fft).real
        
        return normalize_image(result)
    
    def inverse_filter_motion(self, image: np.ndarray, kernel: np.ndarray, 
                            threshold: float = 0.01) -> np.ndarray:
        """
        Apply inverse filter with threshold for motion deblurring.
        
        Args:
            image: Blurred input image
            kernel: Motion blur kernel
            threshold: Threshold to avoid division by small numbers
            
        Returns:
            Deblurred image
        """
        # Pad kernel to image size
        h, w = image.shape
        kernel_padded = np.zeros((h, w))
        kh, kw = kernel.shape
        start_h, start_w = (h - kh) // 2, (w - kw) // 2
        kernel_padded[start_h:start_h + kh, start_w:start_w + kw] = kernel
        
        # Convert to frequency domain
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel_padded)
        
        # Apply threshold to avoid division by small numbers
        kernel_fft = np.where(np.abs(kernel_fft) < threshold, threshold, kernel_fft)
        
        # Inverse filter
        result_fft = image_fft / kernel_fft
        result = np.fft.ifft2(result_fft).real
        
        return normalize_image(result)
    
    def lucy_richardson_motion(self, image: np.ndarray, kernel: np.ndarray, 
                             iterations: int = 30) -> np.ndarray:
        """
        Apply Lucy-Richardson deconvolution for motion deblurring.
        
        Args:
            image: Blurred input image
            kernel: Motion blur kernel
            iterations: Number of iterations
            
        Returns:
            Deblurred image
        """
        # Convert to float
        image = image.astype(np.float64)
        kernel = kernel.astype(np.float64)
        
        # Initialize estimate
        estimate = np.copy(image)
        
        # Flip kernel for correlation
        kernel_flipped = np.flip(kernel)
        
        for i in range(iterations):
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
        
        return normalize_image(estimate)
    
    def remove_motion_blur(self, image: np.ndarray, angle: float, length: int,
                          method: str = 'wiener', iterations: int = 30,
                          noise_power: float = 0.01) -> np.ndarray:
        """
        Main motion deblurring function.
        
        Args:
            image: Input blurred image
            angle: Motion blur angle in degrees
            length: Motion blur length in pixels
            method: Deblurring method ('wiener', 'inverse', or 'lucy_richardson')
            iterations: Number of iterations for iterative methods
            noise_power: Noise power for Wiener filter
            
        Returns:
            Deblurred image
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            is_color = True
            gray_image = rgb_to_grayscale(image)
        else:
            is_color = False
            gray_image = image.copy()
        
        # Create motion kernel
        kernel = self.create_motion_kernel(length, angle)
        
        # Apply deblurring
        if method == 'wiener':
            result = self.wiener_filter_motion(gray_image, kernel, noise_power)
        elif method == 'inverse':
            result = self.inverse_filter_motion(gray_image, kernel)
        elif method == 'lucy_richardson':
            result = self.lucy_richardson_motion(gray_image, kernel, iterations)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert back to color if needed
        if is_color:
            result_color = np.zeros_like(image)
            for i in range(3):
                if method == 'wiener':
                    result_color[:, :, i] = self.wiener_filter_motion(
                        image[:, :, i], kernel, noise_power)
                elif method == 'inverse':
                    result_color[:, :, i] = self.inverse_filter_motion(
                        image[:, :, i], kernel)
                else:
                    result_color[:, :, i] = self.lucy_richardson_motion(
                        image[:, :, i], kernel, iterations)
            result = result_color
        
        processing_time = time.time() - start_time
        print(f"Motion deblurring completed in {processing_time:.2f} seconds")
        
        return result
    
    def estimate_motion_parameters(self, image: np.ndarray) -> Tuple[float, int]:
        """
        Estimate motion blur parameters from an image.
        
        Args:
            image: Blurred input image
            
        Returns:
            Tuple of (angle, length) estimates
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = rgb_to_grayscale(image)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        
        # Use Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None and len(lines) > 0:
            # Find the most common angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.rad2deg(theta) - 90  # Convert to motion blur angle
                angles.append(angle)
            
            # Use the median angle
            estimated_angle = np.median(angles)
            
            # Estimate length based on image gradient
            gradient = np.gradient(gray.astype(np.float64))
            gradient_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
            estimated_length = int(np.mean(gradient_mag) * 2)
            estimated_length = max(5, min(estimated_length, 50))  # Reasonable bounds
        else:
            # Default estimates
            estimated_angle = 0.0
            estimated_length = 15
        
        return estimated_angle, estimated_length
