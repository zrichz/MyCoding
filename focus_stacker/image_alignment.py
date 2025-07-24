"""
Image Alignment Module
======================
Advanced image alignment algorithms for focus stacking.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable
import logging
import time

logger = logging.getLogger(__name__)


class ImageAligner:
    """Advanced image alignment for focus stacking."""
    
    @staticmethod
    def align_images_ecc(images: List[np.ndarray], 
                        reference_idx: int = 0,
                        warp_mode: int = cv2.MOTION_TRANSLATION,
                        max_iterations: int = 1000,
                        termination_eps: float = 1e-6,
                        use_proxy: bool = True,
                        proxy_scale: float = 0.25,
                        progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[List[np.ndarray], float]:
        """
        Align images using Enhanced Correlation Coefficient (ECC) algorithm.
        
        Args:
            images: List of images to align
            reference_idx: Index of reference image
            warp_mode: Type of transformation (TRANSLATION, EUCLIDEAN, AFFINE, HOMOGRAPHY)
            max_iterations: Maximum iterations for ECC
            termination_eps: Termination threshold
            use_proxy: Whether to use downscaled proxy images for faster alignment
            proxy_scale: Scale factor for proxy images (0.1-1.0, smaller = faster)
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Tuple of (aligned images, alignment time in seconds)
        """
        start_time = time.time()
        
        if not images or reference_idx >= len(images):
            raise ValueError("Invalid images or reference index")
        
        if progress_callback:
            alignment_type = f"ECC alignment ({'proxy-based' if use_proxy else 'full-resolution'})"
            progress_callback(0.1, f"Starting {alignment_type}...")
        
        reference = images[reference_idx]
        if len(reference.shape) == 3:
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            reference_gray = reference
        
        # Create proxy images if requested
        if use_proxy and proxy_scale < 1.0:
            if progress_callback:
                progress_callback(0.15, f"Creating proxy images (scale: {proxy_scale:.2f})...")
            
            # Create downscaled reference
            proxy_height = int(reference_gray.shape[0] * proxy_scale)
            proxy_width = int(reference_gray.shape[1] * proxy_scale)
            reference_proxy = cv2.resize(reference_gray, (proxy_width, proxy_height), interpolation=cv2.INTER_AREA)
            
            # Scale factor for transforming proxy coordinates to full resolution
            scale_factor = 1.0 / proxy_scale
        else:
            reference_proxy = reference_gray
            scale_factor = 1.0
        
        aligned_images = [img.copy() for img in images]
        total_images = len(images) - 1  # Exclude reference image
        processed_images = 0
        
        # Define transformation matrix size based on warp mode
        if warp_mode == cv2.MOTION_TRANSLATION:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif warp_mode == cv2.MOTION_EUCLIDEAN:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif warp_mode == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:  # HOMOGRAPHY
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        
        # Set termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                   max_iterations, termination_eps)
        
        for i, img in enumerate(images):
            if i == reference_idx:
                continue
                
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # Create proxy image if using proxy-based alignment
            if use_proxy and proxy_scale < 1.0:
                img_proxy = cv2.resize(img_gray, (reference_proxy.shape[1], reference_proxy.shape[0]), 
                                     interpolation=cv2.INTER_AREA)
            else:
                img_proxy = img_gray
            
            if progress_callback:
                progress = 0.2 + (processed_images / total_images) * 0.7
                alignment_method = "proxy ECC" if use_proxy and proxy_scale < 1.0 else "full ECC"
                progress_callback(progress, f"Aligning image {i+1}/{len(images)} using {alignment_method}...")
            
            try:
                # Find transformation using proxy images
                # NOTE: findTransformECC finds transformation from img_proxy TO reference_proxy
                # We want to align img TO reference, so we swap the arguments
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    _, proxy_warp_matrix = cv2.findTransformECC(
                        img_proxy, reference_proxy, warp_matrix, warp_mode, criteria)
                    
                    # Scale transformation matrix for full resolution
                    if use_proxy and proxy_scale < 1.0:
                        full_warp_matrix = ImageAligner._scale_homography_matrix(proxy_warp_matrix, scale_factor)
                    else:
                        full_warp_matrix = proxy_warp_matrix
                    
                    aligned = cv2.warpPerspective(
                        img, full_warp_matrix, (img.shape[1], img.shape[0]))
                else:
                    _, proxy_warp_matrix = cv2.findTransformECC(
                        img_proxy, reference_proxy, warp_matrix, warp_mode, criteria)
                    
                    # Scale transformation matrix for full resolution
                    if use_proxy and proxy_scale < 1.0:
                        full_warp_matrix = ImageAligner._scale_affine_matrix(proxy_warp_matrix, scale_factor)
                    else:
                        full_warp_matrix = proxy_warp_matrix
                    
                    aligned = cv2.warpAffine(
                        img, full_warp_matrix, (img.shape[1], img.shape[0]))
                
                aligned_images[i] = aligned
                processed_images += 1
                
            except cv2.error as e:
                logger.warning(f"ECC alignment failed for image {i}: {e}")
                processed_images += 1
                # Keep original image if alignment fails
                continue
        
        alignment_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, f"ECC alignment complete! Aligned {len(images)} images in {alignment_time:.2f} seconds")
        
        return aligned_images, alignment_time
    
    @staticmethod
    def _scale_affine_matrix(matrix: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale an affine transformation matrix from proxy to full resolution."""
        scaled_matrix = matrix.copy()
        # Scale translation components
        scaled_matrix[0, 2] *= scale_factor
        scaled_matrix[1, 2] *= scale_factor
        return scaled_matrix
    
    @staticmethod
    def _scale_homography_matrix(matrix: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale a homography transformation matrix from proxy to full resolution."""
        # Create scaling matrix
        scale_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Apply scaling: S * H * S^-1
        inv_scale_matrix = np.array([
            [1.0/scale_factor, 0, 0],
            [0, 1.0/scale_factor, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return scale_matrix @ matrix @ inv_scale_matrix
    
    @staticmethod
    def align_images_feature_based(images: List[np.ndarray], 
                                  reference_idx: int = 0,
                                  max_features: int = 5000,
                                  match_threshold: float = 0.7,
                                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[List[np.ndarray], float]:
        """
        Align images using feature-based matching (SIFT/ORB).
        
        Args:
            images: List of images to align
            reference_idx: Index of reference image
            max_features: Maximum number of features to detect
            match_threshold: Matching threshold for feature matching
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Tuple of (aligned images, alignment time in seconds)
        """
        start_time = time.time()
        
        if not images or reference_idx >= len(images):
            raise ValueError("Invalid images or reference index")
        
        if progress_callback:
            progress_callback(0.1, "Starting feature-based alignment...")
        
        reference = images[reference_idx]
        if len(reference.shape) == 3:
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            reference_gray = reference
        
        # Try SIFT first, fall back to ORB if not available
        try:
            detector = cv2.SIFT_create(nfeatures=max_features)
        except AttributeError:
            detector = cv2.ORB_create(nfeatures=max_features)
        
        if progress_callback:
            progress_callback(0.2, "Detecting features in reference image...")
        
        # Detect keypoints and descriptors for reference
        ref_kp, ref_desc = detector.detectAndCompute(reference_gray, None)
        
        if ref_desc is None:
            logger.warning("No features detected in reference image")
            if progress_callback:
                progress_callback(1.0, "Feature-based alignment failed - no features in reference")
            return images, 0.0
        
        aligned_images = [img.copy() for img in images]
        total_images = len(images) - 1  # Exclude reference image
        processed_images = 0
        
        # Create matcher
        if detector.__class__.__name__ == 'SIFT':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        for i, img in enumerate(images):
            if i == reference_idx:
                continue
            
            if progress_callback:
                progress = 0.3 + (processed_images / total_images) * 0.6
                progress_callback(progress, f"Aligning image {i+1}/{len(images)} using features...")
            
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # Detect keypoints and descriptors
            kp, desc = detector.detectAndCompute(img_gray, None)
            
            if desc is None:
                logger.warning(f"No features detected in image {i}")
                processed_images += 1
                continue
            
            # Match features
            matches = matcher.knnMatch(ref_desc, desc, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < match_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                logger.warning(f"Insufficient matches for image {i}: {len(good_matches)}")
                processed_images += 1
            
            # Extract matched points
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            try:
                M, mask = cv2.findHomography(dst_pts, src_pts, 
                                           cv2.RANSAC, 5.0)
                if M is not None:
                    aligned = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
                    aligned_images[i] = aligned
                else:
                    logger.warning(f"Homography estimation failed for image {i}")
            except cv2.error as e:
                logger.warning(f"Homography estimation failed for image {i}: {e}")
            
            processed_images += 1
        
        alignment_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, f"Feature-based alignment complete! Aligned {len(images)} images in {alignment_time:.2f} seconds")
        
        return aligned_images, alignment_time
    
    @staticmethod
    def align_images_phase_correlation(images: List[np.ndarray], 
                                     reference_idx: int = 0,
                                     progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[List[np.ndarray], float]:
        """
        Align images using phase correlation (translation only).
        
        Args:
            images: List of images to align
            reference_idx: Index of reference image
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Tuple of (aligned images, alignment time in seconds)
        """
        start_time = time.time()
        
        if not images or reference_idx >= len(images):
            raise ValueError("Invalid images or reference index")
        
        if progress_callback:
            progress_callback(0.1, "Starting phase correlation alignment...")
        
        reference = images[reference_idx]
        if len(reference.shape) == 3:
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            reference_gray = reference
        
        aligned_images = [img.copy() for img in images]
        total_images = len(images) - 1  # Exclude reference image
        processed_images = 0
        
        for i, img in enumerate(images):
            if i == reference_idx:
                continue
            
            if progress_callback:
                progress = 0.2 + (processed_images / total_images) * 0.7
                progress_callback(progress, f"Aligning image {i+1}/{len(images)} using phase correlation...")
            
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # Calculate phase correlation
            # phaseCorrelate(img1, img2) returns shift to align img2 to img1
            # We want to align img to reference, so: phaseCorrelate(reference, img)
            shift, _ = cv2.phaseCorrelate(reference_gray.astype(np.float32),
                                        img_gray.astype(np.float32))
            
            # Create translation matrix - shift is already in the correct direction
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            
            # Apply translation
            aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            aligned_images[i] = aligned
            
            processed_images += 1
        
        alignment_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, f"Phase correlation alignment complete! Aligned {len(images)} images in {alignment_time:.2f} seconds")
        
        return aligned_images, alignment_time
    
    @staticmethod
    def auto_align(images: List[np.ndarray], 
                  reference_idx: int = 0,
                  method: str = 'auto',
                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[List[np.ndarray], float]:
        """
        Automatically choose and apply the best alignment method.
        
        Args:
            images: List of images to align
            reference_idx: Index of reference image
            method: Alignment method ('auto', 'ecc', 'feature', 'phase')
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Tuple of (aligned images, alignment time in seconds)
        """
        start_time = time.time()
        
        if not images:
            return images, 0.0
        
        if progress_callback:
            progress_callback(0.05, f"Starting auto alignment with method: {method}")
        
        if method == 'auto':
            # Try methods in order of robustness vs speed
            methods = ['ecc', 'feature', 'phase']
            for i, align_method in enumerate(methods):
                try:
                    if progress_callback:
                        progress_callback(0.1 + i * 0.3, f"Trying {align_method} alignment...")
                    
                    result, align_time = ImageAligner.auto_align(images, reference_idx, align_method, progress_callback)
                    total_time = time.time() - start_time
                    
                    if progress_callback:
                        progress_callback(1.0, f"Auto alignment successful using {align_method} method in {total_time:.2f} seconds")
                    
                    return result, total_time
                except Exception as e:
                    logger.warning(f"Alignment method {align_method} failed: {e}")
                    continue
            
            # If all methods fail, return original images
            logger.warning("All alignment methods failed, returning original images")
            if progress_callback:
                progress_callback(1.0, "All alignment methods failed, using original images")
            return images, time.time() - start_time
        
        elif method == 'ecc':
            return ImageAligner.align_images_ecc(images, reference_idx, 
                                               use_proxy=True, proxy_scale=0.25, 
                                               progress_callback=progress_callback)
        elif method == 'feature':
            return ImageAligner.align_images_feature_based(images, reference_idx, progress_callback=progress_callback)
        elif method == 'phase':
            return ImageAligner.align_images_phase_correlation(images, reference_idx, progress_callback=progress_callback)
        else:
            raise ValueError(f"Unknown alignment method: {method}")


class QualityAssessment:
    """Image quality assessment for focus stacking."""
    
    @staticmethod
    def calculate_focus_measure(image: np.ndarray, method: str = 'laplacian') -> float:
        """
        Calculate focus measure for an image.
        
        Args:
            image: Input image
            method: Focus measure method ('laplacian', 'gradient', 'variance')
            
        Returns:
            Focus measure value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'laplacian':
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()
        elif method == 'gradient':
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            return gradient_mag.mean()
        elif method == 'variance':
            return gray.var()
        else:
            raise ValueError(f"Unknown focus measure method: {method}")
    
    @staticmethod
    def assess_stack_quality(original_images: List[np.ndarray], 
                           stacked_image: np.ndarray) -> dict:
        """
        Assess the quality of a focus stacked result.
        
        Args:
            original_images: List of original images
            stacked_image: Focus stacked result
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Calculate focus measure for each original image
        original_focus = [QualityAssessment.calculate_focus_measure(img) 
                         for img in original_images]
        
        # Calculate focus measure for stacked image
        stacked_focus = QualityAssessment.calculate_focus_measure(stacked_image)
        
        metrics['original_focus_measures'] = original_focus
        metrics['stacked_focus_measure'] = stacked_focus
        metrics['max_original_focus'] = max(original_focus)
        metrics['improvement_ratio'] = stacked_focus / max(original_focus) if max(original_focus) > 0 else 0
        
        # Calculate sharpness metrics
        metrics['mean_gradient'] = QualityAssessment.calculate_focus_measure(
            stacked_image, 'gradient')
        metrics['variance'] = QualityAssessment.calculate_focus_measure(
            stacked_image, 'variance')
        
        return metrics
