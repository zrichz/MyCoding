"""
Unit tests for the DEBLUR project.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.image_utils import load_image, save_image, rgb_to_grayscale, normalize_image
from deblur.gaussian_deblur import GaussianDeblur
from deblur.motion_deblur import MotionDeblur


class TestImageUtils(unittest.TestCase):
    """Test image utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_image_rgb = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
    
    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        gray = rgb_to_grayscale(self.test_image_rgb)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 120))
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Test with float image
        float_image = np.random.random((50, 60)) * 100 + 50
        normalized = normalize_image(float_image)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 255)


class TestGaussianDeblur(unittest.TestCase):
    """Test Gaussian deblurring algorithms."""
    
    def setUp(self):
        """Set up test data."""
        self.deblurrer = GaussianDeblur()
        self.test_image = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
    
    def test_create_gaussian_kernel(self):
        """Test Gaussian kernel creation."""
        kernel = self.deblurrer.create_gaussian_kernel(15, 2.0)
        self.assertEqual(kernel.shape, (15, 15))
        self.assertAlmostEqual(kernel.sum(), 1.0, places=6)
        
        # Check symmetry
        self.assertTrue(np.allclose(kernel, kernel.T))
    
    def test_deblur_grayscale(self):
        """Test deblurring on grayscale image."""
        result = self.deblurrer.deblur_image(self.test_image, kernel_size=9, iterations=5, show_progress=False)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_deblur_color(self):
        """Test deblurring on color image."""
        color_image = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        result = self.deblurrer.deblur_image(color_image, kernel_size=9, iterations=5, show_progress=False)
        self.assertEqual(result.shape, color_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_downsample_functionality(self):
        """Test downsampling and upsampling functionality."""
        # Test downsampling decision
        large_image = np.random.randint(0, 255, (1000, 800), dtype=np.uint8)
        should_downsample, factor = self.deblurrer._should_downsample(large_image)
        self.assertTrue(should_downsample)
        self.assertIn(factor, [2, 4])
        
        # Test actual downsampling
        downsampled = self.deblurrer._downsample_image(large_image, 2)
        self.assertEqual(downsampled.shape, (500, 400))
        
        # Test upsampling
        upsampled = self.deblurrer._upsample_image(downsampled, large_image.shape, 2)
        self.assertEqual(upsampled.shape, large_image.shape)
    
    def test_deblur_with_downsampling(self):
        """Test deblurring with automatic downsampling on large image."""
        large_image = np.random.randint(0, 255, (1000, 800), dtype=np.uint8)
        result = self.deblurrer.deblur_image(large_image, kernel_size=9, iterations=3, 
                                           auto_downsample=True, show_progress=False)
        self.assertEqual(result.shape, large_image.shape)
        self.assertEqual(result.dtype, np.uint8)


class TestMotionDeblur(unittest.TestCase):
    """Test motion deblurring algorithms."""
    
    def setUp(self):
        """Set up test data."""
        self.deblurrer = MotionDeblur()
        self.test_image = np.random.randint(0, 255, (100, 120), dtype=np.uint8)
    
    def test_create_motion_kernel(self):
        """Test motion kernel creation."""
        kernel = self.deblurrer.create_motion_kernel(10, 45)
        self.assertEqual(kernel.shape, (21, 21))  # length * 2 + 1
        self.assertAlmostEqual(kernel.sum(), 1.0, places=6)
    
    def test_motion_deblur_grayscale(self):
        """Test motion deblurring on grayscale image."""
        result = self.deblurrer.remove_motion_blur(
            self.test_image, angle=30, length=10, method='wiener')
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_motion_deblur_color(self):
        """Test motion deblurring on color image."""
        color_image = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        result = self.deblurrer.remove_motion_blur(
            color_image, angle=30, length=10, method='wiener')
        self.assertEqual(result.shape, color_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_estimate_motion_parameters(self):
        """Test motion parameter estimation."""
        angle, length = self.deblurrer.estimate_motion_parameters(self.test_image)
        self.assertIsInstance(angle, (int, float))
        self.assertIsInstance(length, (int, float))
        self.assertTrue(-90 <= angle <= 90)
        self.assertTrue(5 <= length <= 50)


if __name__ == '__main__':
    unittest.main()
