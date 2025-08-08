#!/usr/bin/env python3
"""
Demo script for testing both expansion and reduction functionality
"""

import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path


class SeamCarvingExpandDemo:
    def energy_function(self, image):
        """Calculate energy function using gradient magnitude"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Energy is the magnitude of gradients
        energy = np.sqrt(grad_x**2 + grad_y**2)
        return energy
    
    def find_vertical_seam(self, energy):
        """Find the minimum energy vertical seam using dynamic programming"""
        rows, cols = energy.shape
        dp = np.copy(energy)
        
        # Fill the DP table
        for i in range(1, rows):
            for j in range(cols):
                # Consider three possible paths from previous row
                candidates = []
                
                # From directly above
                candidates.append(dp[i-1, j])
                
                # From top-left (if exists)
                if j > 0:
                    candidates.append(dp[i-1, j-1])
                
                # From top-right (if exists)
                if j < cols - 1:
                    candidates.append(dp[i-1, j+1])
                
                dp[i, j] += min(candidates)
        
        # Backtrack to find the seam
        seam = []
        j = np.argmin(dp[-1])
        
        for i in range(rows - 1, -1, -1):
            seam.append(j)
            
            if i > 0:
                candidates = [dp[i-1, j]]
                indices = [j]
                
                if j > 0:
                    candidates.append(dp[i-1, j-1])
                    indices.append(j-1)
                
                if j < cols - 1:
                    candidates.append(dp[i-1, j+1])
                    indices.append(j+1)
                
                j = indices[np.argmin(candidates)]
        
        seam.reverse()
        return seam
    
    def insert_vertical_seam(self, image, seam):
        """Insert a vertical seam into the image by averaging adjacent pixels"""
        rows, cols, channels = image.shape
        result = np.zeros((rows, cols + 1, channels), dtype=image.dtype)
        
        for i in range(rows):
            j = seam[i]
            
            # Copy pixels before seam
            result[i, :j] = image[i, :j]
            
            # Insert new pixel as average of seam pixel and adjacent pixels
            if j == 0:
                # At left edge, duplicate the seam pixel
                result[i, j] = image[i, j]
            elif j == cols - 1:
                # At right edge, duplicate the seam pixel
                result[i, j] = image[i, j]
            else:
                # Average the seam pixel with its neighbors
                left_pixel = image[i, j-1] if j > 0 else image[i, j]
                right_pixel = image[i, j+1] if j < cols - 1 else image[i, j]
                result[i, j] = ((left_pixel.astype(np.float32) + 
                               image[i, j].astype(np.float32) + 
                               right_pixel.astype(np.float32)) / 3).astype(image.dtype)
            
            # Insert the original seam pixel
            result[i, j+1] = image[i, j]
            
            # Copy pixels after seam
            result[i, j+2:] = image[i, j+1:]
        
        return result
    
    def test_expansion(self, image_path, percentage):
        """Test image expansion functionality"""
        # Load image
        original_image = Image.open(image_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(original_image)
        
        # Calculate dimensions
        height, width, channels = image_array.shape
        target_width = int(width * percentage / 100)
        seams_to_add = target_width - width
        
        print(f"Testing EXPANSION:")
        print(f"Original size: {width}x{height}")
        print(f"Target width: {target_width} ({percentage}%)")
        print(f"Seams to add: {seams_to_add}")
        
        # Calculate region boundaries (first 25% and last 25%)
        first_quarter = width // 4
        last_quarter_start = width - first_quarter
        
        # Split seams between two regions
        seams_first_region = seams_to_add // 2
        seams_last_region = seams_to_add - seams_first_region
        
        print(f"Region boundaries: 0:{first_quarter}, {first_quarter}:{last_quarter_start}, {last_quarter_start}:{width}")
        
        # Process first 25%
        print(f"Expanding first region by {seams_first_region} seams...")
        first_region = image_array[:, 0:first_quarter].copy()
        
        for seam_idx in range(seams_first_region):
            energy = self.energy_function(first_region)
            seam = self.find_vertical_seam(energy)
            first_region = self.insert_vertical_seam(first_region, seam)
            
            if (seam_idx + 1) % 5 == 0:
                print(f"  Added {seam_idx + 1}/{seams_first_region} seams to first region")
        
        # Process last 25%
        print(f"Expanding last region by {seams_last_region} seams...")
        last_region = image_array[:, last_quarter_start:width].copy()
        
        for seam_idx in range(seams_last_region):
            energy = self.energy_function(last_region)
            seam = self.find_vertical_seam(energy)
            last_region = self.insert_vertical_seam(last_region, seam)
            
            if (seam_idx + 1) % 5 == 0:
                print(f"  Added {seam_idx + 1}/{seams_last_region} seams to last region")
        
        # Get middle region (unchanged)
        middle_region = image_array[:, first_quarter:last_quarter_start]
        
        # Combine all regions
        result = np.concatenate([first_region, middle_region, last_region], axis=1)
        
        print(f"Final size: {result.shape[1]}x{result.shape[0]}")
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(result.astype(np.uint8))
        
        # Save the image
        path_obj = Path(image_path)
        output_filename = f"{path_obj.stem}_expanded_width{path_obj.suffix}"
        output_path = path_obj.parent / output_filename
        
        processed_image.save(output_path)
        print(f"Saved: {output_path}")
        
        return processed_image


if __name__ == "__main__":
    demo = SeamCarvingExpandDemo()
    
    # Test with the created test image
    test_image_path = "/home/rich/MyCoding/image_processors/test_image.png"
    
    if os.path.exists(test_image_path):
        print("Testing seam carving EXPANSION with test image...")
        demo.test_expansion(test_image_path, 125.0)  # Expand to 125% of width
        print("Expansion demo completed!")
    else:
        print("Test image not found. Please run test_seam_carving.py first.")
