#!/usr/bin/env python3
"""
Demo script for the Seam Carving Width Reducer application
This script demonstrates the non-GUI version of seam carving
"""

import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path


class SeamCarvingDemo:
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
    
    def remove_vertical_seam(self, image, seam):
        """Remove a vertical seam from the image"""
        rows, cols, channels = image.shape
        result = np.zeros((rows, cols - 1, channels), dtype=image.dtype)
        
        for i in range(rows):
            j = seam[i]
            result[i, :j] = image[i, :j]
            result[i, j:] = image[i, j+1:]
        
        return result
    
    def seam_carve_region(self, image, start_col, end_col, seams_to_remove):
        """Apply seam carving to a specific region of the image"""
        if seams_to_remove <= 0:
            return image[:, start_col:end_col]
        
        # Extract the region
        region = image[:, start_col:end_col].copy()
        
        print(f"Processing region {start_col}:{end_col}, removing {seams_to_remove} seams...")
        
        for seam_idx in range(seams_to_remove):
            # Calculate energy for current region
            energy = self.energy_function(region)
            
            # Find minimum energy seam
            seam = self.find_vertical_seam(energy)
            
            # Remove the seam
            region = self.remove_vertical_seam(region, seam)
            
            if (seam_idx + 1) % 10 == 0:
                print(f"  Removed {seam_idx + 1}/{seams_to_remove} seams")
        
        return region
    
    def process_image(self, image_path, percentage):
        """Process an image with seam carving"""
        # Load image
        original_image = Image.open(image_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(original_image)
        
        # Calculate dimensions
        height, width, channels = image_array.shape
        target_width = int(width * percentage / 100)
        seams_to_remove = width - target_width
        
        print(f"Original size: {width}x{height}")
        print(f"Target width: {target_width} ({percentage}%)")
        print(f"Seams to remove: {seams_to_remove}")
        
        # Calculate region boundaries (first 25% and last 25%)
        first_quarter = width // 4
        last_quarter_start = width - first_quarter
        
        # Split seams between two regions
        seams_first_region = seams_to_remove // 2
        seams_last_region = seams_to_remove - seams_first_region
        
        print(f"Region boundaries: 0:{first_quarter}, {first_quarter}:{last_quarter_start}, {last_quarter_start}:{width}")
        
        # Process first 25%
        first_region = self.seam_carve_region(
            image_array, 0, first_quarter, seams_first_region
        )
        
        # Process last 25%
        last_region = self.seam_carve_region(
            image_array, last_quarter_start, width, seams_last_region
        )
        
        # Get middle region (unchanged)
        middle_region = image_array[:, first_quarter:last_quarter_start]
        
        # Combine all regions
        result = np.concatenate([first_region, middle_region, last_region], axis=1)
        
        print(f"Final size: {result.shape[1]}x{result.shape[0]}")
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(result.astype(np.uint8))
        
        # Save the image
        path_obj = Path(image_path)
        output_filename = f"{path_obj.stem}_reduced_width{path_obj.suffix}"
        output_path = path_obj.parent / output_filename
        
        processed_image.save(output_path)
        print(f"Saved: {output_path}")
        
        return processed_image


if __name__ == "__main__":
    demo = SeamCarvingDemo()
    
    # Test with the created test image
    test_image_path = "/home/rich/MyCoding/image_processors/test_image.png"
    
    if os.path.exists(test_image_path):
        print("Testing seam carving with test image...")
        demo.process_image(test_image_path, 75.0)  # Reduce to 75% of width
        print("Demo completed!")
    else:
        print("Test image not found. Please run test_seam_carving.py first.")
