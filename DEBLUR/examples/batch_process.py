"""
Batch processing script for deblurring multiple images.
"""

import os
import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.image_utils import load_image, save_image
from deblur.gaussian_deblur import GaussianDeblur
from deblur.motion_deblur import MotionDeblur


def process_directory(input_dir, output_dir, method='gaussian', **kwargs):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        method: Deblurring method ('gaussian' or 'motion')
        **kwargs: Additional parameters for deblurring
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize deblurrer
    if method == 'gaussian':
        deblurrer = GaussianDeblur()
    elif method == 'motion':
        deblurrer = MotionDeblur()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Processing with {method} deblurring...")
    
    start_time = time.time()
    processed_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"Processing ({i}/{len(image_files)}): {image_file.name}")
            
            # Load image
            image = load_image(str(image_file))
            
            # Apply deblurring
            if method == 'gaussian':
                result = deblurrer.deblur_image(image, **kwargs)
            else:
                result = deblurrer.remove_motion_blur(image, **kwargs)
            
            # Save result
            output_file = output_path / f"deblurred_{image_file.name}"
            save_image(result, str(output_file))
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nBatch processing completed!")
    print(f"Processed: {processed_count}/{len(image_files)} images")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/processed_count:.2f} seconds")


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Batch Image Deblurring')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for processed images')
    parser.add_argument('--method', '-m', choices=['gaussian', 'motion'], 
                       default='gaussian', help='Deblurring method')
    
    # Gaussian parameters
    parser.add_argument('--kernel-size', '-k', type=int, default=15,
                       help='Kernel size for Gaussian deblur')
    parser.add_argument('--iterations', '-n', type=int, default=30,
                       help='Number of iterations')
    parser.add_argument('--deblur-method', default='richardson_lucy',
                       choices=['richardson_lucy', 'wiener'],
                       help='Gaussian deblur algorithm')
    
    # Motion parameters
    parser.add_argument('--angle', '-a', type=float, default=0,
                       help='Motion blur angle (degrees)')
    parser.add_argument('--length', '-l', type=int, default=20,
                       help='Motion blur length')
    parser.add_argument('--motion-method', default='wiener',
                       choices=['wiener', 'inverse', 'lucy_richardson'],
                       help='Motion deblur algorithm')
    
    args = parser.parse_args()
    
    # Prepare parameters based on method
    if args.method == 'gaussian':
        params = {
            'kernel_size': args.kernel_size,
            'iterations': args.iterations,
            'method': args.deblur_method
        }
    else:
        params = {
            'angle': args.angle,
            'length': args.length,
            'method': args.motion_method,
            'iterations': args.iterations if args.motion_method == 'lucy_richardson' else None
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
    
    # Process directory
    process_directory(args.input_dir, args.output_dir, args.method, **params)


if __name__ == '__main__':
    main()
