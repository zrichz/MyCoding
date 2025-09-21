#!/usr/bin/env python3
"""
DEBLUR - Image Deblurring Application
Main entry point for the deblurring application.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.gui.main_window import DeblurApp
from src.deblur.gaussian_deblur import GaussianDeblur
from src.deblur.motion_deblur import MotionDeblur
from src.utils.image_utils import load_image, save_image


def main():
    parser = argparse.ArgumentParser(description='DEBLUR - Image Deblurring Tool')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--input', '-i', help='Input image path')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--method', '-m', choices=['gaussian', 'motion'], 
                       default='gaussian', help='Deblurring method')
    parser.add_argument('--kernel-size', '-k', type=int, default=15, 
                       help='Kernel size for Gaussian deblur')
    parser.add_argument('--iterations', '-n', type=int, default=30, 
                       help='Number of iterations')
    parser.add_argument('--downsample', '-d', choices=['auto', '1', '2', '4'], 
                       default='auto', help='Downsample factor (auto, 1, 2, or 4)')
    parser.add_argument('--no-progress', action='store_true', 
                       help='Disable progress output')
    parser.add_argument('--angle', '-a', type=float, default=0, 
                       help='Motion blur angle (degrees)')
    parser.add_argument('--length', '-l', type=int, default=20, 
                       help='Motion blur length')
    
    args = parser.parse_args()
    
    if args.gui or (not args.input):
        # Launch GUI
        app = DeblurApp()
        app.run()
    else:
        # Command line processing
        if not args.output:
            args.output = args.input.replace('.', '_deblurred.')
        
        print(f"Loading image: {args.input}")
        image = load_image(args.input)
        
        if args.method == 'gaussian':
            # Parse downsample setting
            if args.downsample == 'auto':
                auto_downsample = True
                downsample_factor = None
            else:
                auto_downsample = False
                downsample_factor = int(args.downsample)
            
            deblurrer = GaussianDeblur()
            result = deblurrer.deblur_image(image, args.kernel_size, args.iterations,
                                          auto_downsample=auto_downsample,
                                          downsample_factor=downsample_factor,
                                          show_progress=not args.no_progress)
        elif args.method == 'motion':
            deblurrer = MotionDeblur()
            result = deblurrer.remove_motion_blur(image, args.angle, args.length)
        
        print(f"Saving result: {args.output}")
        save_image(result, args.output)
        print("Deblurring complete!")


if __name__ == '__main__':
    main()
