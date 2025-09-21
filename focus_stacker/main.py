"""
Image Stacking Tool
==================
Advanced focus stacking application with multiple algorithms and modern GUI.

Features:
- Multiple focus stacking algorithms (Laplacian Pyramid, Gradient-based with smoothing, Variance-based)
- Advanced image alignment (ECC, Feature-based, Phase Correlation)
- Modern GUI with 800x800 image preview
- Batch processing support
- Multiple export formats

Author: AI Assistant
License: MIT
"""

import sys
import os
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gui import FocusStackerGUI
    from focus_stacking_algorithms import FocusStackingAlgorithms
    from image_alignment import ImageAligner, QualityAssessment
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install opencv-python numpy pillow scipy scikit-image matplotlib customtkinter tqdm")
    sys.exit(1)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('focus_stacker.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


class FocusStackerCLI:
    """Command-line interface for focus stacking."""
    
    @staticmethod
    def print_usage():
        """Print usage information."""
        print("""
World's Best Focus Stacker - Command Line Interface

Usage:
    python main.py [options] <image_files>
    python main.py --gui

Options:
    --gui                   Launch graphical interface
    --method METHOD         Stacking method (laplacian, gradient, variance, average)
    --align METHOD          Alignment method (auto, ecc, feature, phase, none)
    --output FILE           Output filename
    --levels N              Pyramid levels for Laplacian method (default: 5)
    --sigma FLOAT           Gaussian sigma (default: 1.0)
    --quality               Show quality assessment
    --help                  Show this help

Examples:
    python main.py --gui
    python main.py *.jpg --method laplacian --output stacked.png
    python main.py img1.jpg img2.jpg img3.jpg --align ecc --quality
        """)
    
    @staticmethod
    def run_cli(args):
        """Run command-line interface."""
        import argparse
        import glob
        import cv2
        
        parser = argparse.ArgumentParser(description="Focus Stacker CLI")
        parser.add_argument('files', nargs='*', help='Input image files')
        parser.add_argument('--gui', action='store_true', help='Launch GUI')
        parser.add_argument('--method', default='laplacian', 
                          choices=['laplacian', 'gradient', 'variance', 'average'],
                          help='Stacking method')
        parser.add_argument('--align', default='auto',
                          choices=['auto', 'ecc', 'feature', 'phase', 'none'],
                          help='Alignment method')
        parser.add_argument('--output', default='stacked_result.png',
                          help='Output filename')
        parser.add_argument('--levels', type=int, default=5,
                          help='Pyramid levels')
        parser.add_argument('--sigma', type=float, default=1.0,
                          help='Gaussian sigma')
        parser.add_argument('--quality', action='store_true',
                          help='Show quality assessment')
        
        parsed_args = parser.parse_args(args)
        
        if parsed_args.gui or not parsed_args.files:
            # Launch GUI
            app = FocusStackerGUI()
            app.run()
            return
        
        # Expand glob patterns
        image_files = []
        for pattern in parsed_args.files:
            matches = glob.glob(pattern)
            if matches:
                image_files.extend(matches)
            else:
                image_files.append(pattern)
        
        if not image_files:
            print("Error: No image files specified")
            FocusStackerCLI.print_usage()
            return
        
        # Load images
        print(f"Loading {len(image_files)} images...")
        images = []
        for file_path in image_files:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ Failed to load {file_path}")
        
        if not images:
            print("Error: No valid images loaded")
            return
        
        print(f"Successfully loaded {len(images)} images")
        
        # Align images
        if parsed_args.align != 'none':
            print(f"Aligning images using {parsed_args.align} method...")
            try:
                if parsed_args.align == 'auto':
                    aligned_images = ImageAligner.auto_align(images)
                elif parsed_args.align == 'ecc':
                    aligned_images = ImageAligner.align_images_ecc(images)
                elif parsed_args.align == 'feature':
                    aligned_images = ImageAligner.align_images_feature_based(images)
                elif parsed_args.align == 'phase':
                    aligned_images = ImageAligner.align_images_phase_correlation(images)
                else:
                    aligned_images = images
                
                print("  ✓ Alignment completed")
                images = aligned_images
            except Exception as e:
                print(f"  ✗ Alignment failed: {e}")
                print("  Proceeding with original images...")
        
        # Stack images
        print(f"Stacking images using {parsed_args.method} method...")
        try:
            if parsed_args.method == 'laplacian':
                result = FocusStackingAlgorithms.laplacian_pyramid_stack(
                    images, levels=parsed_args.levels, sigma=parsed_args.sigma)
            elif parsed_args.method == 'gradient':
                result = FocusStackingAlgorithms.gradient_based_stack(images)
            elif parsed_args.method == 'variance':
                result = FocusStackingAlgorithms.variance_based_stack(images)
            elif parsed_args.method == 'average':
                result = FocusStackingAlgorithms.average_stack(images)
            else:
                raise ValueError(f"Unknown method: {parsed_args.method}")
            
            print("  ✓ Stacking completed")
            
        except Exception as e:
            print(f"  ✗ Stacking failed: {e}")
            return
        
        # Quality assessment
        if parsed_args.quality:
            print("Assessing quality...")
            try:
                metrics = QualityAssessment.assess_stack_quality(images, result)
                print(f"  Stacked Focus Measure: {metrics['stacked_focus_measure']:.2f}")
                print(f"  Max Original Focus: {metrics['max_original_focus']:.2f}")
                print(f"  Improvement Ratio: {metrics['improvement_ratio']:.2f}x")
                print(f"  Mean Gradient: {metrics['mean_gradient']:.2f}")
                print(f"  Variance: {metrics['variance']:.2f}")
            except Exception as e:
                print(f"  ✗ Quality assessment failed: {e}")
        
        # Save result
        print(f"Saving result to {parsed_args.output}...")
        try:
            cv2.imwrite(parsed_args.output, result)
            print(f"  ✓ Saved successfully to {parsed_args.output}")
        except Exception as e:
            print(f"  ✗ Save failed: {e}")


def main():
    """Main entry point."""
    setup_logging()
    
    # Print banner
    print("=" * 60)
    print("          World's Best Focus Stacker")
    print("    Advanced Focus Stacking with Multiple Algorithms")
    print("=" * 60)
    
    if len(sys.argv) == 1 or '--gui' in sys.argv:
        # Launch GUI if no arguments or --gui specified
        try:
            app = FocusStackerGUI()
            app.run()
        except Exception as e:
            logging.error(f"GUI failed to start: {e}")
            print(f"Error: Failed to start GUI - {e}")
            print("You can still use the command-line interface.")
            FocusStackerCLI.print_usage()
    elif '--help' in sys.argv or '-h' in sys.argv:
        FocusStackerCLI.print_usage()
    else:
        # Run CLI
        FocusStackerCLI.run_cli(sys.argv[1:])


if __name__ == "__main__":
    main()