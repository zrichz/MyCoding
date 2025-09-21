"""
NCA Quick Animator - Command Line Version
=========================================

Quick command-line script to generate NCA animations without GUI.
Perfect for batch processing or automation.

Usage:
    python nca_quick_animator.py model.pth output.gif [options]

Examples:
    python nca_quick_animator.py my_model.pth evolution.gif
    python nca_quick_animator.py my_model.pth spiral.gif --init circle --steps 200
    python nca_quick_animator.py my_model.pth tiny.gif --init random_multi --size 16
    python nca_quick_animator.py my_model.pth small.gif --size 32
    python nca_quick_animator.py my_model.pth large.gif --size 512
"""

import argparse
import sys
import os
from pathlib import Path

# Import the animation generator
try:
    from nca_animator import NCAAnimationGenerator
except ImportError:
    print("Error: Could not import NCAAnimationGenerator")
    print("Make sure nca_animator.py is in the same directory.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Generate NCA animations from command line')
    
    # Required arguments
    parser.add_argument('model_path', help='Path to trained NCA model (.pth file)')
    parser.add_argument('output_path', help='Output GIF file path')
    
    # Optional arguments
    parser.add_argument('--init', choices=['center', 'random_single', 'random_multi', 'sparse', 'edge', 'circle'],
                       default='center', help='Initialization type (default: center)')
    parser.add_argument('--steps', type=int, default=128, help='Total animation steps (default: 128)')
    parser.add_argument('--interval', type=int, default=1, help='Frame capture interval (default: 1)')
    parser.add_argument('--size', type=int, choices=[16, 32, 64, 128, 256, 512], default=16, 
                       help='Output image size (default: 16). Small sizes are scaled up with nearest neighbor.')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--no-labels', action='store_true', help='Disable step number labels')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
        
    if not args.output_path.lower().endswith('.gif'):
        print("Warning: Output file should have .gif extension")
        
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("🎬 NCA Quick Animator")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_path}")
    print(f"Settings: {args.init} init, {args.steps} steps, {args.size}x{args.size}px")
    print()
    
    try:
        # Create generator
        generator = NCAAnimationGenerator()
        
        # Load model
        print("📥 Loading model...")
        if not generator.load_model(args.model_path):
            print("❌ Failed to load model")
            sys.exit(1)
            
        # Update animation parameters
        generator.animation_params.update({
            'total_steps': args.steps,
            'frame_interval': args.interval,
            'gif_duration': 33,  # Fixed at 30fps
            'image_size': args.size,
            'add_labels': not args.no_labels,
        })
        
        # Create seed
        print(f"🌱 Creating {args.init} seed...")
        grid_size = 128  # Internal processing size
        seed = generator.create_seed(grid_size, args.init, args.seed)
        
        # Generate frames
        print("🎨 Generating animation frames...")
        frames = generator.generate_animation_frames(seed, args.steps, args.interval)
        
        # Create GIF
        print("💾 Creating GIF...")
        generator.create_gif(args.output_path, optimize_for_social=True)
        
        print()
        print("🎉 Animation created successfully!")
        print(f"📄 File: {args.output_path}")
        print(f"📊 Frames: {len(frames)}")
        
        # File size info
        file_size = os.path.getsize(args.output_path) / (1024 * 1024)
        print(f"💾 Size: {file_size:.2f} MB")
        
        # Social media tips
        if file_size > 8:
            print("⚠️  Note: File size > 8MB may be too large for some social media platforms")
            print("   Consider reducing --steps, increasing --interval, or reducing --size")
        else:
            print("✅ File size is social media friendly!")
            
    except KeyboardInterrupt:
        print("\n❌ Animation generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
