# Gradient-Based Stacking Smoothing Enhancement

## Problem Solved
The gradient-based stacking method was producing noisy results due to hard pixel-wise selection between focus regions. Sharp transitions between selected pixels from different images created artifacts and noise.

## Solution Implemented
Added smooth blending capability to the gradient-based stacking algorithm with two new parameters:

### New Parameters:
1. **smooth_radius** (int, 0-10): Morphological smoothing radius
   - 0 = Hard selection (original noisy method)
   - 3-5 = Recommended for good noise reduction
   - Higher values = More smoothing but less detail

2. **blend_sigma** (float, default 1.0): Gaussian smoothing sigma for weight transitions

### Algorithm Enhancement:
- Created `_gradient_smooth_blend()` helper method
- Uses morphological operations (opening/closing) to smooth weight regions
- Applies Gaussian blur for soft weight transitions
- Re-normalizes weights to maintain proper blending

### GUI Enhancement:
- Added "Smoothing Radius" slider for Gradient-based method
- Shows/hides automatically when method is selected
- Real-time value display with descriptive text
- Default value of 3 for good balance

### Testing Results:
Created demonstration files showing the improvement:
- `demo_gradient_noisy.png` - Original noisy method
- `demo_gradient_light.png` - Light smoothing (radius=2)
- `demo_gradient_medium.png` - Medium smoothing (radius=4)

### Usage:
1. Select "Gradient-based" stacking method in GUI
2. Adjust "Smoothing Radius" slider (3-5 recommended)
3. Higher values reduce noise but may soften fine details
4. 0 = Original behavior for comparison

### Technical Details:
- Maintains backward compatibility (smooth_radius=0 = original method)
- Uses morphological operations for region-based smoothing
- Gaussian weighting for natural transitions
- Progress callbacks for user feedback
- Full resolution processing (no quality loss)

This enhancement significantly reduces the noise artifacts in gradient-based stacking while preserving the method's ability to select the sharpest regions from input images.
