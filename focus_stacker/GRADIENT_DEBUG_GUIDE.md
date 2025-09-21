# Gradient-Based Stacking Debug Visualization

## Overview
The debug window provides a step-by-step visual breakdown of how the gradient-based focus stacking algorithm works. This helps you understand which pixels are selected from each image and why.

## How to Access
1. Launch the Image Stacking Tool
2. Load your focus-bracketed images
3. Select **"Gradient-based"** from the stacking method dropdown
4. Click the **"Debug Process 🔍"** button that appears

## Debug Window Layout

### 840x920 Window with 2x2 Grid
The debug window shows four 400x400 pixel views for each input image:

#### Top Left: Original Image
- Shows the actual input image
- Downsampled for display (processing uses full resolution)
- This is what the algorithm starts with

#### Top Right: Gradient Map  
- Edge detection using Sobel operators
- Bright areas = high gradient (sharp edges/details)
- Dark areas = low gradient (smooth/blurry regions)
- **Key insight**: This shows where the algorithm detects sharpness

#### Bottom Left: Weight Map
- Normalized focus confidence for each pixel
- Brighter pixels = higher chance of being selected
- Derived from gradient map strength relative to other images
- **Key insight**: This shows the algorithm's "voting" system

#### Bottom Right: Smooth Weights
- Weight map after morphological and Gaussian smoothing
- Reduces noise and creates smoother transitions
- Only applied when smoothing radius > 0
- **Key insight**: This shows how smoothing reduces artifacts

## Navigation Controls

### Image Navigation
- **◀ Previous** / **Next ▶** buttons to cycle through input images
- **Image X/Y** counter shows current position
- Each image shows its processing steps side-by-side

### Information Panel
Shows technical details for the current image:
- Original image dimensions
- Gradient range (min/max edge strength)
- Mean gradient (overall sharpness)
- Weight statistics (focus confidence metrics)

## Understanding the Process

### Step 1: Gradient Calculation
```
For each image:
1. Convert to grayscale
2. Apply Sobel edge detection (X and Y directions)
3. Compute gradient magnitude: sqrt(grad_x² + grad_y²)
4. Apply threshold to remove noise
```

### Step 2: Weight Calculation
```
For each pixel position:
1. Compare gradient values across all images
2. Normalize so weights sum to 1.0
3. Higher gradient = higher weight for that image
```

### Step 3: Smoothing (if enabled)
```
For each weight map:
1. Apply morphological operations (closing/opening)
2. Apply Gaussian blur for soft transitions
3. Re-normalize weights to maintain sum = 1.0
```

### Step 4: Final Blending
```
For each pixel:
result = Σ(image[i] × smooth_weight[i]) for all images
```

## Practical Tips

### What to Look For

#### Good Focus Stacking Candidates:
- **High gradient contrast** between images
- **Clear weight separation** (bright regions don't overlap much)
- **Smooth weight transitions** after blending

#### Problem Indicators:
- **Low gradients everywhere** = all images equally blurry
- **Overlapping bright weights** = multiple images equally sharp
- **Noisy weight maps** = increase smoothing radius

### Optimizing Parameters

#### Smoothing Radius:
- **0**: No smoothing (may show noise/artifacts)
- **2-3**: Light smoothing (good for clean images)
- **4-6**: Medium smoothing (reduces most artifacts)
- **7+**: Heavy smoothing (may blur fine details)

#### When to Adjust:
- **Increase smoothing** if you see noise in final result
- **Decrease smoothing** if fine details are lost
- **Check debug** to see if weight maps look reasonable

## Technical Details

### Processing Resolution
- Debug window downsamples images for speed and display
- Actual stacking uses full-resolution images
- Results shown are representative of full processing

### Performance Notes
- Debug processing runs in background thread
- Small delay added for visual feedback
- Limited to first 4 images for manageable display

### File Outputs
The debug feature doesn't save files, but you can:
- Take screenshots of interesting comparisons
- Use the information to adjust parameters
- Compare before/after smoothing effects

## Example Workflow

1. **Load images** and select gradient-based method
2. **Click debug** to open visualization window
3. **Navigate through images** to see gradient patterns
4. **Check weight maps** for reasonable focus selection
5. **Adjust smoothing** if weight maps look noisy
6. **Close debug** and run actual stacking
7. **Compare result** with expectations from debug info

This debug feature transforms the "black box" of focus stacking into a transparent, educational tool that helps you understand and optimize the process!
