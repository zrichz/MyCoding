# Focus Confidence Calculation in Gradient-Based Algorithm

## 📊 Overview
The gradient-based focus stacking algorithm calculates "focus confidence" (weight maps) to determine how much each pixel from each image should contribute to the final result. Here's the complete step-by-step process:

## 🔬 Step-by-Step Process

### Step 1: Gradient Magnitude Calculation
For each input image:

```python
# Convert to grayscale if needed
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate gradients using Sobel operators
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges

# Compute gradient magnitude (edge strength)
gradient_mag = np.sqrt(grad_x² + grad_y²)

# Apply threshold to remove noise
gradient_mag = np.where(gradient_mag > 0.1, gradient_mag, 0)
```

**What this does:**
- **Sobel operators** detect edges in horizontal and vertical directions
- **Gradient magnitude** combines both directions: `√(grad_x² + grad_y²)`
- **Higher values** = sharper edges = better focus
- **Lower values** = smooth areas = poor focus
- **Threshold (0.1)** removes noise from very weak gradients

### Step 2: Focus Confidence (Weight) Calculation
```python
# Stack all gradient maps for comparison
gradient_stack = np.stack(gradient_maps, axis=-1)  # Shape: (H, W, N_images)

# Normalize to create weight maps
epsilon = 1e-6  # Prevent division by zero
weight_sum = np.sum(gradient_stack, axis=-1, keepdims=True) + epsilon
weights = gradient_stack / weight_sum
```

**Mathematical formula:**
```
For pixel (x,y) and image i:
weight[x,y,i] = gradient_mag[x,y,i] / (sum of all gradient_mag[x,y,j] for j=1..N)
```

**What this means:**
- **Weights sum to 1.0** at each pixel location
- **Higher gradient** = higher weight (more contribution)
- **Competitive selection**: Images "compete" based on sharpness
- **Relative confidence**: Not absolute, but relative to other images

## 📈 Weight Map Examples

### Example Pixel Analysis:
```
Pixel (100, 200) across 4 images:
Image 1: gradient_mag = 50  →  weight = 50/100 = 0.50 (50% contribution)
Image 2: gradient_mag = 30  →  weight = 30/100 = 0.30 (30% contribution) 
Image 3: gradient_mag = 15  →  weight = 15/100 = 0.15 (15% contribution)
Image 4: gradient_mag = 5   →  weight = 5/100  = 0.05 (5% contribution)
Total: 100                      Sum = 1.00 (100%)
```

### Interpretation:
- **Image 1** has the sharpest edge at this pixel (highest confidence)
- **Image 4** is very blurry at this pixel (lowest confidence)
- **All images contribute** but in proportion to their sharpness

## 🎯 Focus Confidence Characteristics

### High Confidence (Bright in weight map):
- **Sharp edges** detected
- **High gradient magnitude** 
- **Clear focus** in that region
- **Strong contribution** to final result

### Low Confidence (Dark in weight map):
- **Smooth areas** or blur
- **Low gradient magnitude**
- **Poor focus** in that region  
- **Weak contribution** to final result

### Zero Confidence (Black in weight map):
- **No detectable edges**
- **Gradient below threshold**
- **Very blurry** or uniform areas
- **No contribution** to final result

## 🔧 Smoothing Process (Optional)

### ⚡ When is Smoothing Applied?
The **smoothing radius** parameter in the GUI controls whether the algorithm uses:
- **Radius = 0**: Hard pixel selection (no smoothing) - faster but can create artifacts
- **Radius > 0**: Smooth weight blending (recommended 3-7) - slower but cleaner results

**Critical Decision Point**: After calculating the initial gradient maps and weight maps, the algorithm checks:
```python
if smooth_radius > 0:
    # Use smooth blending with morphological + Gaussian processing
    return _gradient_smooth_blend(images, gradient_maps, smooth_radius, blend_sigma)
else:
    # Use hard pixel selection (winner-takes-all)
    best_focus_indices = np.argmax(gradient_stack, axis=-1)
```

### Step 3: Morphological Smoothing (Applied when radius > 0)
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
weight = cv2.morphologyEx(weight, cv2.MORPH_CLOSE, kernel)  # Fill gaps
weight = cv2.morphologyEx(weight, cv2.MORPH_OPEN, kernel)   # Remove noise
```

### Step 4: Gaussian Smoothing  
```python
weight = cv2.GaussianBlur(weight, (0, 0), blend_sigma)
```

### Step 5: Re-normalization
```python
smooth_weight_stack = np.stack(smooth_weights, axis=-1)
weight_sum = np.sum(smooth_weight_stack, axis=-1, keepdims=True) + epsilon
normalized_weights = smooth_weight_stack / weight_sum
```

**Purpose of smoothing:**
- **Removes noise** from weight maps
- **Creates smoother transitions** between focus regions
- **Reduces artifacts** in final image
- **Maintains weight sum = 1.0** after processing

### 🎯 Hard vs Smooth Selection:

#### Hard Selection (radius = 0):
```python
# Winner-takes-all approach
best_focus_indices = np.argmax(gradient_stack, axis=-1)
result[y, x] = images[best_index][y, x]  # Binary choice
```
- **Fastest processing**
- **Sharp boundaries** between focus regions
- **Can create visible seams** or artifacts
- **No blending** between images

#### Smooth Selection (radius > 0):
```python
# Weighted blending approach  
final_pixel = Σ(image[i] × smooth_weight[i]) for all i
```
- **Slower processing** (morphological + Gaussian operations)
- **Smooth transitions** between focus regions
- **Eliminates seams** and artifacts
- **Natural blending** of overlapping focus areas

## 🎨 Visual Debug Window Interpretation

In the debug window's 2x2 grid:

### Top Right - "Gradient Map":
- **Brightness** = gradient magnitude (edge strength)
- **White areas** = sharp, high-contrast edges
- **Gray areas** = moderate edges  
- **Black areas** = smooth, no edges

### Bottom Left - "Weight Map":
- **Brightness** = focus confidence (0.0 to 1.0)
- **White areas** = high confidence (will dominate final result)
- **Gray areas** = moderate confidence  
- **Black areas** = low confidence (minimal contribution)

### Bottom Right - "Smooth Weights":
- **Same as weight map** but after morphological + Gaussian smoothing
- **Smoother transitions** between regions
- **Less noise** and artifacts
- **Still normalized** to sum to 1.0

## 🧮 Mathematical Summary

```
Focus Confidence Algorithm:
1. gradient_mag[i] = √(sobel_x² + sobel_y²) for each image i
2. weight[i] = gradient_mag[i] / Σ(gradient_mag[j]) for all j
3. smooth_weight[i] = smooth(weight[i]) if smoothing enabled
4. final_pixel = Σ(image[i] × smooth_weight[i]) for all i
```

The beauty of this approach is that it's **locally adaptive** - at each pixel, the algorithm automatically selects the sharpest source image(s) based on actual edge detection, creating a seamless blend of the best-focused regions from all input images.
