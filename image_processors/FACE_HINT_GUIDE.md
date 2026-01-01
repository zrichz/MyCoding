# Face Hint Feature - User Guide

## Overview
The Face Morph Video Creator now includes an optional **Face Hint** feature to help detect faces that are small or difficult to detect automatically.

## When to Use Face Hints
Use face hints when:
- Automatic detection fails with "No face detected" error
- Faces are small (less than 10% of image size)
- Face is partially obscured or at an unusual angle
- Multiple faces in image and you want to select a specific one

## How to Use

### Step 1: Upload Your Images
Upload both images as usual in the app.

### Step 2: Get Face Center Coordinates
If automatic detection fails, you'll need to find the approximate center coordinates of the face:

**Method 1: Using Windows Paint/Photos**
1. Right-click the image → Open with → Paint
2. Hover your mouse over the approximate center of the face
3. Look at the bottom-left corner of Paint to see coordinates (X, Y)
4. Write down these numbers

**Method 2: Using Image Viewer with Coordinates**
- Use any image viewer that shows pixel coordinates when hovering
- Popular options: IrfanView, GIMP, Photoshop

**Method 3: Rough Estimate**
- If image is 1000x1000 and face is centered: try (500, 500)
- If face is in upper-left quadrant: try (250, 250)
- You don't need perfect precision - roughly center of the face works

### Step 3: Enter Coordinates
In the app interface:
- Below "First Face" image: Enter X and Y values in the "Face Hint" fields
- Below "Second Face" image: Enter X and Y values if needed
- Leave blank if you want automatic detection for that image

### Step 4: Detect Faces
Click "Detect Faces" to preview. The hint location will be shown as a green circle and crosshair on the preview if used.

### Step 5: Create Video
If detection succeeds, click "Create Morph Video" as usual.

## Technical Details

### How It Works
When you provide face hint coordinates:
1. App creates a search region (60% of image size) centered on your hint
2. Image is upscaled to 1600px for better detection
3. Face detection runs only in the search region
4. This makes small faces appear larger relative to the search area
5. Coordinates are automatically transformed back to original image space

### ROI Size
- Default: 60% of the smaller image dimension
- Example: 768x1152 image → 460x460 pixel search region
- This balances between focused search and enough context

### Coordinate Format
- Origin (0, 0) is top-left corner of image
- X increases going right
- Y increases going down
- Example: (400, 300) means 400 pixels from left, 300 pixels from top

## Examples

### Example 1: Small Face in Portrait
- Image size: 768x1152
- Face located at approximately: (384, 200)
- Enter: Face Hint X = 384, Face Hint Y = 200

### Example 2: Face in Upper-Left
- Image size: 1024x1024  
- Face in upper-left quadrant
- Enter: Face Hint X = 300, Face Hint Y = 300

### Example 3: Off-Center Face
- Image size: 1920x1080
- Face on right side at: (1400, 540)
- Enter: Face Hint X = 1400, Face Hint Y = 540

## Troubleshooting

### "No face detected" even with hint
- Try moving the hint coordinates slightly
- Ensure coordinates are within image bounds
- Face should be at least 50x50 pixels
- Ensure face is reasonably frontal (not extreme profile)

### Wrong face detected in multi-face image
- Adjust hint to be closer to desired face center
- Move hint further from other faces

### Coordinates don't match what I see
- Make sure you're looking at the ORIGINAL image, not a scaled preview
- Use an image viewer that shows true pixel coordinates
- Remember: (0,0) is top-left, not bottom-left

## Tips
- **Precision not required**: Being within 50 pixels of face center usually works
- **Leave blank for automatic**: Only use hints when automatic detection fails
- **Test with preview**: Use "Detect Faces" button to verify before creating video
- **Green markers**: Successfully used hints are shown with green circles in preview
- **Both or one**: You can use hints for just one image or both

## Algorithm Details (Advanced)

The face hint creates a Region of Interest (ROI):
```
roi_size = min(image_width, image_height) * 0.6
x1 = hint_x - roi_size / 2
y1 = hint_y - roi_size / 2
x2 = x1 + roi_size
y2 = y1 + roi_size
```

Detection runs with pyramid upsampling (levels 0, 1, 2) on the ROI, then coordinates are transformed:
```
original_x = (detected_x + roi_x1) / scale_factor
original_y = (detected_y + roi_y1) / scale_factor
```

This approach:
- Reduces search space (faster processing)
- Increases relative face size in detection window
- Maintains accuracy through proper coordinate transformation
- Falls back to full-image search if hint not provided
