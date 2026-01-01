# Click-to-Set Face Hint - Quick Guide

## 🎯 New Feature: Click on Images to Set Face Hints!

No more typing coordinates! Just click directly on the nose area of your images.

## How It Works

### Step 1: Upload Your Images
Upload both face images as usual in the "First Face" and "Second Face" sections.

### Step 2: Try Automatic Detection First
Click **"Detect Faces"** to see if automatic detection works.
- ✅ If successful → Great! Proceed to create your video
- ❌ If it says "No face detected" → Continue to Step 3

### Step 3: Click on the Nose
**Simply click on the nose area** (or center) of the face in the uploaded image.
- Click directly on the image preview
- Aim for roughly the center of the face (nose area works best)
- You'll see coordinates appear in the "Face Hint" field below the image
- Example: "384, 200"

### Step 4: Detect Again
Click **"Detect Faces"** again.
- The preview will show:
  - Your landmarks (blue/red dots)
  - Green circle and crosshair at your hint location
  - "Hint" label
- This confirms the hint was used

### Step 5: Create Video
If detection succeeded, adjust settings and click **"Create Morph Video"**!

## Tips

### ✨ Best Practices
- **Click on the nose** - Most accurate hint location
- **Center of face** - Anywhere near the middle works
- **One click per image** - Last click overrides previous
- **No need to be precise** - Within 50 pixels of center is fine

### 🔄 Clear and Try Again
- Click **"Clear Face 1 Hint"** or **"Clear Face 2 Hint"** to remove hint
- Try automatic detection again without hint
- Or click a different spot

### 📏 Coordinates Explained
When you click, you'll see something like: `384, 200`
- First number (384) = X coordinate (horizontal position)
- Second number (200) = Y coordinate (vertical position)
- You can manually edit these if needed, but clicking is easier!

## Visual Guide

```
┌─────────────────────────────────┐
│  Upload Image                    │
│  ┌──────────────────────────┐  │
│  │                          │  │
│  │         👤 Face          │  │
│  │        👃 ← Click here!  │  │
│  │                          │  │
│  └──────────────────────────┘  │
│                                 │
│  Face Hint: 384, 200            │
│  [Clear Face 1 Hint]            │
└─────────────────────────────────┘
```

## Troubleshooting

### Click doesn't register
- Make sure image is fully loaded
- Click directly on the image preview area
- You should see coordinates update in the hint field below

### Wrong face in multi-face images
- Click closer to your desired face
- Click far from other faces
- The hint creates a search region around your click

### Still no face detected with hint
- Try clicking slightly different location (up/down 20-30 pixels)
- Ensure face is at least 50x50 pixels
- Face should be reasonably frontal (not extreme profile)

### Want to start over
- Click "Clear Face X Hint" button
- Upload a different image
- Or click a new location (overwrites previous)

## Example Workflow

1. **Upload** → Two face images
2. **Click "Detect Faces"** → Says "No face detected in first image"
3. **Click on nose** of first image → Hint field shows "400, 300"
4. **Click "Detect Faces"** again → ✓ Face detected! Green hint marker visible
5. **Adjust settings** → Duration, resolution
6. **Click "Create Morph Video"** → Video created!

## Advantages Over Manual Coordinates

❌ **Old way**: Open Paint → hover → read coordinates → type numbers → check → retype

✅ **New way**: Click on nose → Done!

Much faster and more intuitive! 🎉

## Technical Note

When you click at position (X, Y), the app:
1. Creates a 60% search region centered on your click
2. Upscales the region to 1600px for better detection  
3. Runs face detection with pyramid upsampling
4. Transforms coordinates back to original image
5. Shows green marker at your click location in preview

The hint guides the algorithm to focus on the area you specified, making small faces effectively larger within the detection window.

---

**Happy Morphing!** 🎭✨
