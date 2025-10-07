# Filmic Effects Processor

A GUI application for applying cinematic film grain and vignette effects to 720x1600 pixel images.

## Features

### Film Grain Effect
- **Subtle noise texture** that mimics analog film grain
- **Edge-enhanced grain** - more pronounced towards image edges for natural falloff
- **Adjustable intensity** (0.0 to 0.5)
- **Edge boost multiplier** (1.0x to 3.0x) for controlling edge enhancement

### Vignette Effect
- **Circular vignette** (not elliptical) for proper cinematic look
- **Corner darkening** up to 25% (default 10%)
- **Smooth radial falloff** from center to edges
- **Maintains aspect ratio** regardless of image dimensions

### User Interface
- **Directory selection** - Choose folder containing 720x1600 images
- **Live preview** - See effects applied in real-time
- **Image navigation** - Browse through images with Previous/Next
- **Batch processing** - Apply effects to entire directory
- **Progress tracking** - Visual progress bar and status updates

## Usage

### 1. Launch Application
```bash
run_filmic_effects.bat
```

### 2. Select Directory
- Click "📁 Select Directory" 
- Choose folder containing your 720x1600 images
- Application will scan and count valid images

### 3. Preview and Adjust
- Click "🔍 Load Preview" to see first image
- Use "◀ Previous" and "Next ▶" to browse images
- Adjust sliders to fine-tune effects:
  - **Film Grain Intensity**: Overall grain strength
  - **Edge Grain Boost**: How much more grain at edges
  - **Vignette Strength**: Corner darkening amount

### 4. Process All Images
- Click "🎬 Process All Images" when satisfied with preview
- Confirm processing dialog
- Wait for completion (progress shown)
- Output images saved with "_filmic" suffix

## Technical Details

### Image Requirements
- **Dimensions**: Exactly 720x1600 pixels
- **Formats**: JPG, JPEG, PNG, BMP, TIFF, TIF
- **Color**: RGB images (other modes converted automatically)

### Effect Implementation
- **Film Grain**: Gaussian noise with distance-based intensity mapping
- **Edge Enhancement**: Quadratic falloff from center increases grain
- **Vignette**: Circular mask with smooth quadratic darkening
- **Quality Preservation**: JPEG saved at 95% quality, lossless for other formats

### Output
- Original files are **never modified**
- New files saved with "_filmic" suffix
- Same format and directory as source images
- Example: `photo.jpg` → `photo_filmic.jpg`

## Requirements

- Python 3.7+
- PIL/Pillow
- NumPy
- tkinter (usually included with Python)

## Installation

The application uses the existing `.venv` environment in the MyCoding directory. All required packages should already be installed.

If you encounter missing package errors, activate the virtual environment and install:
```bash
pip install pillow numpy
```

## Performance

- **Preview**: Real-time updates (0.1-0.5 seconds per adjustment)
- **Processing**: ~0.5-2 seconds per 720x1600 image
- **Memory**: Efficient streaming processing, minimal RAM usage
- **Threading**: Background processing doesn't freeze UI

## Tips

1. **Start subtle** - Film grain should enhance, not overpower the image
2. **Edge boost 1.8-2.2** works well for most images  
3. **Vignette 0.08-0.12** provides natural corner darkening
4. **Preview different images** to ensure consistent results
5. **Batch process** only after confirming settings on previews