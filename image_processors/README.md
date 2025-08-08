# Image Processors Collection

A comprehensive collection of image processing tools organized by functionality.

## Directory Structure

### 🔄 image_rotator/
Tools for rotating images in various ways
- `rotate_all_jpegs_in_dir_90CW.py` - Batch rotate JPEG files 90° clockwise

### 📏 image_expander/
Tools for expanding images with various effects
- `image_expander_vertical_ver1.py` - Auto-expand images to 720x1600 with blur/luminance effects
- `image_expander_cylindrical.py` - Cylindrical image expansion

### ✂️ image_cropper/
Tools for cropping and selecting image regions
- `interactive_image_cropper.py` - Interactive GUI cropping tool
- `click2crop_images.py` - Click-to-crop functionality
- `image_gridder_chopper_saver.py` - Grid-based image chopping
- `verify_cropper.py` - Cropping verification tool

### 🎯 seam_carving/
Advanced seam carving image resizing
- `demo_seam_carving.py` - Seam carving demonstration
- `seam_carving_width_reducer.py` - Width reduction via seam carving

### 🔄 batch_processors/
Tools for batch processing multiple images
- `create_test_images.py` - Generate test images
- `image_remapper.py` - Batch image remapping

### 🎨 filters_effects/
Filters and visual effects
- `Gausian_kernel_sigma_visualisation.py` - Gaussian kernel visualization
- `motion_extraction_4sec.py` - Motion extraction effects
- `svg_editor_ver1.py` - SVG editing tools

### 🛠️ utilities/
Utility tools and documentation
- `tempCodeRunnerFile.py` - Temporary code runner
- `image resizing rules.txt` - Image resizing guidelines
- `THUMBNAIL_FIX_SUMMARY.md` - Thumbnail fixes documentation

### 🧪 tests/
Test files for all image processing tools
- Various test files for different functionalities

## Setup

All tools share the same virtual environment located in `.venv/`

### Activate Environment
```cmd
.venv\Scripts\activate
```

### Install Dependencies
```cmd
pip install -r requirements.txt
```

## Usage

Run any tool from the main directory:
```cmd
python image_expander\image_expander_vertical_ver1.py
python image_rotator\rotate_all_jpegs_in_dir_90CW.py
python image_cropper\interactive_image_cropper.py
```

Or navigate to specific directories and run from there:
```cmd
cd image_expander
python image_expander_vertical_ver1.py
```

## Requirements

- Python 3.10+
- PIL/Pillow
- NumPy
- SciPy
- Tkinter (usually included with Python)
- OpenCV (for some tools)

## Contributing

When adding new tools, place them in the appropriate subdirectory based on their functionality.
