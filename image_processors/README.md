# Image Processors Collection

A comprehensive collection of image processing tools organized by functionality.

## Directory Structure

### 🎮 Interactive Viewers & Animation Tools
Creative tools for viewing and interacting with images

- `falling_blocks_viewer.py` - **Interactive Image Viewer with Falling Blocks Animation**
  - 🎬 **Falling Blocks Effect**: Images are divided into 32x32 pixel blocks that fall into place
  - 🎨 **Original Colors**: Blocks display in their natural colors with gaussian blur effect
  - 🌫️ **Blur Effect**: Blocks start blurred and fade to sharp once settled
  - 🎮 **Interactive Controls**: SPACE to start/pause, arrow keys to navigate
  - 📁 **Directory Support**: Select a folder and view all images sequentially
  - ⚡ **Simple Physics**: Realistic falling animation with gravity-based settling
  - **Run with:** `run_falling_blocks.bat`/`.sh` or `python falling_blocks_viewer.py`

### 🔄 Format Converters & Batch Processing
Tools for converting between formats and batch processing workflows

- `png_to_jpg_processor.py` - **PNG to JPG Batch Converter with Rotation**
  - 📁 **Batch Processing**: Converts all PNG files in directory
  - 📐 **Smart Scaling**: Scales to max 1080x1440 preserving aspect ratio  
  - 🌊 **Blur Fill**: Uses intelligent blur + luminance reduction for aspect ratio gaps
  - 🔄 **Auto Rotation**: Rotates 90° counter-clockwise after scaling
  - 📱 **Final Stretch**: Stretches to final 1920x1080 dimensions
  - 💾 **High Quality**: Saves as high-quality JPEG (quality=88, no subsampling)
  - 📝 **Numbered Output**: Saves as IMG00001.jpg, IMG00002.jpg format
  - **Run with:** `run_png_processor.bat` or `python png_to_jpg_processor.py`

### 🔄 image_rotator/
Tools for rotating images in various ways
- `rotate_all_jpegs_in_dir_90CW.py` - Batch rotate JPEG files 90° clockwise

### 📏 image_expander/
Tools for expanding images with various effects
- `image_expander_720x1600.py` - Auto-expand images to 720x1600 with blur/luminance effects
  - Uses timestamp-based filename schema: `YYYYMMDDHHMMSS_nnn_720x1600.ext`
  - Automatic conflict resolution with incremented counter (001, 002, etc.)
- `image_expander_cylindrical.py` - Cylindrical image expansion

### ✂️ image_cropper/
Interactive tools for cropping and selecting image regions with batch processing capabilities

**Main Tool:**
- `interactive_image_cropper.py` - **Interactive GUI Batch Cropper**
  - 🖱️ **Interactive Cropping**: Click and drag to select crop areas visually
  - 📁 **Batch Processing**: Process entire directories of images sequentially  
  - 🔍 **Smart Thumbnails**: Display images at max 800x800 while preserving aspect ratio
  - 💾 **Full Resolution**: Crops are applied to original high-resolution images
  - 📐 **Automatic Resizing**: Applies intelligent resizing rules before saving
  - 📝 **Smart Filenames**: Adds final dimensions to filenames (e.g., `photo_720x1280.jpg`)
  - 📂 **Auto Organization**: Saves cropped images to "cropped" subdirectory
  - ⏭️ **Easy Navigation**: Previous/Next/Skip buttons for workflow control
  - 🔄 **Conflict Handling**: Automatic filename conflict resolution

**Additional Tools:**
- `click2crop_images.py` - Click-to-crop functionality
- `image_gridder_chopper_saver.py` - Grid-based image chopping
- `verify_cropper.py` - Cropping verification tool

**Image Resizing Rules Applied Automatically:**
1. **Width Constraint**: If width > 720px → reduce to 720px maintaining aspect ratio
2. **Height Constraint**: If height > 1600px → reduce to 1600px maintaining aspect ratio  
3. **Minimum Width Rule**: If width < 720px and height < (20/9) × width → increase width to 720px
4. **Aspect Ratio Rule**: If height < 1600px and width < (9/20) × height → increase width to 720px

**Supported Formats:** JPG, JPEG, PNG, BMP, TIFF, TIF, GIF

### 🔧 SuperJPEG Encoder/Decoder
Custom JPEG implementation from first principles with configurable block sizes

**Main Tool:**
- `superjpeg_encoder.py` - **SuperJPEG Encoder/Decoder GUI**
  - 🔧 **Custom JPEG Implementation**: Built from first principles using pure mathematical transforms
  - 📐 **Variable Block Sizes**: Configurable from 4x4 to 64x64 in steps of 4 (4, 8, 12, 16, ..., 64)
  - 🎛️ **Quality Control**: Adjustable quality settings (1-100) with adaptive quantization
  - 📊 **Multiple Variants**: Automatically generates files for ALL block sizes from single input
  - 🔄 **Full Pipeline**: Complete encode/decode cycle with verification testing
  - 📁 **Smart Naming**: Descriptive filenames with block size and quality info
  - 🎨 **Color Support**: Full RGB and grayscale image processing
  - 📈 **Progress Tracking**: Real-time encoding progress and detailed logging

**Technical Features:**
- **DCT Implementation**: Custom Discrete Cosine Transform matrices for any block size
- **Adaptive Quantization**: Automatically scaled quantization tables based on block size
- **Zigzag Scanning**: Proper frequency ordering for optimal compression
- **Run-Length Encoding**: Efficient data compression for sparse coefficients  
- **YUV Color Space**: Professional color space conversion for optimal compression
- **Lossless Decode**: Perfect reconstruction verification for encoding validation

**Output Format:**
- Creates SuperJPEG files in JSON format containing all compression data
- Filename pattern: `{original}_superjpeg_{blocksize}x{blocksize}_q{quality}.json`
- Example: `photo_superjpeg_8x8_q50.json`, `photo_superjpeg_16x16_q75.json`

**Use Cases:**
- **Research**: Compare compression efficiency across different block sizes
- **Education**: Learn JPEG compression principles with visual feedback
- **Optimization**: Find optimal block size for specific image types
- **Analysis**: Study frequency domain characteristics of images

### 🎯 seam_carving/
Advanced seam carving image resizing using content-aware algorithms

**Main Tool:**
- `seam_carving_width_reducer.py` - **GUI Width Reduction Tool**
  - 🎯 **Selective Seam Carving**: Applies algorithm only to outer 50% of image (first and last 25%)
  - 🔧 **Configurable Reduction**: User can specify reduction percentage (50-99%)
  - 🖥️ **GUI Interface**: Easy-to-use tkinter-based interface
  - 💾 **Auto-save**: Processed images automatically saved with "_reduced_width" suffix
  - 📊 **Progress Tracking**: Real-time progress bar and status updates
  - 🎨 **Content-Aware**: Preserves important middle content while reducing sides

**Additional Tools:**
- `demo_seam_carving.py` - Seam carving demonstration

**How Seam Carving Works:**
1. **Energy Calculation**: Uses gradient magnitude to identify important image features
2. **Seam Detection**: Finds minimum-energy vertical paths through the image
3. **Seam Removal**: Removes low-energy seams to reduce width
4. **Selective Application**: Only processes first 25% and last 25% of image width

**Example:** For a 1000px wide image with 70% reduction:
- Target width: 700 pixels (300 pixels removed)
- First 250 pixels: ~150 seams removed
- Middle 500 pixels: unchanged
- Last 250 pixels: ~150 seams removed

**Supported Formats:** JPG, JPEG, PNG, BMP, TIFF
**Dependencies:** OpenCV, NumPy, Pillow

### 🎥 video_optical_flow_openCV/
Video analysis and optical flow visualization
- `optical_flow_visualizer.py` - Dense optical flow visualization with GUI controls
  - Color-coded motion direction visualization
  - Adjustable flow density, line thickness, and motion threshold
  - Supports multiple video formats (MP4, AVI, MOV, etc.)
  - Real-time processing progress with status updates

### 🌊 GrayScott_filter/
Gray-Scott diffusion filtering for artistic image processing
- `GrayScott_filter.py` - GUI application for Gray-Scott diffusion processing
  - Iterative sharpening and blurring operations
  - Automatic grayscale conversion and image resizing
  - Customizable iteration count for different effects
  - Supports common image formats (PNG, JPEG, etc.)

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
- `phrase_generator.py` - **Human Situation Phrase Generator**
  - 🎲 **Random Phrase Generation**: Creates diverse human situation descriptions
  - 🧩 **Modular Building Blocks**: Uses verbs, adjectives, nouns, and connecting words
  - 🎛️ **GUI Interface**: Easy-to-use tkinter interface with customizable options
  - 💾 **Save Functionality**: Export generated phrases to text files
  - 📊 **Batch Generation**: Create 1-50 phrases at once
  - 🔤 **Smart Grammar**: Intelligent article selection (a/an) and capitalization
- `tempCodeRunnerFile.py` - Temporary code runner
- `image resizing rules.txt` - Image resizing guidelines
- `THUMBNAIL_FIX_SUMMARY.md` - Thumbnail fixes documentation

### 🧪 tests/
Test files for all image processing tools
- Various test files for different functionalities

## Detailed Tool Guides

### Interactive Image Cropper - Complete Workflow

The Interactive Image Cropper is a powerful batch processing tool that allows you to visually select crop areas on thumbnails and apply them to full-resolution images.

**Step-by-Step Usage:**
1. **Launch**: Run `run_image_cropper.bat` (Windows) or `./run_image_cropper.sh` (Linux/Mac)
2. **Select Directory**: Click "Browse" to choose a folder containing images
3. **Crop Images**:
   - Click and drag on the thumbnail to select crop area
   - Click "Crop & Save" to apply to full-resolution image
   - Use "Next" to move to the next image
   - Use "Skip" to skip an image without cropping
4. **Find Results**: Cropped images are saved in the "cropped" subdirectory

**Interface Controls:**
| Button | Function |
|--------|----------|
| **Browse** | Select directory containing images |
| **Previous** | Go to previous image |
| **Crop & Save** | Apply crop to full-resolution image and save |
| **Skip** | Move to next image without cropping |
| **Next** | Move to next image |
| **Clear Selection** | Remove current crop selection |

**File Organization:**
```
Your Image Directory/
├── image1.jpg
├── image2.png
├── image3.jpg
└── cropped/           # Auto-created
    ├── image1_720x1280.jpg     # Cropped versions with dimensions
    ├── image2_720x1280.png
    └── image3_720x1280_crop_1.jpg  # Conflict resolution
```

**Technical Features:**
- **Coordinate Mapping**: Thumbnail coordinates automatically scaled to original image dimensions
- **Memory Management**: Only current image loaded in memory for efficiency
- **Error Handling**: Graceful handling of unsupported or corrupted files

### Seam Carving Width Reducer - Advanced Usage

The Seam Carving tool uses content-aware algorithms to intelligently reduce image width while preserving important central content.

**Detailed Process:**
1. **Launch**: Run `run_seam_carving.bat` (Windows) or `./run_seam_carving.sh` (Linux/Mac)
2. **Select Image**: Click "Browse" to select an image file
3. **Set Reduction**: Enter percentage between 50-99% for width reduction
4. **Process**: Click "Process Image" to apply seam carving
5. **Result**: Processed image displayed and auto-saved with "_reduced_width" suffix

**Algorithm Details:**
- **Energy Function**: Calculates gradient magnitude to identify image importance
- **Dynamic Programming**: Finds optimal seam paths with minimum energy
- **Selective Processing**: Only affects outer edges (first/last 25% of width)
- **Content Preservation**: Middle 50% of image remains untouched

**Best Practices:**
- Works best with images having less important content on the sides
- Ideal for landscape photos, banners, or wide-format images
- Processing time scales with image size and reduction amount
- Preserves important vertical structures in the center

### Human Situation Phrase Generator - Creative Text Tool

The Phrase Generator creates diverse, realistic human situation descriptions using modular building blocks for creative writing, brainstorming, or entertainment.

**Detailed Process:**
1. **Launch**: Run `run_phrase_generator.bat` (Windows) or `./run_phrase_generator.sh` (Linux/Mac)
2. **Set Count**: Choose number of phrases to generate (1-50)
3. **Generate**: Click "Generate Phrases" for multiple or "Generate One" for single phrases
4. **Review**: Browse generated phrases in the scrollable text area
5. **Save**: Use "Save to File" to export phrases with timestamp

**Building Block Categories:**
- **Action Verbs**: cooking, building, painting, dancing, exploring, etc.
- **Mood Adjectives**: happy, excited, calm, adventurous, focused, etc.
- **Descriptive Adjectives**: beautiful, complex, colorful, ancient, etc.
- **Object Nouns**: tractor, bicycle, guitar, computer, mountain, etc.
- **Food Nouns**: meal, soup, coffee, pizza, chocolate, etc.
- **Activity Nouns**: lesson, performance, game, adventure, etc.
- **Place Nouns**: home, park, restaurant, mountains, etc.
- **Manner Adverbs**: provocatively, skillfully, enthusiastically, gracefully, etc.
- **Weather Conditions**: sunny, rainy, cloudy, windy, stormy, etc.

**Three Streamlined Template Patterns:**
1. `{adverb_manner} {verb_action}, {noun_food}, {adjective_descriptive} {noun_object}`
2. `{adjective_mood}, {verb_action}, {noun_object}, {noun_place}`
3. `{verb_action}, {noun_activity}, {weather_condition} weather`

**Example Generated Phrases:**
- `"Skillfully exercising, fruit, golden guitar","Energetic, discovering, boat, countryside","Sitting, exhibition, snowy weather","Passionately painting, chocolate, ancient sculpture"`

**Output Format:**
- **Comma-separated**: All phrases in one continuous line
- **Quoted phrases**: Each phrase wrapped in double quotes
- **No numbering**: Clean format without line numbers or capitalization
- **Easy parsing**: Format suitable for data processing or copying

**Technical Features:**
- **Simplified Structure**: No prepositions or articles for cleaner, more direct phrases
- **Comma-Separated Format**: Clean, list-like phrase structure for easy parsing
- **Quoted Output**: Each phrase wrapped in quotes, all on one line
- **Streamlined Templates**: Three focused sentence patterns for consistent output
- **Modular Design**: Organized word categories for reliable phrase generation
- **Uniqueness Checking**: Prevents duplicate phrases in batch generation
- **Easy Copying**: Single-line format perfect for data processing or transfer
- **Duplicate Prevention**: Ensures unique phrases in batch generation
- **File Export**: Saves with timestamps and formatting

## Quick Start

### Easy Launch Scripts
For the most commonly used tools, use the provided launcher scripts:

**Falling Blocks Image Viewer:**
```cmd
run_falling_blocks.bat    # Windows
./run_falling_blocks.sh   # Linux/Mac
```

**Interactive Image Cropper:**
```cmd
run_image_cropper.bat     # Windows
./run_image_cropper.sh    # Linux/Mac
```

**Image Expander 720x1600:**
```cmd
run_image_expander.bat    # Windows
./run_image_expander.sh   # Linux/Mac
```

**Gray-Scott Filter:**
```cmd
run_grayscott_filter.bat  # Windows
./run_grayscott_filter.sh # Linux/Mac
```

**Video Optical Flow:**
```cmd
run_optical_flow.bat      # Windows
./run_optical_flow.sh     # Linux/Mac
```

**Seam Carving Width Reducer:**
```cmd
run_seam_carving.bat      # Windows
./run_seam_carving.sh     # Linux/Mac
```

**Phrase Generator:**
```cmd
run_phrase_generator.bat  # Windows
./run_phrase_generator.sh # Linux/Mac
```

These scripts will automatically:
- Activate the virtual environment
- Check for required dependencies
- Install missing packages if needed
- Launch the selected application

### Manual Setup

All tools share the same virtual environment located in `.venv/`

#### Activate Environment
**Windows:**
```cmd
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

#### Install Dependencies
```cmd
pip install -r requirements.txt
```

## Usage

Run any tool from the main directory:
```cmd
python image_expander\image_expander_720x1600.py
python image_rotator\rotate_all_jpegs_in_dir_90CW.py
python image_cropper\interactive_image_cropper.py
```

Or navigate to specific directories and run from there:
```cmd
cd image_expander
python image_expander_720x1600.py
```

### Quick Launch Scripts
**Windows (.bat files):**
- `run_image_cropper.bat` - Launch Interactive Image Cropper
- `run_image_expander.bat` - Launch Image Expander (720x1600)
- `run_seam_carving.bat` - Launch Seam Carving Width Reducer
- `run_grayscott_filter.bat` - Launch GrayScott Reaction-Diffusion Filter
- `run_optical_flow.bat` - Launch Optical Flow Visualizer
- `run_phrase_generator.bat` - Launch Phrase Generator
- `run_superjpeg_encoder.bat` - Launch SuperJPEG Encoder/Decoder

**Linux/Mac (.sh files):**
- `run_image_cropper.sh` - Launch Interactive Image Cropper
- `run_image_expander.sh` - Launch Image Expander (720x1600)
- `run_seam_carving.sh` - Launch Seam Carving Width Reducer
- `run_grayscott_filter.sh` - Launch GrayScott Reaction-Diffusion Filter
- `run_optical_flow.sh` - Launch Optical Flow Visualizer
- `run_phrase_generator.sh` - Launch Phrase Generator
- `run_superjpeg_encoder.sh` - Launch SuperJPEG Encoder/Decoder

## Requirements

### System Requirements
- Python 3.10+
- Virtual environment support

### Python Packages
All required packages are listed in `requirements.txt`:

- **Pillow** (11.3.0) - Python Imaging Library for all image processing operations
- **numpy** (2.2.6) - Array operations and mathematical functions
- **scipy** (1.15.3) - Scientific computing (used for image filtering and blur effects)
- **opencv-python** (4.12.0.88) - Computer vision library (seam carving, optical flow)
- **scikit-image** (0.25.2) - Scientific image processing (CLAHE, color space conversions)
- **pygame** (2.6.0) - Game development framework (interactive viewers and animations)

### Standard Library Modules
These are included with Python and don't need separate installation:
- tkinter (GUI framework)
- os, sys, datetime, pathlib, threading, tempfile
- math, re, subprocess, argparse

### Installation
Install all dependencies using:
```cmd
pip install -r requirements.txt
```

## Troubleshooting & Tips

### Common Issues

**"No images found" (Image Cropper)**
- Check directory contains supported image formats
- Verify file extensions are correct (.jpg, .png, etc.)
- Ensure files are not corrupted

**"Failed to load image"**
- Image file may be corrupted
- Use Skip button to continue with next image
- Try opening the image in another program to verify

**Crop selection not working**
- Ensure you click and drag within the image area
- Make selection large enough (minimum 10x10 pixels)
- Clear selection and try again

**Application won't start**
- Verify Python and required packages are installed
- Check virtual environment is activated
- Run `pip install -r requirements.txt` to ensure dependencies

**Seam carving takes too long**
- Try smaller images or lower reduction percentages
- Processing time scales with image dimensions
- Close other applications to free up memory

### Best Practices

**Image Cropper:**
1. **Image Quality**: Start with high-resolution images for best crop results
2. **Crop Selection**: Make precise selections on thumbnails - they map accurately to originals
3. **Aspect Ratios**: Consider the final use when selecting crop dimensions
4. **Batch Workflow**: Use Skip liberally for unwanted images
5. **File Naming**: Original filenames are preserved in cropped versions

**Seam Carving:**
1. **Image Selection**: Works best with images having less important content on sides
2. **Reduction Amount**: Start with 60-70% reduction for best balance
3. **Content Type**: Ideal for landscapes, banners, or wide-format images
4. **Preview**: Check results before applying to multiple images

**General Tips:**
1. **Backup**: Always keep backups of original images
2. **Testing**: Use `create_test_images.py` to generate test images for experimentation
3. **Organization**: Tools automatically create output subdirectories
4. **Formats**: Stick to common formats (JPG, PNG) for best compatibility

### Performance Optimization

- **Memory**: Close unused applications when processing large images
- **Storage**: Ensure sufficient disk space for output images
- **Batch Size**: Process images in smaller batches for very large collections
- **Virtual Environment**: Use the shared `.venv` for consistent performance

## Testing

Generate test images for experimentation:
```bash
python batch_processors/create_test_images.py
```
This creates sample images in `test_images/` directory with various dimensions and patterns.

## Contributing

When adding new tools, place them in the appropriate subdirectory based on their functionality.
