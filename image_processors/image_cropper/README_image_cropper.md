# Interac- **💾 Full Resolution**: Crops are applied to original high-resolution images
- **📐 Automatic Resizing**: Applies intelligent resizing rules before saving
- **📝 Smart Filenames**: Adds final dimensions to filenames (e.g., `photo_720x1280.jpg`)
- **📂 Auto Organization**: Saves cropped images to "cropped" subdirectorye Image Cropper

A Python GUI application for batch cropping images with interactive selection and automatic resizing. Users can visually select crop areas on thumbnail previews, and the crops are applied to full-resolution original images with intelligent resizing rules.

## Features

- **🖱️ Interactive Cropping**: Click and drag to select crop areas visually with enforced 9:20 minimum aspect ratio
- **📁 Batch Processing**: Process entire directories of images sequentially  
- **🔍 Smart Thumbnails**: Display images at max 800x800 while preserving aspect ratio
- **💾 Full Resolution**: Crops are applied to original high-resolution images
- **� Automatic Resizing**: Applies intelligent resizing rules before saving
- **�📂 Auto Organization**: Saves cropped images to "cropped" subdirectory
- **⏭️ Easy Navigation**: Previous/Next/Skip buttons for workflow control
- **🔄 Conflict Handling**: Automatic filename conflict resolution
- **📊 Progress Tracking**: Shows current image and total count

## Crop Selection Constraints

During the interactive crop selection process, the application enforces a **minimum 9:20 aspect ratio** (width:height). This means:

- The crop rectangle width must be at least (9/20) × height
- As you drag to select a crop area, the application automatically adjusts the dimensions to maintain this minimum aspect ratio
- The constraint is applied both during dragging AND when the mouse button is released, ensuring consistent results
- This constraint ensures that cropped images will work well with the subsequent resizing rules

## Image Resizing Rules

Before saving each cropped image, the following resizing rules are automatically applied to ensure optimal dimensions:

1. **Width Constraint**: If width > 720px → reduce to 720px maintaining aspect ratio
2. **Height Constraint**: If height > 1600px → reduce to 1600px maintaining aspect ratio  
3. **Minimum Width Rule**: If width < 720px and height < (20/9) × width → increase width to 720px
4. **Aspect Ratio Rule**: If height < 1600px and width < (9/20) × height → increase width to 720px

These rules ensure images are optimally sized while maintaining quality and aspect ratios. The final image will always respect the maximum constraints (≤720px width, ≤1600px height).

## Requirements

- Python 3.7+
- Pillow (PIL) library

```bash
pip install Pillow
```

## Usage

### Quick Start

1. **Run the application**:
   ```bash
   python interactive_image_cropper.py
   # Or use the launcher scripts:
   ./run_image_cropper.sh       # Linux/Mac
   run_image_cropper.bat        # Windows
   ```

2. **Select image directory**: Click "Browse" to choose a folder containing images

3. **Crop images**:
   - Click and drag on the thumbnail to select crop area
   - Click "Crop & Save" to apply to full-resolution image
   - Use "Next" to move to the next image
   - Use "Skip" to skip an image without cropping

4. **Find results**: Cropped images are saved in the "cropped" subdirectory

### Detailed Workflow

1. **Directory Selection**: Browse and select a directory containing images
2. **Image Display**: Each image is shown as a thumbnail (max 800x800 pixels)
3. **Crop Selection**: Click and drag to define the desired crop area
4. **Crop Application**: The crop is applied to the original full-resolution image
5. **Auto-Save**: Cropped image is saved to "cropped" subdirectory
6. **Navigation**: Move through images using Previous/Next/Skip buttons

## Supported Formats

- **Input**: JPG, JPEG, PNG, BMP, TIFF, TIF, GIF
- **Output**: Same format as original image

## Key Features Explained

### Thumbnail Display
- Images are displayed at maximum 800x800 pixels
- Aspect ratio is preserved
- Coordinate mapping ensures accurate crop selection

### Full Resolution Processing
- Crop selections on thumbnails are mapped to original image coordinates
- Final crops maintain maximum possible resolution
- Scale factor automatically calculated and applied

### Batch Processing
- Process entire directories sequentially
- Progress tracking shows current position
- Skip unwanted images easily
- Previous button allows going back

### File Management
- Creates "cropped" subdirectory automatically
- Handles filename conflicts with automatic numbering
- Preserves original image format and quality
- **Filename Format**: Adds final dimensions to filenames
  - Example: `photo.jpg` → `photo_720x1280.jpg`
  - Conflicts resolved as: `photo_720x1280_crop_1.jpg`, `photo_720x1280_crop_2.jpg`, etc.

## Interface Controls

| Button | Function |
|--------|----------|
| **Browse** | Select directory containing images |
| **Previous** | Go to previous image |
| **Crop & Save** | Apply crop to full-resolution image and save |
| **Skip** | Move to next image without cropping |
| **Next** | Move to next image |
| **Clear Selection** | Remove current crop selection |

## File Organization

```
Your Image Directory/
├── image1.jpg
├── image2.png
├── image3.jpg
└── cropped/           # Auto-created
    ├── image1.jpg     # Cropped versions
    ├── image2.png
    └── image3_crop_1.jpg  # Conflict resolution
```

## Tips for Best Results

1. **Image Quality**: Start with high-resolution images for best crop results
2. **Crop Selection**: Make precise selections on thumbnails - they map accurately to originals
3. **Aspect Ratios**: Consider the final use when selecting crop dimensions
4. **Batch Workflow**: Use Skip liberally for unwanted images
5. **File Naming**: Original filenames are preserved in cropped versions

## Technical Details

### Coordinate Mapping
- Thumbnail coordinates are automatically scaled to original image dimensions
- Scale factor calculated as: `original_size / thumbnail_size`
- Ensures pixel-perfect accuracy in final crops

### Memory Management
- Only current image loaded in memory
- Thumbnails created on-demand
- Efficient processing of large image collections

### Error Handling
- Graceful handling of unsupported files
- Automatic skipping of corrupted images
- User-friendly error messages

## Troubleshooting

### Common Issues

**"No images found"**
- Check directory contains supported image formats
- Verify file extensions are correct (.jpg, .png, etc.)

**"Failed to load image"**
- Image file may be corrupted
- Use Skip button to continue with next image

**Crop selection not working**
- Ensure you click and drag within the image area
- Make selection large enough (minimum 10x10 pixels)

**Application won't start**
- Verify Python and Pillow are installed
- Check virtual environment is activated

## Example Workflow

1. **Prepare**: Create test images using `create_test_images.py`
2. **Launch**: Run `python interactive_image_cropper.py`
3. **Select**: Browse to your image directory
4. **Crop**: 
   - Image 1/5 loads automatically
   - Drag to select crop area
   - Click "Crop & Save"
   - Automatically moves to Image 2/5
5. **Continue**: Repeat for all images
6. **Complete**: "All images processed" message appears

## Testing

Generate test images for experimentation:

```bash
python create_test_images.py
```

This creates sample images in `test_images/` directory with various dimensions and patterns.

## License

This project is for educational and personal use. Feel free to modify and distribute according to your needs.
