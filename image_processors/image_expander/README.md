# Image Expansion Tools

Tools for expanding images with various effects and algorithms.

## Files

### image_expander_720x1600.py
- **Purpose**: Auto-expand images to exactly 720x1600 pixels with batch processing
- **Features**:
  - Intelligent dual-axis expansion
  - Fixed 160px blur and 50% luminance reduction for natural fade effect
  - Batch processing with progress tracking
  - Centers original image within expanded canvas
  - **Timestamp filename schema**: `YYYYMMDDHHMMSS_nnn_720x1600.ext`
  - Automatic conflict resolution (increments counter 001, 002, etc.)
- **Usage**: 
  - **Easy launch**: `../run_image_expander.sh` (Linux/Mac) or `../run_image_expander.bat` (Windows)
  - **Direct**: `python image_expander_720x1600.py`

### image_expander_cylindrical.py
- **Purpose**: Cylindrical image expansion
- **Features**: Specialized cylindrical expansion algorithm
- **Usage**: `python image_expander_cylindrical.py`
