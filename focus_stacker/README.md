# World's Worst Focus Stacker 🔥

focus stacking app with multiple algorithms, GUI, and image alignment capabilities.

## Features ✨

### Multiple Stacking Algorithms
- **Laplacian Pyramid**: Multi-scale analysis with configurable levels and Gaussian sigma
- **Gradient-based**: edge detection using Sobel operators
- **Variance-based**: Local variance analysis with feathered transitions
- **basic averaging**: average all frames

### Image Alignment
- **Auto Alignment**: Intelligent method selection
- **ECC (Enhanced Correlation Coefficient)**: Robust geometric alignment
- **Feature-based**: SIFT/ORB feature matching with RANSAC
- **Phase Correlation**: Fast translation-only alignment

### GUI
- Dark theme with CustomTkinter
- **Real-time Preview**: Zoom, pan, and navigate through images
- **Parameter Adjustment**: Live sliders for algorithm parameters

### Command Line Interface
- **Flexible Usage**: GUI or CLI modes
- **Batch Operations**: Process multiple files with wildcards
- **Multiple Formats**: Support for JPEG, PNG, TIFF, BMP

## Installation 📦

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pillow scipy scikit-image matplotlib customtkinter tqdm
   ```

## Usage 🚀

### GUI Mode (Recommended)
```bash
python main.py
```
or
```bash
python main.py --gui
```

### Command Line Mode
```bash
# Basic usage
python main.py image1.jpg image2.jpg image3.jpg

# With specific algorithm and alignment
python main.py *.jpg --method laplacian --align ecc --output result.png

# All options
python main.py *.jpg --method laplacian --align auto --levels 6 --sigma 1.5 --output stacked.png
```

### CLI Options
- `--method`: Stacking algorithm (`laplacian`, `gradient`, `variance`)
- `--align`: Alignment method (`auto`, `ecc`, `feature`, `phase`, `none`)
- `--output`: Output filename (default: `stacked_result.png`)
- `--levels`: Pyramid levels for Laplacian method (default: 5)
- `--sigma`: Gaussian sigma for blurring (default: 1.0)

## How It Works 🔬

### Focus Stacking Process
1. **Image Loading**: Support for multiple formats including RAW (with rawpy)
2. **Alignment**: Compensate for camera movement between shots
3. **Focus Analysis**: Identify sharpest regions in each image
4. **Blending**: Combine sharp regions with smooth transitions
5. **Output**: High-quality extended depth-of-field image

### Algorithm Details

#### Laplacian Pyramid
- Builds multi-scale representations of each image
- Uses variance of Laplacian as focus measure
- Combines pyramid levels based on sharpness
- Reconstructs final image from combined pyramid

#### Gradient-based
- Calculates gradient magnitude using Sobel operators
- Selects pixels with highest gradient values
- Applies threshold to eliminate noise
- Direct pixel-wise selection with optional smoothing
- Optional Debug visualisation

#### Variance-based
- Computes local variance in sliding windows
- Smooths focus maps with Gaussian filtering
- Uses feathered blending for seamless transitions
- Best for macro photography with fine details

### Software Tips
1. **Alignment First**: Always align images before stacking
2. **Algorithm Selection**: 
   - Laplacian Pyramid: General purpose, best overall
   - Gradient-based: High contrast subjects
   - Variance-based: Macro photography with fine textures
3. **Parameter Tuning**: Adjust pyramid levels and sigma based on image content

## Supported Formats 📁

### Input Formats
- JPEG/JPG
- PNG
- TIFF/TIF
- BMP
- RAW formats (with rawpy)

### Output Formats
- PNG (recommended for quality)
- JPEG (smaller file size)
- TIFF (professional workflows)

## Performance 🚀

### Optimization Features
- **Multi-threaded Processing**: Utilizes multiple CPU cores
- **Memory Efficient**: Processes images in chunks for large files
- **Progress Monitoring**: Real-time feedback during processing
- **Error Recovery**: Graceful handling of problematic images

### Typical Processing Times
- **Small images** (1MP): ~2-5 seconds for 5 images
- **Medium images** (12MP): ~15-30 seconds for 5 images  
- **Large images** (24MP+): ~1-2 minutes for 5 images

*Times vary based on CPU, algorithm choice, and number of images*


## Dependencies 📋

### Required
- Python 3.7+
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- SciPy
- scikit-image
- matplotlib
- customtkinter
- tqdm

### Optional
- rawpy (for RAW image support)

## License 📜
MIT License

## Support 💬
*zero* support
