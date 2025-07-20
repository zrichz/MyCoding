# World's Best Focus Stacker 🔥

An advanced focus stacking application with multiple state-of-the-art algorithms, modern GUI, and comprehensive image alignment capabilities.

## Features ✨

### Multiple Stacking Algorithms
- **Laplacian Pyramid**: Multi-scale analysis with configurable levels and Gaussian sigma
- **Gradient-based**: High-precision edge detection using Sobel operators
- **Variance-based**: Local variance analysis with feathered transitions
- **Wavelet-based**: Discrete wavelet transform stacking (optional)

### Advanced Image Alignment
- **Auto Alignment**: Intelligent method selection
- **ECC (Enhanced Correlation Coefficient)**: Robust geometric alignment
- **Feature-based**: SIFT/ORB feature matching with RANSAC
- **Phase Correlation**: Fast translation-only alignment

### Modern GUI
- **Beautiful Interface**: Dark theme with CustomTkinter
- **Real-time Preview**: Zoom, pan, and navigate through images
- **Parameter Adjustment**: Live sliders for algorithm parameters
- **Quality Assessment**: Built-in metrics and analysis
- **Batch Processing**: Handle multiple image sets

### Command Line Interface
- **Flexible Usage**: GUI or CLI modes
- **Batch Operations**: Process multiple files with wildcards
- **Quality Metrics**: Automated assessment reporting
- **Multiple Formats**: Support for JPEG, PNG, TIFF, BMP

## Installation 📦

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pillow scipy scikit-image matplotlib customtkinter tqdm
   ```
3. **Optional for wavelet stacking**:
   ```bash
   pip install PyWavelets
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

# With quality assessment
python main.py photos/*.tiff --method variance --quality

# All options
python main.py *.jpg --method laplacian --align auto --levels 6 --sigma 1.5 --output stacked.png --quality
```

### CLI Options
- `--method`: Stacking algorithm (`laplacian`, `gradient`, `variance`)
- `--align`: Alignment method (`auto`, `ecc`, `feature`, `phase`, `none`)
- `--output`: Output filename (default: `stacked_result.png`)
- `--levels`: Pyramid levels for Laplacian method (default: 5)
- `--sigma`: Gaussian sigma for blurring (default: 1.0)
- `--quality`: Show quality assessment metrics

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

#### Variance-based
- Computes local variance in sliding windows
- Smooths focus maps with Gaussian filtering
- Uses feathered blending for seamless transitions
- Best for macro photography with fine details

### Quality Metrics
- **Focus Measure**: Quantifies overall sharpness
- **Improvement Ratio**: Compares result to best input
- **Gradient Analysis**: Edge preservation assessment
- **Variance Analysis**: Detail retention evaluation

## Best Practices 📸

### Photography Tips
1. **Use a Tripod**: Minimize camera shake between shots
2. **Manual Focus**: Take shots at different focus distances
3. **Consistent Exposure**: Lock exposure settings
4. **Overlap Focus**: Ensure adequate focus overlap between shots
5. **Stable Lighting**: Avoid changing light conditions

### Software Tips
1. **Alignment First**: Always align images before stacking
2. **Algorithm Selection**: 
   - Laplacian Pyramid: General purpose, best overall
   - Gradient-based: High contrast subjects
   - Variance-based: Macro photography with fine textures
3. **Parameter Tuning**: Adjust pyramid levels and sigma based on image content
4. **Quality Check**: Use built-in metrics to verify results

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

## Examples 🖼️

### Macro Photography
```bash
python main.py macro_stack/*.jpg --method variance --align ecc --levels 6
```

### Landscape Photography
```bash
python main.py landscape/*.tiff --method laplacian --align auto --sigma 1.2
```

### Scientific/Microscopy
```bash
python main.py microscopy/*.png --method gradient --align feature --quality
```

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

## Troubleshooting 🔧

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade opencv-python numpy pillow scipy scikit-image matplotlib customtkinter
   ```

2. **Memory Issues with Large Images**
   - Use fewer pyramid levels
   - Process in smaller batches
   - Reduce image size before stacking

3. **Alignment Failures**
   - Try different alignment methods
   - Ensure sufficient overlap between shots
   - Check for excessive camera movement

4. **Poor Stacking Results**
   - Verify input image quality
   - Adjust algorithm parameters
   - Use quality assessment to compare methods

### Performance Issues
- Close other applications during processing
- Use SSD storage for faster I/O
- Ensure sufficient RAM (8GB+ recommended)

## Advanced Features 🎛️

### Custom Parameters
- Pyramid levels: Control detail vs. speed tradeoff
- Gaussian sigma: Adjust smoothing strength
- Window sizes: Customize local analysis regions
- Thresholds: Fine-tune sensitivity

### Batch Processing
- Process multiple image sets
- Automated parameter optimization
- Quality-based algorithm selection
- Export format customization

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
- PyWavelets (for wavelet-based stacking)
- rawpy (for RAW image support)

## License 📜

MIT License - Feel free to use, modify, and distribute!

## Contributing 🤝

Contributions welcome! Areas for improvement:
- Additional stacking algorithms
- GPU acceleration
- Raw image processing
- Advanced alignment methods
- UI/UX enhancements

## Support 💬

For issues, questions, or feature requests, please check the troubleshooting section first or create an issue with:
- Python version
- Operating system
- Input image details
- Error messages
- Expected vs. actual behavior

---

**Happy Focus Stacking!** 📷✨
