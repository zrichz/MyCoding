# SuperJPEG Encoder Documentation

## Overview
The SuperJPEG Encoder is a custom JPEG implementation built from first principles with configurable block sizes. Unlike standard JPEG which uses fixed 8x8 blocks, SuperJPEG allows block sizes from 4x4 to 64x64 in steps of 4.

## Key Features

### 🔧 Custom JPEG Implementation
- **Built from scratch**: Complete DCT, quantization, and encoding pipeline
- **Mathematical accuracy**: Proper implementation of all JPEG algorithms
- **Verification system**: Built-in encode/decode testing for validation

### 📐 Variable Block Sizes
- **Range**: 4x4, 8x8, 12x12, 16x16, ..., 64x64 (16 different sizes)
- **Automatic generation**: Creates all block size variants from single input
- **Smart naming**: Files include block size and quality in filename

### 🎛️ Quality Control
- **Range**: 1-100 quality levels
- **Adaptive quantization**: Tables automatically scaled for different block sizes
- **Visual quality**: Similar behavior to standard JPEG quality settings

### 📊 Comprehensive Output
When you encode one image, SuperJPEG creates:
- 16 different SuperJPEG files (one for each block size)
- Descriptive filenames with block size and quality
- JSON format containing all compression data
- Complete metadata for reconstruction

## Technical Implementation

### DCT (Discrete Cosine Transform)
```
Custom DCT matrices generated for each block size:
- 4x4 matrix for 4x4 blocks
- 8x8 matrix for 8x8 blocks  
- 12x12 matrix for 12x12 blocks
- ... up to 64x64 matrix for 64x64 blocks
```

### Quantization Tables
- Base quantization values scaled appropriately for each block size
- **Energy scaling**: Block size scaling factor using square root scaling (N/8.0)^0.5
- **Frequency mapping**: Maps spatial positions to equivalent 8x8 frequencies  
- **Frequency-dependent scaling**: Higher frequencies get more aggressive quantization in larger blocks
- Quality factor applied to control compression level
- Frequency-based coefficients for optimal visual quality across all block sizes

### Color Space Processing
- **RGB to YUV conversion**: Professional color space handling
- **Channel separation**: Y (luminance), Cb/Cr (chrominance) processing
- **YUV to RGB reconstruction**: Accurate color space conversion back

### Data Organization
- **Zigzag scanning**: Proper frequency ordering for compression
- **Run-length encoding**: Efficient representation of sparse data
- **JSON storage**: Human-readable format with all compression data

## File Format

### SuperJPEG File Structure
```json
{
  "magic": "SUPERJPEG",
  "version": "1.0",
  "block_size": 8,
  "quality": 50,
  "original_width": 256,
  "original_height": 256,
  "padded_width": 256,
  "padded_height": 256,
  "channels": 3,
  "quantization_table": [[16, 11, 10, ...], ...],
  "encoded_blocks": [channel_data...]
}
```

### Filename Convention
```
{original_name}_superjpeg_{block_size}x{block_size}_q{quality}.json

Examples:
- photo_superjpeg_4x4_q50.json
- photo_superjpeg_8x8_q75.json
- photo_superjpeg_16x16_q90.json
- landscape_superjpeg_32x32_q25.json
```

## Usage Guide

### 1. Launch the Application
**Windows:**
```cmd
run_superjpeg_encoder.bat
```

**Linux/Mac:**
```bash
./run_superjpeg_encoder.sh
```

**Direct Python:**
```cmd
python superjpeg_encoder.py
```

### 2. Encoding Process
1. **Select Image**: Click "Select Image" and choose a PNG or JPG file
2. **Set Quality**: Adjust quality slider (1-100, default 50)
3. **Encode**: Click "Encode to SuperJPEG (All Block Sizes)"
4. **Wait**: Progress bar shows encoding status for all 16 block sizes
5. **Results**: 16 SuperJPEG files created in same directory as input

### 3. Decoding Process  
1. **Select SuperJPEG**: Click "Select SuperJPEG" and choose a .json file
2. **Decode**: Click "Decode to Image"
3. **View**: Decoded image displays with metadata info
4. **Save**: Decoded PNG automatically saved

## Use Cases

### 🔬 Research & Analysis
- **Block size optimization**: Find optimal block size for different image types
- **Compression efficiency**: Compare file sizes across block sizes
- **Visual quality studies**: Analyze quality vs compression trade-offs
- **Frequency analysis**: Study how different block sizes affect frequency representation

### 📚 Educational Applications
- **JPEG learning**: Understand JPEG compression step-by-step
- **DCT visualization**: See effects of different transform block sizes
- **Quantization effects**: Observe how quality settings affect image data
- **Algorithm understanding**: Complete implementation shows all JPEG components

### 🎨 Creative Applications
- **Artistic effects**: Different block sizes create unique visual artifacts
- **Stylistic compression**: Use large blocks for mosaic-like effects
- **Progressive quality**: Create image sets with varying compression characteristics

### ⚡ Performance Testing
- **Processing speed**: Compare encoding/decoding times across block sizes
- **Memory usage**: Analyze memory requirements for different configurations
- **Quality metrics**: Measure PSNR, SSIM, or other quality metrics

## Block Size Characteristics

### Small Blocks (4x4, 8x8)
- **Higher detail preservation**: Better for images with fine details
- **More processing overhead**: More blocks to process
- **Standard compatibility**: 8x8 matches standard JPEG
- **Fine quantization**: Precise quantization for detail preservation

### Medium Blocks (12x12, 16x16, 20x20)
- **Balanced approach**: Good detail vs efficiency trade-off
- **Smooth gradients**: Better for images with gentle transitions
- **Moderate processing**: Reasonable computational requirements
- **Adaptive quantization**: Scaled quantization for optimal compression

### Large Blocks (32x32, 48x48, 64x64)
- **High compression**: Fewer coefficients to store
- **Aggressive high-frequency quantization**: Optimized for strong compression
- **Energy-aware scaling**: Proper quantization scaling prevents excessive mosaic effects
- **Fast processing**: Fewer blocks to compute
- **Improved quality**: Better quantization scaling maintains visual quality even at large block sizes

## Technical Notes

### Padding Behavior
- Images automatically padded to be divisible by block size
- Padding preserves edge characteristics using "edge" mode
- Original dimensions preserved in metadata for exact reconstruction

### Memory Considerations
- Large block sizes require more memory per block
- Very large images may need chunked processing for large blocks
- JSON format is human-readable but larger than binary formats

### Quality Settings
- Quality 1-49: Aggressive compression, visible artifacts
- Quality 50-75: Balanced compression, good for most uses
- Quality 76-100: High quality, minimal compression artifacts

## Testing and Validation

### Built-in Verification
- Every encoding automatically tests decoding
- Dimension matching verification
- Error reporting for failed operations
- Progress logging shows success/failure for each block size

### Demo Image
Use `create_demo_image.py` to generate a test image:
```cmd
python create_demo_image.py
```
This creates `superjpeg_demo.png` with:
- Colorful gradients
- Geometric shapes
- Text elements
- Perfect for testing different block sizes

## Troubleshooting

### Common Issues
- **"scipy not found"**: Install scipy with `pip install scipy`
- **Memory errors**: Try smaller images or reduce quality for large blocks
- **JSON errors**: Ensure SuperJPEG files are valid JSON format
- **Quality artifacts**: Lower quality settings show more compression artifacts

### Performance Tips
- **Start with small images**: Test with 256x256 or 512x512 images first
- **Monitor memory**: Large blocks with large images use significant memory
- **Quality selection**: Use quality 50-75 for most applications
- **File management**: 16 files per input image - organize output directories

## Future Enhancements

### Potential Improvements
- **Binary format**: More compact file format option
- **Huffman coding**: Add entropy coding for better compression
- **Progressive JPEG**: Multi-resolution encoding support
- **Batch processing**: Process multiple images automatically
- **Comparison tools**: Visual comparison between block sizes
- **Metrics calculation**: Automatic PSNR/SSIM calculation

### Advanced Features
- **Custom quantization**: User-defined quantization tables
- **Subsampling options**: Chroma subsampling support
- **Color space options**: Support for other color spaces
- **Lossless mode**: Optional lossless compression mode

## Technical Deep Dive

### Improved Quantization Scaling

The SuperJPEG encoder uses advanced quantization scaling that properly accounts for block size differences:

#### Energy Scaling
```python
block_scale_factor = (N / 8.0) ** 0.5  # Square root scaling for energy
```
- Larger blocks capture more energy in DCT coefficients
- Square root scaling accounts for the 2D nature of the transform
- 64x64 blocks have ~2.83x energy scaling vs 8x8 blocks

#### Frequency Mapping
```python
freq_i = i * 8.0 / N  # Map block position to equivalent 8x8 frequency
freq_j = j * 8.0 / N
```
- Maps spatial positions to equivalent frequencies in standard 8x8 blocks
- Preserves frequency relationships across different block sizes
- Ensures consistent compression behavior

#### Frequency-Dependent Scaling
```python
freq_scale = 1.0 + (freq_magnitude / 8.0) * (N / 8.0 - 1.0) * 0.5
```
- Higher frequencies get more aggressive quantization in larger blocks
- Accounts for how larger blocks spread high-frequency content
- Progressive scaling based on both frequency position and block size

#### Benefits of Proper Scaling
- **Consistent perceptual quality** across all block sizes
- **Energy-aware quantization** appropriate for each block size
- **Better compression efficiency** while maintaining quality
- **Reduced mosaic effects** in large blocks while preserving compression benefits

---

**SuperJPEG Encoder - Custom JPEG Implementation with Variable Block Sizes**  
*Part of the Image Processors Collection*
