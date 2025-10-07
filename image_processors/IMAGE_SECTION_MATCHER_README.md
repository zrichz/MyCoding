# Image Section Matcher

## Overview
The Image Section Matcher creates new images by finding the best-fitting sections between two source images. It divides both input images into small sections and uses similarity metrics to match each section from the first image with the most similar section from the second image.

## How It Works
1. **Load Images**: Select two 512x512 images (automatically resized if needed)
2. **Section Division**: Both images are divided into equal-sized sections (8x8, 16x16, 32x32, or 64x64 pixels)
3. **Similarity Matching**: For each section in the source image, find the best matching section from the target image using various similarity algorithms
4. **Result Generation**: Create a new image by assembling the matched sections

## Features

### Section Sizes
- **8×8 pixels**: Very fine detail, 4096 sections per image
- **16×16 pixels**: Good detail balance, 1024 sections per image (default)
- **32×32 pixels**: Larger chunks, 256 sections per image
- **64×64 pixels**: Coarse matching, 64 sections per image

### Similarity Methods
- **Mean Squared Error (MSE)**: Measures pixel-level differences (good for exact color matching)
- **Structural Similarity (SSIM)**: Considers luminance, contrast, and structure
- **Normalized Cross-Correlation (NCC)**: Measures pattern similarity regardless of brightness
- **Histogram Correlation**: Compares color distribution patterns

## Usage

### Running the Tool
- **Linux/Mac**: `./run_image_section_matcher.sh`
- **Windows**: `run_image_section_matcher.bat`
- **Direct**: `python image_section_matcher.py` (requires numpy and pillow)

### Step-by-Step Process
1. **Select Source Image**: This image provides the "template" sections to match
2. **Select Target Image**: This image provides the "palette" of sections to choose from  
3. **Choose Section Size**: Smaller = more detail, larger = more abstract results
4. **Choose Similarity Method**: Different methods emphasize different aspects
5. **Generate**: Click "Generate Matched Image" to create the result
6. **Save**: Save the result image when satisfied

## Creative Applications
- **Texture Transfer**: Apply the texture/style of one image to the structure of another
- **Photomosaics**: Create images that look like one image but are made from pieces of another
- **Art Style Mixing**: Combine artistic styles from different images
- **Pattern Matching**: Find how well patterns from different images can be combined

## Technical Details
- Input images are automatically resized to 512×512 pixels
- All similarity calculations work on RGB color data
- Results are saved as PNG files with timestamp
- Memory usage scales with section size (smaller sections = more memory)
- Processing time depends on section count and similarity method

## Tips for Best Results
- **For photorealistic results**: Use MSE with 16×16 or 32×32 sections
- **For artistic effects**: Try SSIM or NCC with different section sizes
- **For texture mapping**: Use smaller sections (8×8 or 16×16)
- **For abstract art**: Use larger sections (32×32 or 64×64)
- Choose images with complementary content - the target should have variety to match the source

## Performance
- 16×16 sections (1024 comparisons): ~5-10 seconds
- 8×8 sections (4096 comparisons): ~15-30 seconds  
- Processing time varies by similarity method complexity
