# Font Similarity ASCII Art Generator

A Python GUI application that creates ASCII art using least-squares error similarity to match font characters with image chunks.

## How It Works

Unlike traditional ASCII art that maps brightness to predetermined characters, this tool:

1. **Renders each character** from your chosen font as an 8×8 bitmap
2. **Divides your image** into a uniform N×N grid (8, 16, 32, or 64 pixels per chunk)
3. **Converts each chunk** to grayscale and resizes to 8×8 for comparison
4. **Calculates least-squares error** between each chunk and every character bitmap
5. **Selects the character** with the lowest error (best visual match)

This approach produces ASCII art that actually resembles the shape and texture of your image, not just its brightness levels.

## Features

### **Grid Size Selection**
- **8×8 pixels**: Very detailed, high resolution ASCII (64×64 character grid)
- **16×16 pixels**: Good detail, manageable size (32×32 character grid) 
- **32×32 pixels**: Medium detail, larger characters (16×16 character grid)
- **64×64 pixels**: Low detail, very large characters (8×8 character grid)

### **Font Selection**
- Browse and select any **TTF or OTF font file**
- Each character rendered as 8×8 bitmap for comparison
- Supports system fonts and custom fonts

### **Character Sets**
- **Printable ASCII**: All standard printable characters (95 chars)
- **Letters + Numbers**: Alphanumeric characters only
- **Letters Only**: A-Z and a-z only
- **Numbers + Symbols**: Digits and common symbols
- **Extended ASCII**: Full extended character set
- **Custom**: Define your own character set

### **Output Control**
- **Scale Factor**: 0.5× to 4.0× output sizing
- **Color Options**: Black text on white, or white text on black
- **Save Function**: Export as PNG or JPEG

## Requirements

- Python 3.6+
- NumPy (for array operations)
- Pillow/PIL (for image processing)
- Tkinter (usually included with Python)

## Installation & Usage

### Quick Start (Recommended):
```bash
# Linux/Mac:
chmod +x run_font_similarity_ascii.sh
./run_font_similarity_ascii.sh

# Windows:
run_font_similarity_ascii.bat
```

### Manual Installation:
```bash
pip install numpy pillow
python font_similarity_ascii.py
```

## How to Use

1. **Select Image**: Browse to choose any image (will be resized to 512×512)
2. **Choose Grid Size**: Select 8, 16, 32, or 64 pixel chunks
3. **Select Font**: Browse to any TTF/OTF font file on your system
4. **Pick Character Set**: Choose preset or define custom characters
5. **Adjust Settings**: Set scale factor and colors if desired
6. **Generate**: Click "Generate Font-Matched ASCII Art"
7. **Save**: Use the Save button to export your result

## Technical Details

### **Least-Squares Error Calculation**
For each image chunk and character bitmap:
```
LSE = Σ(pixel_chunk - pixel_char)²
```
The character with the minimum LSE is selected.

### **Processing Pipeline**
1. Image → 512×512 RGB
2. Divide into N×N pixel chunks  
3. Convert chunks to grayscale using luminance weights (0.299R + 0.587G + 0.114B)
4. Resize chunks to 8×8 for comparison
5. Compare with pre-rendered 8×8 character bitmaps
6. Select best match using LSE
7. Render final ASCII art at chosen scale

### **Memory Efficiency**
- Character bitmaps cached after first generation
- Cache cleared when font or character set changes
- Progressive processing with progress updates

## Example Results

- **Detailed (8×8 grid)**: 4096 characters, very fine detail
- **Balanced (16×16 grid)**: 1024 characters, good balance of detail and readability  
- **Readable (32×32 grid)**: 256 characters, clear and readable
- **Artistic (64×64 grid)**: 64 characters, bold and stylized

## Tips for Best Results

1. **Font Choice**: Monospace fonts work best (Courier, Consolas, etc.)
2. **Character Sets**: More characters = better matches, but slower processing
3. **Grid Size**: Smaller grids = more detail but harder to read
4. **Images**: High contrast images produce more interesting results
5. **Scale Factor**: Increase for better readability in final output

This tool creates true font-based ASCII art where each character is chosen for its visual similarity to the image content, resulting in remarkably detailed and accurate representations!
