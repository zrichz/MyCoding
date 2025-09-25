# Image Numberer 1440x3200

A GUI application for batch processing 1440x3200 pixel images by adding unique sequential numbers to the four corners of each image.

## Features

- **Target Resolution**: Works exclusively with 1440x3200 pixel images
- **Corner Numbering**: Adds four sequential numbers to each image's corners
  - First image: 0001 (top-left), 0002 (top-right), 0003 (bottom-left), 0004 (bottom-right)
  - Second image: 0005, 0006, 0007, 0008
  - And so on...
- **Text Style**: Small black text with white border for maximum visibility
- **High Quality Output**: Saves as JPEG with maximum quality (Q=1/100%)
- **Batch Processing**: Process entire directories at once
- **Counter Management**: Reset counter or continue from where you left off

## Usage

### Windows
```cmd
run_image_numberer.bat
```

### Linux/Mac
```bash
./run_image_numberer.sh
```

### Manual Python Execution
```bash
python image_numberer_1440x3200.py
```

## Requirements

- Python 3.x
- PIL/Pillow
- tkinter (usually included with Python)

Install dependencies:
```bash
pip install Pillow
```

## How It Works

1. **Select Directory**: Choose a folder containing your 1440x3200 images
2. **Automatic Validation**: Only processes images that are exactly 1440x3200 pixels
3. **Sequential Numbering**: Each image gets 4 consecutive numbers in its corners
4. **Output**: Creates a `numbered` subdirectory with processed images
5. **Filename Format**: `YYYYMMDDHHMMSS_XXX_numbered_1440x3200.jpg`

## File Structure

```
your_images/
├── image1.jpg (1440x3200)
├── image2.png (1440x3200)
├── other_file.jpg (different size - skipped)
└── numbered/           # Created automatically
    ├── 20240923143022_001_numbered_1440x3200.jpg
    └── 20240923143022_002_numbered_1440x3200.jpg
```

## Text Positioning

- **Top-left**: 20px from left edge, 20px from top
- **Top-right**: 20px from right edge, 20px from top  
- **Bottom-left**: 20px from left edge, 20px from bottom
- **Bottom-right**: 20px from right edge, 20px from bottom

## Counter Management

The application maintains a global counter across all processed images:
- Use "Reset Counter" to start from 0001
- Counter persists during the session
- Each image consumes 4 consecutive numbers

## Error Handling

- Skips images that aren't exactly 1440x3200 pixels
- Logs all processing attempts with timestamps
- Creates unique output filenames to prevent conflicts
- Graceful handling of file access errors

## Output Quality

- **Format**: JPEG
- **Quality**: 100% (equivalent to Q=1 in JPEG terminology)
- **Optimization**: Enabled for best compression without quality loss