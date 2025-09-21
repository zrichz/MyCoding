# DEBLUR - Image Deblurring Project

A Python-based project for implementing and testing various image deblurring techniques including motion blur removal, Gaussian blur correction, and other deconvolution methods.

## Features

- **Gaussian Deblurring**: Richardson-Lucy and Wiener deconvolution algorithms
  - Progress display with iteration counting
  - Automatic downsampling for large images (>768px) for faster processing
  - Manual downsample control (1x, 2x, 4x)
  - Smart upsampling back to original resolution
- **Motion Deblurring**: Wiener filtering, inverse filtering, and Lucy-Richardson deconvolution
- **GUI Interface**: User-friendly tkinter interface with downsample options
- **Batch Processing**: Process multiple images automatically
- **Performance Metrics**: PSNR calculation and timing comparisons
- **Example Scripts**: Comprehensive examples and test cases

## Installation

The project uses a Python virtual environment. Dependencies are automatically installed.

### Quick Start

1. **Launch GUI Application**:
```bash
python main.py --gui
```

2. **Command Line Usage**:
```bash
# Gaussian deblurring with progress display
python main.py -i input.jpg -o output.jpg --method gaussian --kernel-size 15 --iterations 30

# Gaussian deblurring with manual 2x downsampling
python main.py -i large_image.jpg -o output.jpg --method gaussian --downsample 2

# Gaussian deblurring with auto-downsampling disabled
python main.py -i input.jpg -o output.jpg --method gaussian --downsample 1

# Motion deblurring  
python main.py -i input.jpg -o output.jpg --method motion --angle 45 --length 20

# Silent processing (no progress output)
python main.py -i input.jpg -o output.jpg --method gaussian --no-progress
```

3. **Batch Processing**:
```bash
python examples/batch_process.py input_folder output_folder --method gaussian
```

## Project Structure

```
DEBLUR/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source code
│   ├── deblur/           # Core deblurring algorithms
│   │   ├── gaussian_deblur.py    # Gaussian blur removal
│   │   └── motion_deblur.py      # Motion blur removal
│   ├── gui/              # GUI interface
│   │   └── main_window.py        # Main application window
│   └── utils/            # Utility functions
│       └── image_utils.py        # Image I/O and processing
├── examples/             # Example scripts
│   ├── gaussian_example.py      # Gaussian deblurring demo
│   ├── motion_example.py        # Motion deblurring demo
│   └── batch_process.py         # Batch processing script
├── tests/               # Unit tests
│   └── test_deblur.py           # Test suite
└── .venv/              # Python virtual environment
```

## Usage Examples

### GUI Interface
```python
from src.gui.main_window import DeblurApp
app = DeblurApp()
app.run()
```

### Gaussian Deblurring
```python
from src.deblur.gaussian_deblur import GaussianDeblur
from src.utils.image_utils import load_image, save_image

deblurrer = GaussianDeblur()
image = load_image('blurred.jpg')

# Basic deblurring
result = deblurrer.deblur_image(image, kernel_size=15, iterations=30)

# With automatic downsampling for large images
result = deblurrer.deblur_image(image, kernel_size=15, iterations=30, 
                               auto_downsample=True, show_progress=True)

# With manual 2x downsampling
result = deblurrer.deblur_image(image, kernel_size=15, iterations=30, 
                               downsample_factor=2, show_progress=True)

save_image(result, 'deblurred.jpg')
```

### Motion Deblurring
```python
from src.deblur.motion_deblur import MotionDeblur
from src.utils.image_utils import load_image, save_image

deblurrer = MotionDeblur()
image = load_image('motion_blurred.jpg')
result = deblurrer.remove_motion_blur(image, angle=45, length=20)
save_image(result, 'deblurred.jpg')
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Or run the test script directly:
```bash
python tests/test_deblur.py
```

## Development

The project follows these guidelines:
- Modular design with separate algorithms
- Comprehensive error handling
- Type hints for better documentation
- Performance timing for algorithm comparison
- Extensive documentation and examples

## Dependencies

- **OpenCV**: Image processing operations
- **NumPy**: Numerical computations  
- **SciPy**: Scientific computing functions
- **scikit-image**: Advanced image processing
- **Pillow**: Image I/O operations
- **Matplotlib**: Visualization and results display

## License

MIT License
