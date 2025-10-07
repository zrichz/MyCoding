# Clifford Pickover Biomorphs Generator

## Overview
Interactive fractal generator based on Clifford Pickover's biomorph algorithm. This Python version recreates and enhances the classic DOS BASIC radiolarian biomorph program with a modern GUI, real-time parameter control, and high-resolution output up to 1600x1600 pixels.

## What are Biomorphs?
Biomorphs are fractals that often resemble biological forms - hence the name "bio-morphs" (life forms). They are generated using the iteration:
```
z = z³ + c
```
Where z starts at each pixel coordinate and c is a complex constant. The resulting patterns can look like microscopic organisms, cellular structures, or alien life forms.

## Features

### Interactive Controls
- **Real-time sliders** for all fractal parameters
- **Live preview** - fractal updates as you adjust parameters  
- **Click to zoom** - click anywhere on the fractal to center the view there
- **Preset configurations** for quick exploration

### Fractal Parameters
- **Constant (Real/Imaginary)**: Controls the complex constant added each iteration
- **Zoom Level**: How far zoomed in (0.1x to 10x)
- **Center X/Y**: Pan around the fractal plane (-5 to +5)
- **Max Iterations**: Detail level (10-1000 iterations)
- **Escape Radius**: How far points can go before "escaping" (2-50)
- **Image Size**: Fixed Full HD output resolution (1200x800 pixels)

### Display Options
- **Color Palettes**: Beautiful gradient colors based on iteration count
  - Rainbow, Fire, Ocean, Plasma, Sunset, Forest palettes
  - Smooth color transitions from structure to background  
- **Color Inversion**: Invert colors for different visual effects
- **Grayscale Mode**: Traditional black and white biomorph display
- **High Resolution**: Generate fractals up to 1600x1600 for crisp detail
- **Scrollable View**: Navigate large images with scrollbars

### Built-in Presets
1. **Classic Radiolarian** (0.5 + 0i) - The original BASIC program settings
2. **Spiral Biomorph** (0.7 + 0.2i) - Creates spiral patterns
3. **Complex Branch** (-0.3 + 0.8i) - Branching tree-like structures  
4. **Delicate Web** (0.1 - 0.6i) - Fine web patterns
5. **Dense Forest** (-0.8 + 0.3i) - Dense organic patterns
6. **Radial Pattern** (0.0 + 1.0i) - Radially symmetric forms

## Usage

### Running the Program
- **Linux**: `./run_biomorph_generator.sh`
- **Windows**: `run_biomorph_generator.bat` 
- **Direct**: `python biomorph_generator.py` (requires numpy and pillow)

### Basic Operation
1. **Start** - Program loads with the classic radiolarian preset
2. **Explore** - Use sliders to adjust parameters and watch real-time changes
3. **Navigate** - Click on interesting areas to center the view there
4. **Zoom** - Adjust zoom slider to see fine details or overall structure
5. **Save** - Click "Save Fractal" to export high-resolution PNG files

### Advanced Tips
- **For organic forms**: Try constants with small imaginary parts (0.0 to 0.3)
- **For geometric patterns**: Use larger imaginary constants (0.5 to 1.0)
- **For fine detail**: Increase max iterations (200-500) and use higher resolution
- **For exploration**: Start with presets then make small adjustments
- **Performance**: Lower resolution (400x400) for real-time exploration, high resolution for final images

## Technical Details
- **Algorithm**: Classic z³ + c iteration with escape-time coloring
- **Threading**: Generation runs in background thread (GUI stays responsive)
- **Resolution**: Square images from 400x400 to 1600x1600 pixels
- **Output**: Grayscale PNG/JPEG/BMP with optional color inversion
- **Performance**: ~2-10 seconds for 800x800 image depending on parameters

## Mathematical Background
The biomorph equation z = z³ + c creates patterns because:
- **z³** produces three-fold rotational symmetry
- **Complex arithmetic** creates interference patterns  
- **Iteration** amplifies small differences in starting conditions
- **Escape conditions** define the boundary between stable and chaotic regions

Different constant values (c) produce dramatically different morphologies, from simple radial patterns to complex branching structures that mimic biological forms.

## Comparison to Original BASIC
- **Resolution**: 800x800 default vs. 429x321 original
- **Interactivity**: Real-time parameter adjustment vs. recompile/run cycle
- **Features**: Zoom, pan, presets, high-res export vs. basic display
- **Performance**: Multi-threaded with progress vs. blocking computation
- **Quality**: Anti-aliasing and high-res output vs. low-res pixels
