# 2D Neural Cellular Automata for Image Recreation

This project implements a Neural Cellular Automata (NCA) that can learn to recreate target images from random noise. The system includes a user-friendly tkinter GUI for easy interaction.

## Features

- **Enhanced Large Display**: Optimized for 1920x1080 displays with larger visualization panels
- **Image Loading**: Load any image file (PNG, JPG, JPEG, BMP, GIF) as a target
- **Neural CA Training**: Train the model to recreate the target image from noise
- **Real-time Visualization**: See training progress with live plots and image updates
- **Auto-Generation**: Automatically generate images every 8 seconds during training
- **Enhanced Progress Monitoring**: 
  - Live epoch counter and training rate display
  - Real-time loss visualization with trend analysis
  - Cell state visualization with live cell count
  - Model status indicators
- **Model Persistence**: Save and load trained models
- **Adjustable Parameters**: Customize channel count and fire rate
- **Interactive GUI**: Easy-to-use interface built with tkinter
- **Device Information**: Shows whether using CPU or GPU with device details

## How Neural Cellular Automata Work

Neural Cellular Automata are a type of neural network that operates on a grid of cells, similar to Conway's Game of Life, but with learned rules instead of fixed ones. Each cell has multiple channels (like RGBA plus additional hidden channels), and the network learns update rules that can create complex patterns and images.

Key components:
- **Perception**: Each cell "sees" its local neighborhood using Sobel filters
- **Update Network**: A neural network that decides how each cell should change
- **Living Mask**: Cells with low alpha values are considered "dead" and don't update
- **Fire Rate**: Controls how often cells actually update (adds stochasticity)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python NCA_baseline.py
```

## Usage

### Basic Workflow

1. **Load an Image**: Click "Load Image" and select your target image
   - Images are automatically resized to maximum 256x256 pixels for better detail
   - RGBA format is used (RGB + alpha channel)

2. **Initialize Model**: Click "Initialize Model" to create a new NCA
   - Clear visual feedback shows when initialization is complete
   - Model status indicator changes from "No model" to "Model Ready"
   - Adjust parameters before initializing if desired
   - Channels: Number of cell channels (8-32, default 16)
   - Fire Rate: Probability of cell updates (0.1-1.0, default 0.5)

3. **Start Training**: Click "Start Training" to begin the learning process
   - The model will learn to recreate your image from a single seed cell
   - Watch the real-time visualization showing progress
   - Training loss should decrease over time
   - Live metrics show epochs per second and current loss

4. **Auto-Generation**: Enable "Auto-generate every 8s" for continuous preview
   - Automatically generates new images every 8 seconds during training
   - Shows progress without manual intervention
   - Can be toggled on/off anytime

5. **Manual Generation**: Click "Generate Now" to create a new image from scratch
   - Uses the current model state
   - Starts from a single cell and evolves for 128 steps
   - Shows similarity score compared to target

6. **Save/Load Models**: Preserve your trained models
   - Save trained models as .pth files
   - Load previously trained models to continue training or generate images

### Tips for Best Results

- **Image Choice**: Simple images with clear shapes work better than complex photographs
- **Training Time**: Let the model train for several hundred epochs for good results
- **Parameters**: 
  - More channels allow for more complex patterns but train slower
  - Lower fire rates create more stable patterns
  - Higher fire rates allow for more dynamic evolution

### Understanding the Enhanced Visualization

- **Target Image**: Your loaded reference image
- **Current Output**: The model's current attempt at recreation
  - Shows epoch number in title
  - Displays similarity score when generated manually
- **Training Loss**: How well the model is performing (lower is better)
  - Real-time plot with trend analysis
  - Recent trend line shows learning direction
- **Cell States (Alpha)**: Shows which cells are "alive" 
  - Colorful visualization using viridis colormap
  - Live cell count displayed in title
  - Colorbar shows alpha intensity scale

### Real-time Feedback Features

- **Status Indicators**: Clear visual feedback for model state
- **Progress Metrics**: Live display of epochs per second
- **Auto-updates**: Continuous display refresh every 500ms
- **Enhanced Plots**: Larger, more detailed visualizations
- **Similarity Scoring**: Automatic similarity calculation for generated images

## Technical Details

### Architecture

The NCA uses a simple but effective architecture:
- **Perception Layer**: 3×3 convolution detecting local patterns using Sobel filters
- **Update Network**: Two-layer MLP that processes perceived information
- **Residual Updates**: Cells update by adding small changes rather than replacing values

### Training

- **Loss Function**: Mean Squared Error between generated and target images
- **Optimizer**: Adam with learning rate 2e-3 and exponential decay
- **Batch Size**: Single image per batch
- **Steps**: Random number of evolution steps (64-96) per training iteration

### Performance

- Supports both CPU and GPU (CUDA) training
- Automatically detects available hardware
- GUI updates run in separate thread to maintain responsiveness

## Troubleshooting

### Common Issues

1. **"Please load a target image first"**: Load an image before initializing the model
2. **"Please initialize model first"**: Initialize the model before training or generating
3. **Training not converging**: Try adjusting parameters or using a simpler target image
4. **GUI freezing**: Training runs in background thread - GUI should remain responsive

### Performance Tips

- Use GPU if available for faster training
- Larger images (up to 256x256) provide more detail but train slower
- Reduce number of channels if training is too slow

## Examples

The system works well with:
- Simple geometric shapes
- Logos and symbols
- Pixel art
- Simple drawings

It may struggle with:
- Photographic images with complex textures
- Very detailed images
- Images with many small features

## License

This project is created for educational and research purposes. Feel free to modify and experiment with the code.

## Credits

Based on the Neural Cellular Automata concept from "Growing Neural Cellular Automata" by Mordvintsev et al. (2020).
