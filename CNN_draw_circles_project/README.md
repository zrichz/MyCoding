# CNN Draw Circles Project

This project creates a dataset of 64x64 binary images with random circular rings and trains an autoencoder to recreate them from noise.

## Features

- **Dataset Generation**: Creates 600 binary images (64x64) with 1-5 random circular rings
- **Persistent Storage**: Saves images to `ring_dataset/` directory for reuse
- **Ring Properties**: Variable centers (can be outside image bounds), radii, and thickness
- **Autoencoder Architecture**: Encoder-decoder CNN with 32-dimensional latent space
- **Visualization**: Shows original images, reconstructions, and generations from pure noise
- **Statistics**: Training loss plots and comprehensive metrics
- **Model Persistence**: Saves trained model as `ring_autoencoder.pth`

## Dataset

- **Size**: 600 images (64x64 pixels)
- **Format**: PNG files in `ring_dataset/` directory
- **Content**: 1-5 random white rings on black background
- **Generation**: Automatic on first run, reuses existing dataset on subsequent runs

## Running the Script

### Option 1: Using the Shell Script (Recommended)
```bash
./run_with_venv.sh
```

### Option 2: Direct Python Execution
```bash
./CNN_drawCircles.py
```

### Option 3: Manual Virtual Environment Activation
```bash
source /home/rich/MyCoding/textual_inversions/.venv/bin/activate
python3 CNN_drawCircles.py
```

## Virtual Environment

The script uses the existing virtual environment at:
`/home/rich/MyCoding/textual_inversions/.venv/`

This environment includes:
- PyTorch 2.8.0
- NumPy 2.3.2  
- Matplotlib 3.10.5
- Pillow (for image I/O)

## Output

The script will display:
1. Dataset generation progress (if creating new dataset)
2. Dataset statistics (size, intensity distributions)
3. Sample dataset images
4. Model parameter count
5. Training progress (loss per epoch for 50 epochs)
6. Training loss plot with grid
7. Original vs reconstructed images comparison
8. Images generated from random noise
9. Comprehensive final statistics

## Model Architecture

- **Encoder**: 3 conv layers (16→32→64 channels) + linear layer to 32D latent space
- **Decoder**: Linear layer + 3 transposed conv layers back to 64x64 image
- **Training**: 50 epochs with Adam optimizer and BCE loss
- **Parameters**: 348,865 total parameters
- **Device**: CPU (due to old GPU compatibility issues)

## Files Created

- `ring_dataset/`: Directory containing 600 PNG images (ring_0000.png to ring_0599.png)
- `ring_autoencoder.pth`: Trained model weights and metadata
- Training visualizations (displayed during execution)

## Performance

With 600 images and 50 epochs:
- **Loss improvement**: ~95.6% (from ~0.43 to ~0.019)
- **Final loss**: ~0.019 BCE loss
- **Training time**: ~10-15 minutes on CPU

## Notes

- The script automatically detects existing datasets and reuses them
- To generate a new dataset, delete the `ring_dataset/` directory
- Ring centers can be outside the 64x64 image boundary for partial rings
- Images are normalized to [0,1] range for training
- The autoencoder learns to compress ring patterns into a 32D latent space
