# NCA Animation Generator 🎬

Create mesmerizing animated GIFs from your trained Neural Cellular Automata models! Perfect for sharing the beautiful emergence process on social media.

## 🌟 Features

✅ **Multiple Output Sizes**: 32x32, 64x64, 128x128, 256x256, and 512x512 pixel outputs  
✅ **Smart Scaling**: 32x32 and 64x64 are automatically scaled up with nearest neighbor for crisp pixelated look  
✅ **Optimized for Social Media**: Fixed 30fps (33ms per frame) for smooth playback  
✅ **Multiple Initialization Types**: Center, random single, random multi, sparse, edge, and circle patterns  
✅ **User-Friendly GUI**: Streamlined interface focused on GIF creation  
✅ **Command-Line Support**: Batch processing and automation capabilities  
✅ **Step Labeling**: Optional frame numbering for analysis  
✅ **CUDA Acceleration**: GPU support for faster processing

## What it does

The animation generator loads your trained NCA models and captures every step of the evolution process, creating smooth animated GIFs that show how your patterns emerge from simple seeds.

## Files

- **`nca_animator.py`** - Full-featured GUI application
- **`nca_quick_animator.py`** - Command-line version for batch processing
- **`launch_animator.bat`** - Windows launcher for the GUI

## 🚀 Quick Start

### GUI Version (Recommended)

1. **Train and save a model** using `NCA_baseline.py`
2. **Launch the animator**:
   ```cmd
   launch_animator.bat
   ```
3. **Load your model** (.pth file)
4. **Choose settings** and click "Create GIF"

### Command Line Version

```cmd
# Basic usage
python nca_quick_animator.py my_model.pth evolution.gif

# Advanced usage with new size options
python nca_quick_animator.py my_model.pth tiny.gif --init circle --steps 200 --size 32
python nca_quick_animator.py my_model.pth small.gif --init random_multi --size 64
```

## 🎨 Initialization Types

| Type | Description | Best for |
|------|-------------|----------|
| `center` | Single pixel at center | Classic emergence |
| `random_single` | One random point | Varied starting positions |
| `random_multi` | Multiple random points | Complex interactions |
| `sparse` | Scattered random pixels | Organic growth |
| `edge` | Starting from edge | Directional growth |
| `circle` | Small circle seed | Radial expansion |

## ⚙️ Settings Guide

### Output Sizes
- **32x32**: Processed at 32x32, then scaled 4x to 128x128 with nearest neighbor for pixelated look
- **64x64**: Processed at 64x64, then scaled 2x to 128x128 with nearest neighbor for pixelated look  
- **128x128**: Native resolution, no scaling
- **256x256**: Native resolution, no scaling
- **512x512**: Native resolution, no scaling

### Animation Parameters

### Animation Settings

- **Total Steps**: 50-500 (more = longer evolution)
- **Frame Interval**: 1-10 (lower = smoother, larger files)
- **Output Size**: 128/256/512px (balance quality vs file size)
- **GIF Speed**: 50-500ms per frame (lower = faster)

### File Size Tips

For **social media friendly** GIFs (< 8MB):
- Use 256px or smaller
- Frame interval of 2-3
- 100-150 total steps
- Enable optimization

For **high quality** showcase:
- Use 512px
- Frame interval of 1
- 200+ steps
- Larger file sizes

## 📱 Social Media Specs

| Platform | Max Size | Recommended |
|----------|----------|-------------|
| **Mastodon** | 8MB | 256px, 150 steps |
| **Twitter** | 15MB | 512px, 200 steps |
| **Reddit** | 100MB | Any size |
| **Discord** | 8MB | 256px, 150 steps |

## 🎯 Pro Tips

1. **Preview first** - Always generate a preview to check your settings
2. **Use random seeds** - Add `--seed 42` for reproducible results
3. **Batch processing** - Use the command line version for multiple GIFs
4. **Experiment with initialization** - Different init types create unique patterns
5. **Step labels** - Enable step numbers to show the evolution process

## 🛠️ Advanced Usage

### Batch Generate Multiple Animations

```batch
@echo off
for %%i in (*.pth) do (
    python nca_quick_animator.py "%%i" "animations\%%~ni_center.gif" --init center
    python nca_quick_animator.py "%%i" "animations\%%~ni_random.gif" --init random_multi
    python nca_quick_animator.py "%%i" "animations\%%~ni_circle.gif" --init circle
)
```

### Custom Sizes for Different Platforms

```cmd
# Twitter optimized
python nca_quick_animator.py model.pth twitter.gif --size 512 --steps 150 --interval 2

# Mastodon optimized  
python nca_quick_animator.py model.pth mastodon.gif --size 256 --steps 120 --interval 3

# High quality showcase
python nca_quick_animator.py model.pth showcase.gif --size 512 --steps 300 --interval 1
```

## 🎪 Example Workflows

### For Social Media Posts

1. Train an interesting NCA model
2. Generate 3-4 different animations with different init types
3. Pick the most visually appealing one
4. Share with caption about the emergence process!

### For Technical Documentation

1. Use `--seed` for reproducible results
2. Enable step labels to show progression
3. Use higher resolution for clarity
4. Create multiple views of the same model

## 🐛 Troubleshooting

**"Model failed to load"**
- Check that the .pth file is from NCA_baseline.py
- Ensure the file isn't corrupted

**"Out of memory"**
- Reduce output size to 256px or smaller
- Use GPU if available
- Close other applications

**"GIF too large"**
- Increase frame interval (capture fewer frames)
- Reduce total steps
- Use smaller output size
- Enable optimization

**"Animation looks choppy"**
- Decrease frame interval (capture more frames)
- Increase total steps for longer evolution
- Adjust GIF speed (duration per frame)

## 🎊 Sharing Your Creations

When sharing on social media, consider adding:

- Brief explanation of Neural Cellular Automata
- Details about your model (training time, target image)
- Initialization method used
- Invite others to experiment with the code!

Example post:
```
🧬 Neural Cellular Automata in action! 

This AI learned to recreate a target image by having each pixel follow simple rules based on its neighbors. Starting from a single seed pixel, watch the pattern emerge through self-organization!

Model: 16 channels, trained for 500 epochs
Init: Circle seed, 150 evolution steps

#NeuralCellularAutomata #GenerativeAI #EmergentBehavior
```

Happy animating! 🎬✨
