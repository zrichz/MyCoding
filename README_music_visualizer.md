# 🎵 Music Visualizer

A non-realtime music visualization tool that creates spectrogram videos from audio files with transient detection.

## Features

- **Audio Analysis**: Analyzes WAV, MP3, FLAC, and M4A files using librosa
- **Spectrogram Generation**: Creates detailed frequency-time visualizations
- **Transient Detection**: Identifies sudden changes in audio energy (beats, hits, etc.)
- **Video Output**: Generates 512x512 60fps MP4 videos
- **Interactive GUI**: Simple tkinter interface for easy use
- **Real-time Preview**: Shows spectrogram and transient analysis before video generation

## Installation & Usage

### Quick Start
```bash
# Make launcher executable (Linux/Mac)
chmod +x run_music_visualizer.sh

# Run the application (Linux/Mac)
./run_music_visualizer.sh

# Or on Windows
run_music_visualizer.bat
```

### Manual Installation
If you prefer to install manually:

```bash
# Activate your existing virtual environment
source .venv/bin/activate

# Install requirements
pip install -r music_visualizer_requirements.txt

# Run the application
python music_visualizer.py
```

### Testing
Run the test script to verify everything works:
```bash
python test_music_visualizer.py
```

## How It Works

1. **Load Audio**: Browse and select your audio file (WAV/MP3/FLAC/M4A)
2. **Analyze**: Click "Analyze" to process the audio
   - Computes spectrogram using Short-Time Fourier Transform (STFT)
   - Detects transients using librosa's onset detection
   - Shows preview with transients marked in red
3. **Generate Video**: Configure output settings and click "Generate Video"
   - Creates animated spectrogram visualization
   - Highlights transients as they occur
   - Exports as MP4 video file

## Technical Details

### Audio Processing
- **Sample Rate**: Preserves original or defaults to librosa's default
- **STFT Parameters**: 
  - FFT size: 2048 samples
  - Hop length: 512 samples
  - Window: Hann (librosa default)
- **Frequency Range**: 0 Hz to Nyquist frequency
- **Time Resolution**: ~23ms (at 22kHz sample rate)

### Transient Detection
- Uses librosa's onset detection algorithm
- Based on spectral flux and energy changes
- Configurable sensitivity (currently optimized for music)

### Video Generation
- **Resolution**: 256x256, 512x512, 720x720, or 1024x1024 pixels
- **Frame Rate**: 24, 30, or 60 FPS
- **Codec**: MP4V (widely compatible)
- **Colormap**: Plasma (vibrant colors for better visibility)

### Visualization Features
- **Dynamic Spectrogram**: Frequency content evolves over time
- **Transient Highlighting**: Red vertical lines mark detected transients
- **Current Time Indicator**: Yellow line shows current playback position
- **Frequency Scale**: Logarithmic frequency axis (Hz)
- **Amplitude Scale**: dB scale (-80 dB to 0 dB)

## File Formats

### Supported Input Formats
- **WAV**: Uncompressed audio (best quality)
- **MP3**: Compressed audio (most common)
- **FLAC**: Lossless compressed audio
- **M4A**: Apple's audio format

### Output Format
- **MP4**: H.264 compatible video file
- **Naming**: `{filename}_spectrogram_{size}x{size}_{fps}fps_{timestamp}.mp4`

## Dependencies

### Core Libraries
- **librosa**: Audio analysis and music information retrieval
- **numpy**: Numerical computing for signal processing
- **matplotlib**: Plotting and visualization
- **opencv-python**: Video generation and computer vision
- **scipy**: Scientific computing (used by librosa)
- **soundfile**: Audio file I/O (used by librosa)

### Optional
- **ffmpeg-python**: Enhanced video processing capabilities
- **pydub**: Additional audio format support

## Project Structure

```
MyCoding/
├── music_visualizer.py              # Main GUI application
├── run_music_visualizer.sh          # Linux/Mac launcher
├── run_music_visualizer.bat         # Windows launcher
├── test_music_visualizer.py         # Test script
├── music_visualizer_requirements.txt # Python dependencies
├── README_music_visualizer.md       # This file
└── .venv/                           # Virtual environment (existing)
```

## Examples

### Typical Workflow
1. Run launcher: `./run_music_visualizer.sh`
2. Load audio file: "Browse" → select your music file
3. Analyze: Click "Analyze" button
4. Review: Check spectrogram preview and transient detection
5. Configure: Set output size (512x512) and FPS (60)
6. Generate: Click "Generate Video" and wait for completion

### Expected Results
- **Electronic Music**: Clear transients on beats, rich harmonic content
- **Classical Music**: Complex frequency patterns, subtle transients
- **Rock/Pop**: Strong transients on drums, clear bass and vocal tracks
- **Ambient**: Gradual changes, fewer sharp transients

## Troubleshooting

### Common Issues

1. **"No module named 'librosa'"**
   - Run the launcher script to auto-install dependencies
   - Or manually: `pip install librosa`

2. **Audio file not loading**
   - Check file format (WAV, MP3, FLAC, M4A supported)
   - Try converting to WAV if other formats fail
   - Ensure file isn't corrupted

3. **Video generation fails**
   - Check available disk space
   - Ensure OpenCV is properly installed
   - Try smaller output resolution first

4. **GUI not appearing**
   - Ensure tkinter is available (usually built-in with Python)
   - Check virtual environment activation

### Performance Tips
- **Large Files**: Consider shorter clips for testing (full songs work but take longer)
- **High Resolution**: 1024x1024 videos take significantly longer to generate
- **Frame Rate**: 60 FPS provides smooth motion but larger file sizes

## Future Enhancements

Possible improvements for future versions:
- Real-time audio playback during analysis
- Multiple visualization styles (waveform, 3D spectrogram, etc.)
- Batch processing for multiple files
- Custom color schemes and visual effects
- Beat tracking and rhythm visualization
- Integration with music metadata (tempo, key, etc.)

## License

This project is part of the MyCoding repository. Use freely for personal and educational purposes.