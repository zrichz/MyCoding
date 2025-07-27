# Slitscanner - Video to Image Processor

🎬 **Slitscanner** is a creative video processing application that transforms videos into unique artistic images using the slitscanning technique.

## What is Slitscanning?

Slitscanning is a creative video processing technique that:
- Extracts a single vertical pixel column from the center of each video frame
- Arranges these columns side-by-side to create a single composite image
- Creates mesmerizing visual effects that show motion and time in a unique way

## Features

✨ **Core Functionality**
- Process any common video format (MP4, AVI, MOV, MKV, WMV, FLV, WebM)
- Extract center pixel columns from video frames
- Combine columns into a single horizontal image
- Automatic stopping at 3000 pixels width to prevent memory issues

🎛️ **Flexible Processing Options**
- **Every Frame**: Use all frames for maximum detail
- **Every 2 Frames**: Skip alternate frames for faster processing
- **Every N Frames**: Custom frame sampling (user-defined interval)

🖥️ **Modern GUI Interface**
- Dark-themed CustomTkinter interface
- Real-time progress tracking with progress bar
- Image preview with automatic scaling
- Easy video file selection
- One-click save functionality

## Installation

1. **Clone or download** this project to your local machine

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## Required Dependencies

- `opencv-python==4.8.1.78` - Video file processing and frame extraction
- `customtkinter==5.2.0` - Modern GUI framework
- `Pillow==10.0.1` - Image processing and display
- `numpy==1.24.3` - Numerical operations for image arrays

## How to Use

1. **Launch the application** by running `python main.py`

2. **Select a video file** using the "Select Video" button

3. **Choose frame sampling**:
   - **Every Frame**: Processes all frames (highest quality, longer processing)
   - **Every 2 Frames**: Skips every other frame (balanced quality/speed)
   - **Every N Frames**: Custom interval (enter your preferred number)

4. **Click "Process Video"** to start the slitscanning process

5. **Monitor progress** with the real-time progress bar and status updates

6. **Preview the result** in the scrollable image display area

7. **Save your creation** using the "Save Image" button (PNG or JPEG format)

## Technical Details

### Processing Logic
- Extracts the center vertical column from each frame (`frame_width // 2`)
- Builds a composite image with dimensions: `(frame_height, processed_frames, 3)`
- Automatically limits output width to 3000 pixels to prevent memory issues
- Converts color space from BGR (OpenCV) to RGB (PIL) for proper display

### Performance Features
- **Background processing**: Uses threading to prevent GUI freezing
- **Memory efficient**: Processes frames sequentially rather than loading all into memory
- **Progress tracking**: Real-time updates on processing status
- **Automatic scaling**: Displays large images at manageable sizes

### Supported Video Formats
- MP4, AVI, MOV, MKV, WMV, FLV, WebM
- Any format supported by OpenCV

## Example Output

The slitscanner creates unique images where:
- **Stationary objects** appear as vertical lines
- **Moving objects** create interesting patterns and streaks
- **Camera movement** produces flowing, wave-like effects
- **Time progression** is visible across the horizontal axis

## Troubleshooting

**Video won't load?**
- Ensure the video file isn't corrupted
- Try a different video format
- Check that the file path doesn't contain special characters

**Processing is slow?**
- Use "Every 2 Frames" or "Every N Frames" for faster processing
- Consider using shorter videos for testing
- Close other applications to free up system resources

**Out of memory errors?**
- The application automatically stops at 3000 pixels width
- Use higher frame sampling (Every N Frames with larger N values)
- Process shorter video segments

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for:
- Performance improvements
- Additional video format support
- New slitscanning algorithms
- UI/UX enhancements
