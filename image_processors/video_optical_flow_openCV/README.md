# Video Optical Flow Visualizer

A powerful GUI tool for visualizing optical flow in videos using OpenCV's dense optical flow algorithms.

## Features

- **User-friendly GUI**: Simple tkinter interface for video file selection
- **Dense Optical Flow**: Uses OpenCV's Farneback algorithm for accurate motion detection
- **Customizable Visualization**: 
  - Adjustable flow line density (5-30)
  - Variable line thickness (1-5 pixels)
  - Motion threshold filtering (0.1-5.0)
- **Color-coded Motion**: Flow directions are color-coded for easy interpretation:
  - Red: Rightward motion
  - Yellow: Up-right motion
  - Green: Upward motion
  - Cyan: Up-left motion
  - Blue: Leftward motion
  - Magenta: Down-left motion
- **Real-time Progress**: Progress bar and status updates during processing
- **Multiple Video Formats**: Supports MP4, AVI, MOV, MKV, WMV, FLV, WEBM

## How to Use

### Windows
1. Run `run_optical_flow.bat`
2. Click "Browse" to select your input video file
3. Adjust the optical flow parameters as desired:
   - **Flow Line Density**: Controls how many flow vectors are displayed
   - **Flow Line Thickness**: Thickness of the flow arrows
   - **Motion Threshold**: Minimum motion magnitude to display (filters out noise)
4. Click "Process Video" to start the analysis
5. The output video will be saved with "_optical_flow" suffix in the same directory

### Linux/Mac
1. Make the script executable: `chmod +x run_optical_flow.sh`
2. Run `./run_optical_flow.sh`
3. Follow the same steps as Windows

## Output

The tool creates a new video file showing:
- Original video frames as the background
- Colored arrows indicating motion direction and magnitude
- Semi-transparent overlay for clear visualization

## Technical Details

- **Algorithm**: Gunnar Farneback's dense optical flow
- **Processing**: Frame-by-frame analysis with configurable parameters
- **Visualization**: HSV color space mapping for flow directions
- **Performance**: Optimized for real-time processing with progress feedback

## Dependencies

- OpenCV (opencv-python)
- NumPy
- tkinter (usually included with Python)
- PIL/Pillow

All dependencies are managed through the shared virtual environment in the parent directory.

## Tips for Best Results

1. **Higher density** (20-30): Better for detailed motion analysis but slower processing
2. **Lower density** (5-15): Faster processing, good for general motion overview
3. **Higher threshold** (2.0-5.0): Filters out camera shake and noise
4. **Lower threshold** (0.1-1.0): Shows subtle motions but may include noise
5. **Thicker lines** (3-5): Better visibility on high-resolution videos
6. **Thinner lines** (1-2): Less cluttered appearance for dense motion

## Troubleshooting

- If the video doesn't open, ensure it's a supported format
- For large videos, processing may take several minutes
- If arrows appear too dense, increase the Flow Line Density value
- If no motion is detected, try lowering the Motion Threshold
