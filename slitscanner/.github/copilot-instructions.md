<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Slitscanner Project Instructions

This is a Python video processing application that creates artistic images using the slitscanning technique.

## Project Overview
- **Purpose**: Extract vertical pixel columns from video frames and combine them horizontally
- **GUI Framework**: CustomTkinter for modern dark-themed interface
- **Video Processing**: OpenCV for video file handling and frame extraction
- **Image Processing**: PIL/Pillow and NumPy for image manipulation

## Key Features
- Video file selection with common format support
- Frame sampling options (every frame, every 2 frames, every N frames)
- Real-time progress tracking with progress bar
- Image preview with automatic scaling
- Save functionality for processed images
- Maximum output width limit of 3000 pixels

## Technical Details
- **Center Column Extraction**: Takes pixel column from frame_width // 2
- **Color Space**: Converts BGR (OpenCV) to RGB (PIL) for proper display
- **Threading**: Uses background processing to prevent GUI freezing
- **Memory Management**: Processes frames sequentially to handle large videos

## Code Style Guidelines
- Use descriptive variable names and comprehensive docstrings
- Follow PEP 8 formatting standards
- Implement proper error handling with user-friendly messages
- Use threading for long-running operations
- Maintain separation of concerns (GUI, processing, file handling)
