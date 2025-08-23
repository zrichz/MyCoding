@echo off
echo PNG to JPG Batch Processor - GUI Version
echo ===========================================
echo This script processes PNG files through the following pipeline:
echo.
echo For PORTRAIT/SQUARE images:
echo 1. Scales to max 1080x1440 preserving aspect ratio (with blur fill)
echo 2. Rotates 90 degrees counter-clockwise
echo 3. Stretches to 1920x1080
echo 4. Saves as high-quality JPEG files
echo.
echo For LANDSCAPE images:
echo 1. Scales to max 1440x1080 preserving aspect ratio (with blur fill)
echo 2. Skips rotation step (already in correct orientation)
echo 3. Stretches to 1920x1080
echo 4. Saves as high-quality JPEG files
echo.
echo Launching GUI interface...
echo.

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: .venv folder not found. Please run from the image_processors directory.
    pause
    exit /b 1
)

REM Activate virtual environment and launch GUI
call .venv\Scripts\activate.bat
python png_to_jpg_processor.py

echo.
echo Processing complete!
pause
