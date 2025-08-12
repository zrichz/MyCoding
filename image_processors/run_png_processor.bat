@echo off
echo PNG to JPG Batch Processor
echo ===========================================
echo This script processes all PNG files through the following pipeline:
echo 1. Scales to max 1080x1440 preserving aspect ratio (with blur fill)
echo 2. Rotates 90 degrees counter-clockwise
echo 3. Stretches to 1920x1080
echo 4. Saves as high-quality JPEG files
echo.
echo Choose mode:
echo 1. GUI Mode (recommended)
echo 2. Command-line mode (current directory)
echo.
set /p choice=Enter choice (1 or 2): 

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: .venv folder not found. Please run from the image_processors directory.
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

if "%choice%"=="1" (
    echo Launching GUI...
    python png_to_jpg_processor.py --gui
) else if "%choice%"=="2" (
    echo Processing PNG files in current directory...
    python png_to_jpg_processor.py
) else (
    echo Invalid choice. Launching GUI by default...
    python png_to_jpg_processor.py --gui
)

echo.
echo Processing complete!
pause
