@echo off
REM Canny Batch Processor Launcher for Windows
REM Uses .venv environment with OpenCV, PIL and numpy

cd /d "%~dp0"

echo Starting Canny Batch Processor...

REM Use the local .venv in image_processors directory
if exist ".venv\Scripts\activate.bat" (
    echo Using local virtual environment...
    call .venv\Scripts\activate.bat
    
    REM Install opencv-python if not already installed
    python -c "import cv2" >nul 2>&1 || (
        echo Installing opencv-python in virtual environment...
        pip install opencv-python
    )
    
    python canny_batch_processor.py
    call deactivate
) else (
    echo Virtual environment not found at .venv\
    echo Creating virtual environment and installing packages...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install opencv-python numpy pillow
    python canny_batch_processor.py
    call deactivate
)

echo Canny Batch Processor closed.
pause
