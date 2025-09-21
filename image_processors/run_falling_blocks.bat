@echo off
REM Falling Blocks Viewer Launcher Script (Windows)
REM Activates virtual environment and runs the falling blocks image viewer

echo ====================================================
echo        Falling Blocks Image Viewer Launcher
echo ====================================================
echo.

REM Get script directory and change to it
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found at .venv\
    echo Please run setup first or create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated: %cd%\.venv
echo.

echo Checking for required dependencies...

REM Check for pygame
python -c "import pygame; print('✓ pygame available')" >nul 2>&1
if errorlevel 1 (
    echo ❌ pygame not found. Installing...
    pip install pygame==2.6.0
    if errorlevel 1 (
        echo ❌ Failed to install pygame
        pause
        exit /b 1
    )
    echo ✓ pygame installed successfully
)

REM Check for other dependencies
python -c "import PIL; print('✓ Pillow available')" >nul 2>&1
if errorlevel 1 (
    echo ❌ Pillow not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✓ All dependencies available
echo.
echo Starting Falling Blocks Image Viewer...
echo Press Ctrl+C to exit
echo.

REM Run the falling blocks viewer
python falling_blocks_viewer.py

echo.
echo Falling Blocks Image Viewer closed.
pause
