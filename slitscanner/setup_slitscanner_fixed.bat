@echo off
title Setup Slitscanner Dependencies (Updated)
echo Setting up Slitscanner dependencies with compatibility fixes...
echo.

:: Navigate to the MyCoding root directory to access .venv  
cd /d "C:\MyCoding\MyCoding"

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv\Scripts\
    echo Please ensure the .venv is properly set up in the MyCoding directory.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install/upgrade slitscanner requirements with compatibility
echo.
echo Installing slitscanner requirements with compatibility fixes...
echo.
cd slitscanner

:: Install requirements with --upgrade to handle version conflicts
pip install --upgrade -r requirements.txt

:: If there are still conflicts, upgrade Pillow to latest compatible version
echo.
echo Checking for compatibility issues...
pip install --upgrade "Pillow>=10.1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo Slitscanner dependencies installed/updated successfully!
echo ========================================
echo.
echo You can now run the slitscanner using:
echo   run_slitscanner.bat
echo.

:: Show installed versions for verification
echo Installed versions:
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import customtkinter; print(f'CustomTkinter: {customtkinter.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

:: Deactivate virtual environment
deactivate

pause