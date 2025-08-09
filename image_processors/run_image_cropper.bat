@echo off
REM Interactive Image Cropper - Easy Launch Script for Windows
REM This batch file activates the virtual environment and runs the cropper application

echo ====================================================
echo        Interactive Image Cropper Launcher
echo ====================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the .venv folder exists in the current directory.
    echo.
    echo To create a virtual environment, run:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

REM Check if activation was successful
if "%VIRTUAL_ENV%" == "" (
    echo ERROR: Failed to activate virtual environment!
    echo.
    pause
    exit /b 1
)

echo Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Check if Pillow is installed
echo Checking for required dependencies...
python -c "import PIL; print('Pillow version:', PIL.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Required dependencies not found! Installing...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies!
        echo.
        pause
        exit /b 1
    )
)

REM Run the image cropper
echo.
echo Starting Interactive Image Cropper...
echo Press Ctrl+C to exit
echo.
python interactive_image_cropper.py

REM Deactivate virtual environment when done
deactivate

echo.
echo Interactive Image Cropper has closed.
pause
