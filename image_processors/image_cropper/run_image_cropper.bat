@echo off
REM Interactive Image Cropper - Easy Launch Script
REM This batch file activates the virtual environment and runs the cropper application

echo Starting Interactive Image Cropper...
echo.

REM Check if virtual environment exists
if not exist "..\\.venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the .venv folder exists in the parent directory.
    echo You may need to create it first by running:
    echo   python -m venv ..\.venv
    echo   ..\.venv\Scripts\activate.bat
    echo   pip install Pillow
    echo.
    pause
    exit /b 1
)

REM Check if the main script exists
if not exist "interactive_image_cropper.py" (
    echo ERROR: interactive_image_cropper.py not found!
    echo Please make sure you're running this batch file from the correct directory.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment and run the application
echo Activating virtual environment...
call ..\\.venv\Scripts\activate.bat

echo Running Interactive Image Cropper GUI...
echo.
python interactive_image_cropper.py

REM Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    pause
)

echo.
echo Application closed.
pause
