@echo off
REM Image Numberer 1440x3200 - Easy Launch Script for Windows
REM This batch file activates the virtual environment and runs the image numberer application

echo ====================================================
echo        Image Numberer 1440x3200 Launcher
echo ====================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists in parent directory
if not exist "..\.venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the .venv folder exists in the parent directory.
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
call "..\.venv\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    echo.
    pause
    exit /b 1
)

echo Virtual environment activated successfully.
echo.

REM Check if the Python script exists
if not exist "image_numberer_1440x3200.py" (
    echo ERROR: image_numberer_1440x3200.py not found!
    echo Please make sure the script is in the current directory.
    echo.
    pause
    exit /b 1
)

REM Run the image numberer application
echo Starting Image Numberer 1440x3200...
echo.
echo This application will:
echo - Process 1440x3200 pixel images only
echo - Add sequential numbers (0001, 0002, etc.) to each corner
echo - Save as highest quality JPEG
echo - Skip images that are not exactly 1440x3200 pixels
echo.

python image_numberer_1440x3200.py

REM Check if the script ran successfully
if errorlevel 1 (
    echo.
    echo ERROR: The image numberer application encountered an error!
    echo.
) else (
    echo.
    echo Image numberer application completed successfully.
)

echo.
echo Press any key to exit...
pause >nul