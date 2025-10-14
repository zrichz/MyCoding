@echo off
REM Run Iterative Midpoint Polygon Generator with image_processors virtual environment

REM Get the directory of this script
set SCRIPT_DIR=%~dp0

REM Path to the image_processors virtual environment
set VENV_PATH=%SCRIPT_DIR%..\..\image_processors\.venv

REM Check if virtual environment exists
if not exist "%VENV_PATH%" (
    echo Error: Virtual environment not found at %VENV_PATH%
    echo Please create the virtual environment first or install dependencies globally.
    pause
    exit /b 1
)

REM Activate virtual environment and run the script
echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

echo Running Iterative Midpoint Polygon Generator...
python "%SCRIPT_DIR%..\iterative_midpoint_polygon.py"

echo Done.
pause
