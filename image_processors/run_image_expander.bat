@echo off
REM Image Expander 720x1600 - Easy Launch Script for Windows
REM This batch file activates the virtual environment and runs the image expander application

echo ====================================================
echo        Image Expander 720x1600 Launcher
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

REM Check if activation was successful
if "%VIRTUAL_ENV%" == "" (
    echo ERROR: Failed to activate virtual environment!
    echo.
    pause
    exit /b 1
)

echo Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Check if required dependencies are installed
echo Checking for required dependencies...
python -c "import PIL, numpy, scipy; print('All dependencies available')" 2>nul
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

REM Run the image expander
echo.
echo Starting Image Expander 720x1600...
echo Press Ctrl+C to exit
echo.
python image_expander_720x1600.py

REM Deactivate virtual environment when done
deactivate

echo.
echo Image Expander 720x1600 has closed.
pause
