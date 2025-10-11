@echo off
echo.
echo ================================================
echo    Gray-Scott Filter with Morphological Ops
echo ================================================
echo.
echo Features:
echo - Gray-Scott iterative filtering
echo - Image binarization (50%% threshold)
echo - Morphological operations (erosion/dilation)
echo - Full-size image display with scrollbars
echo - 1070px height optimized for 1920x1080 screens
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "..\.venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at ..\.venv\Scripts\
    echo Please ensure the .venv directory exists in the parent folder.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "..\.venv\Scripts\activate.bat"

REM Check if Python script exists
if not exist "GrayScott_filter.py" (
    echo Error: GrayScott_filter.py not found in current directory.
    pause
    exit /b 1
)

echo Starting Gray-Scott Filter GUI...
python GrayScott_filter.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to run Gray-Scott Filter
    echo This might be due to missing dependencies.
    echo Try running: pip install pillow scipy numpy
    echo.
)

pause
