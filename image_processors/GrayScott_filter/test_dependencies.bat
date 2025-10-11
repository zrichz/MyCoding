@echo off
REM Test script for Gray-Scott Filter dependencies

echo Testing Gray-Scott Filter dependencies...
echo.

cd /d "%~dp0"

REM Test virtual environment
if not exist "..\.venv\Scripts\python.exe" (
    echo ERROR: Virtual environment Python not found
    exit /b 1
) else (
    echo ✓ Virtual environment found
)

REM Test Python script
if not exist "GrayScott_filter.py" (
    echo ERROR: GrayScott_filter.py not found
    exit /b 1
) else (
    echo ✓ GrayScott_filter.py found
)

REM Test dependencies
echo Testing Python dependencies...
"..\.venv\Scripts\python.exe" -c "import tkinter; import numpy; import scipy; from PIL import Image; from GrayScott_filter import GrayScottFilterApp; print('✓ All dependencies OK')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Gray-Scott Filter is ready to run!
    echo You can now use run_grayscott_filter.bat
) else (
    echo.
    echo ERROR: Some dependencies are missing
    echo Try running: pip install pillow scipy numpy
)

echo.
pause