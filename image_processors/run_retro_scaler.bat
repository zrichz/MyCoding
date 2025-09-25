@echo off
echo Starting Retro Scaler - Intelligent Montage Creator...
echo.

cd /d "%~dp0"

rem Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.7+ and try again.
    pause
    exit /b 1
)

rem Check if virtual environment exists
if exist "..\..\.venv\Scripts\python.exe" (
    echo Using virtual environment...
    "..\..\.venv\Scripts\python.exe" retro_scaler_720x1600.py
) else (
    echo Using system Python...
    python retro_scaler_720x1600.py
)

if errorlevel 1 (
    echo.
    echo Script encountered an error.
    pause
)