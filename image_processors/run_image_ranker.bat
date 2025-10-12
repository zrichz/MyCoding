@echo off
REM Image Ranker Launcher for Windows
REM This script launches the image ranker with optimized 2560x1440 layout

echo.
echo ================================================
echo    Image Ranker - Tournament Style Ranking
echo ================================================
echo.
echo Features:
echo - Optimized for 2560x1440 resolution
echo - Side-by-side image comparison
echo - Tournament-style pairwise ranking
echo - Automatic file renaming with rank prefixes
echo - No popup interruptions
echo.

REM Change to the image_processors directory
cd /d "%~dp0"

REM Check if Python virtual environment exists
if exist "..\venv_mycoding\Scripts\python.exe" (
    echo Using virtual environment...
    "..\venv_mycoding\Scripts\python.exe" image_ranker.py
) else if exist "..\.venv\Scripts\python.exe" (
    echo Using .venv virtual environment...
    "..\.venv\Scripts\python.exe" image_ranker.py
) else (
    echo Using system Python...
    python image_ranker.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to launch Image Ranker
    echo Please check that Python and required packages are installed
    echo.
    pause
)