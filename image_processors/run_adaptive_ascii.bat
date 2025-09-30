@echo off
rem Adaptive ASCII Art Generator Launcher
rem Uses existing virtual environment to run the adaptive ASCII art tool

cd /d "%~dp0"

rem Check if virtual environment exists
if exist "..\..\.venv\Scripts\python.exe" (
    echo Using existing virtual environment...
    "..\..\.venv\Scripts\python.exe" adaptive_ascii_art.py
) else if exist "..\.venv\Scripts\python.exe" (
    echo Using existing virtual environment...
    "..\.venv\Scripts\python.exe" adaptive_ascii_art.py
) else (
    echo Virtual environment not found. Please ensure .venv exists.
    echo Trying system Python...
    python adaptive_ascii_art.py
)

if errorlevel 1 (
    echo.
    echo Error running the application.
    echo Make sure the virtual environment is set up and contains the required packages:
    echo - numpy
    echo - pillow
    echo.
    pause
)