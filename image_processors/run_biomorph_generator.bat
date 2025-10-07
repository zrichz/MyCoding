@echo off
REM Biomorph Generator Launcher for Windows
REM Uses .venv environment with PIL and numpy

cd /d "%~dp0"

echo Starting Biomorph Generator...

REM Use the local .venv in image_processors directory
if exist ".venv\Scripts\activate.bat" (
    echo Using local virtual environment...
    call .venv\Scripts\activate.bat
    python biomorph_generator.py
    call deactivate
) else (
    echo Virtual environment not found at .venv\
    echo Creating virtual environment and installing packages...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install numpy pillow
    python biomorph_generator.py
    call deactivate
)

echo Biomorph Generator closed.
pause
