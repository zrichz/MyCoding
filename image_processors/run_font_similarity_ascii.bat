@echo off
REM Font Similarity ASCII Art Launcher for Windows
REM Uses .venv environment with PIL and numpy

cd /d "%~dp0"

echo Starting Font Similarity ASCII Art Generator...

REM Use the local .venv in image_processors directory
if exist ".venv\Scripts\activate.bat" (
    echo Using local virtual environment...
    call .venv\Scripts\activate.bat
    python font_similarity_ascii.py
    call deactivate
) else (
    echo Virtual environment not found at .venv\
    echo Creating virtual environment and installing packages...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install numpy pillow
    python font_similarity_ascii.py
    call deactivate
)

echo Font Similarity ASCII Art closed.
pause
