@echo off
REM Image Section Matcher Launcher for Windows
REM Uses .venv environment with PIL and numpy

cd /d "%~dp0"

echo Starting Image Section Matcher...

REM Use the local .venv in image_processors directory
if exist ".venv\Scripts\activate.bat" (
    echo Using local virtual environment...
    call .venv\Scripts\activate.bat
    python image_section_matcher.py
    call deactivate
) else (
    echo Virtual environment not found at .venv\
    echo Creating virtual environment and installing packages...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install numpy pillow
    python image_section_matcher.py
    call deactivate
)

echo Image Section Matcher closed.
pause
