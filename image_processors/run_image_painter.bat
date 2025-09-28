@echo off
REM Image Painter Launcher for Windows
REM Uses .venv environment with PIL and numpy

cd /d "%~dp0"

echo Starting Image Painter GUI...

REM Activate virtual environment and run
call .venv\Scripts\activate.bat
python image_painter.py
call .venv\Scripts\deactivate.bat

echo Image Painter closed.
pause
