@echo off
REM Image Quad Combiner Launcher for Windows
REM Creates horizontal combinations of 4 images scaled to fit 2560x1440

cd /d "%~dp0"

echo Starting Image Quad Combiner GUI...
echo.
echo Process: 4x 720x1600 images -> 2880x1600 combo -> 2560x1422 scaled output
echo.

REM Activate virtual environment and run
call .venv\Scripts\activate.bat
python image_quad_combiner.py
call .venv\Scripts\deactivate.bat

echo Image Quad Combiner closed.
pause
