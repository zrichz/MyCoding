@echo off
REM Image Pair Combiner Launcher for Windows
REM Uses .venv environment

cd /d "%~dp0"

echo Starting Image Pair Combiner GUI...

REM Activate virtual environment and run
call .venv\Scripts\activate.bat
python image_pair_combiner.py
call .venv\Scripts\deactivate.bat

echo Image Pair Combiner closed.
pause
