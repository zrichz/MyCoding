@echo off
REM Image Pair Combiner Launcher for Windows
REM PIL-only version - no external dependencies required

cd /d "%~dp0"

echo Starting Image Pair Combiner GUI...

python image_pair_combiner.py

echo Image Pair Combiner closed.
pause
