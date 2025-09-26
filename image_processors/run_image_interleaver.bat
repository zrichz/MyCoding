@echo off
REM Image Interleaver Launcher for Windows
REM PIL-only version - no external dependencies required

cd /d "%~dp0"

echo Starting Image Interleaver GUI...

python image_interleaver.py

echo Image Interleaver closed.
pause
