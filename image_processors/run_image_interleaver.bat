@echo off
REM Image Interleaver Launcher for Windows
REM Uses .venv environment

cd /d "%~dp0"

echo Starting Image Interleaver GUI...

REM Activate virtual environment and run
call .venv\Scripts\activate.bat
python image_interleaver.py
call .venv\Scripts\deactivate.bat

echo Image Interleaver closed.
pause
