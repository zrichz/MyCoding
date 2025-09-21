@echo off
REM Ensure we are in the script's directory
cd /d %~dp0

REM Run the script with the correct Python executable
c:\MyCoding\MyCoding\.venv\Scripts\python.exe CLIP_imager.py %*