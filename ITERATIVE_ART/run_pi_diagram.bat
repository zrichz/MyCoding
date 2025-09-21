@echo off
REM Batch file to run pi_diagram.py
echo Running Pi Diagram Script...

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python script using the virtual environment
call "%~dp0..\image_processors\.venv\Scripts\python.exe" pi_diagram.py

REM Pause to see any output
pause
