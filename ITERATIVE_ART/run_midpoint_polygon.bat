@echo off
REM Batch file to run iterative_midpoint_polygon.py
echo Running Iterative Midpoint Polygon Generator...

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python script using the virtual environment
call "%~dp0..\image_processors\.venv\Scripts\python.exe" iterative_midpoint_polygon.py

REM Pause to see any output
pause
