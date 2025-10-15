@echo off
title TI Changer
echo Running TI Changer with virtual environment...
echo Virtual Environment: ..\.venv
echo Current Directory: %cd%
echo Python executable: ..\.venv\Scripts\python.exe
echo Script: TI_CHANGER.py
echo.

REM Ensure the console is properly initialized
cd /d "%~dp0"

REM Add a small delay to ensure console is ready
timeout /t 1 /nobreak >nul

echo Starting Python script...
REM Run the Python script
"..\.venv\Scripts\python.exe" "TI_CHANGER.py"

echo.
echo Script execution completed.
pause
