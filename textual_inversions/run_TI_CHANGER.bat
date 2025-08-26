@echo off
title TI Changer Multiple
echo Running TI Changer Multiple with virtual environment...
echo Virtual Environment: C:\MyPythonCoding\MyCoding\image_processors\.venv
echo.

REM Ensure the console is properly initialized
cd /d "%~dp0"

REM Add a small delay to ensure console is ready
timeout /t 1 /nobreak >nul

REM Run the Python script
"C:\MyPythonCoding\MyCoding\image_processors\.venv\Scripts\python.exe" "TI_CHANGER_MULTIPLE_2024_10_22.py"

echo.
echo Script execution completed.
pause
