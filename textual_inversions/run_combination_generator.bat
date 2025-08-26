@echo off
title Combination Generator
echo Running Combination Generator with virtual environment...
echo Virtual Environment: C:\MyPythonCoding\MyCoding\image_processors\.venv
echo.

REM Ensure the console is properly initialized
cd /d "%~dp0"

REM Add a small delay to ensure console is ready
timeout /t 1 /nobreak >nul

REM Run the Python script
"C:\MyPythonCoding\MyCoding\image_processors\.venv\Scripts\python.exe" "combination_generator.py"

echo.
echo Script execution completed.
pause
