@echo off
title Combination Generator
echo Running Combination Generator...
echo Using system Python (no virtual environment needed)
echo.

REM Ensure the console is properly initialized
cd /d "%~dp0"

REM Add a small delay to ensure console is ready
timeout /t 1 /nobreak >nul

REM Run the Python script using system Python
python "combination_generator.py"

echo.
echo Script execution completed.
pause
