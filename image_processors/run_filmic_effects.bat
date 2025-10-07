@echo off
title Filmic Effects Processor
echo Starting Filmic Effects Processor...
echo.

:: Navigate to the MyCoding root directory to access .venv
cd /d "C:\MyCoding\MyCoding"

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv\Scripts\
    echo Please ensure the .venv is properly set up in the MyCoding directory.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Navigate to image_processors directory
cd image_processors

:: Check if script exists
if not exist "filmic_effects_processor.py" (
    echo ERROR: filmic_effects_processor.py not found in image_processors directory
    pause
    exit /b 1
)

:: Run the filmic effects processor
echo Running Filmic Effects Processor...
echo.
python filmic_effects_processor.py

:: Deactivate virtual environment
echo.
echo Deactivating virtual environment...
deactivate

echo.
echo Filmic Effects Processor finished.
pause