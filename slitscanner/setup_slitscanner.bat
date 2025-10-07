@echo off
title Setup Slitscanner Dependencies
echo Setting up Slitscanner dependencies...
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

:: Install slitscanner requirements
echo.
echo Installing slitscanner requirements...
echo.
cd slitscanner
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo Slitscanner dependencies installed successfully!
echo ========================================
echo.
echo You can now run the slitscanner using:
echo   run_slitscanner.bat
echo.

:: Deactivate virtual environment
deactivate

pause