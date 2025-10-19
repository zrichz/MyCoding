@echo off
title MyCoding Virtual Environment
echo Activating MyCoding virtual environment...
echo Location: C:\MyCoding\MyCoding\.venv
echo.

cd /d "C:\MyCoding\MyCoding"
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv
    echo Please make sure the virtual environment is properly set up.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo *******************************************************
echo *******   Virtual environment activated!   ************
echo Python executable: %cd%\.venv\Scripts\python.exe
echo Current directory: %cd%
echo.
echo *******************************************************
echo *****   You can now run any Python scripts in this environment.
echo *****  To deactivate, simply type: deactivate
echo *******************************************************
echo.

cmd /k
