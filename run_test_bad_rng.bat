@echo off
REM Test Bad RNG - Simple Analysis without Graphics

echo ====================================================
echo         Bad RNG Test - Pattern Analysis
echo ====================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists in parent directory
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the .venv folder exists in the current directory.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

REM Run the test
echo.
echo Running Bad RNG analysis...
echo.
python test_bad_rng.py

REM Deactivate virtual environment when done
deactivate

echo.
pause