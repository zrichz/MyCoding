@echo off
REM Seam Carving Width Adjuster - Easy Launch Script
REM This batch file activates the virtual environment and runs the GUI application
REM Supports both width reduction (50-99%) and expansion (100-150%)

echo Starting Seam Carving Width Adjuster...
echo.

REM Check if virtual environment exists
if not exist "..\\.venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the .venv folder exists in the parent directory.
    echo You may need to create it first by running:
    echo   python -m venv ..\.venv
    echo   ..\.venv\Scripts\activate.bat
    echo   pip install -r seam_carving_requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check if the main script exists
if not exist "seam_carving_width_reducer.py" (
    echo ERROR: seam_carving_width_reducer.py not found!
    echo Please make sure you're running this batch file from the correct directory.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment and run the application
echo Activating virtual environment...
call ..\\.venv\Scripts\activate.bat

echo Running Seam Carving Width Adjuster GUI...
echo.
python seam_carving_width_reducer.py

REM Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    pause
)

echo.
echo Application closed.
pause
