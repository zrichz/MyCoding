@echo off
REM Bad RNG 3D Polar Visualization - Easy Launch Script for Windows
REM This batch file activates the virtual environment and runs the bad RNG visualization

echo ====================================================
echo    RNG Comparison: Bad LCG vs Xorshift Launcher
echo ====================================================
echo.
echo Comparing BAD vs GOOD random number generators:
echo   Red:  x[i+1] = (5*x[i] + 1) mod 256  [TERRIBLE]
echo   Blue: x ^= x^(x<<13)^(x>>17)^(x<<5)  [EXCELLENT]
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists in current directory
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure the .venv folder exists in the MyCoding directory.
    echo.
    echo To create a virtual environment, run:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   pip install matplotlib numpy scipy
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

REM Check if activation was successful
if "%VIRTUAL_ENV%" == "" (
    echo ERROR: Failed to activate virtual environment!
    echo.
    pause
    exit /b 1
)

echo Virtual environment activated: %VIRTUAL_ENV%
echo.

REM Check if required dependencies are installed
echo Checking for required dependencies...
python -c "import matplotlib, numpy; print('All dependencies available')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Required dependencies not found! Installing...
    pip install matplotlib numpy scipy
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies!
        echo.
        pause
        exit /b 1
    )
)

REM Run the RNG comparison visualization
echo.
echo Starting RNG Comparison Visualization...
echo Press Ctrl+C to exit
echo.
python bad_rng_3d_polar_visualization.py

REM Deactivate virtual environment when done
deactivate

echo.
echo RNG comparison visualization has closed.
pause