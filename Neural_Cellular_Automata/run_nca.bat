@echo off
echo Starting Neural Cellular Automata...
echo.

REM Check if we're in the correct directory
if not exist "NCA_baseline.py" (
    echo Error: NCA_baseline.py not found in current directory
    echo Please run this script from the Neural_Cellular_Automata folder
    pause
    exit /b 1
)

REM Navigate to parent directory to find venv
cd ..

REM Check for various virtual environment names
if exist "venv_mycoding\Scripts\activate.bat" (
    echo Activating virtual environment: venv_mycoding
    call venv_mycoding\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment: venv
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment: .venv
    call .venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found
    echo Continuing with system Python...
)

echo.
echo Running Neural Cellular Automata GUI...
echo Press Ctrl+C to stop the application
echo.

REM Run the script from the Neural_Cellular_Automata directory
python Neural_Cellular_Automata\NCA_baseline.py

echo.
echo Neural Cellular Automata has closed.
pause
