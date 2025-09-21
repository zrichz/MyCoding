@echo off
echo Launching Neural Cellular Automata with CUDA-enabled PyTorch...
echo.

REM First try the .venv in current workspace (newly upgraded)
if exist "..\\.venv\\Scripts\\python.exe" (
    echo Using CUDA-enabled environment: .venv
    call "..\\.venv\\Scripts\\python.exe" NCA_baseline.py
) else if exist "C:\\MyPythonCoding\\MyCoding\\venv_mycoding\\Scripts\\python.exe" (
    echo Using fallback environment: venv_mycoding
    call "C:\\MyPythonCoding\\MyCoding\\venv_mycoding\\Scripts\\python.exe" NCA_baseline.py
) else (
    echo Error: No Python environment found
    echo Please check your virtual environment setup
    pause
    exit /b 1
)

echo.
echo Application closed.
pause
