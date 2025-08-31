@echo off
echo Neural Cellular Automata - CUDA Launch with Device Check
echo =========================================================
echo.

REM Test device consistency first
echo Step 1: Testing device consistency...
C:/Users/richm/MyCoding/MyCoding/.venv/Scripts/python.exe test_device_consistency.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Device consistency test failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ✅ Device consistency test passed!
echo.

REM Run CUDA performance test
echo Step 2: Running CUDA performance test...
C:/Users/richm/MyCoding/MyCoding/.venv/Scripts/python.exe test_cuda.py

echo.
echo Step 3: Launching Neural Cellular Automata GUI...
echo.

REM Launch the main application
C:/Users/richm/MyCoding/MyCoding/.venv/Scripts/python.exe NCA_baseline.py

echo.
echo Neural Cellular Automata has closed.
pause
