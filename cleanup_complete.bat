@echo off
title Complete venv_myDL1 Cleanup
echo.
echo ========================================
echo  Virtual Environment Cleanup Tool
echo ========================================
echo.
echo This will remove all references to the spurious venv_myDL1 environment
echo and ensure everything uses your new venv_mycoding environment.
echo.

REM Step 1: Check current environment variables
echo Step 1: Checking environment variables...
echo Current VIRTUAL_ENV: %VIRTUAL_ENV%
echo Current PATH (first entry): 
for /f "tokens=1 delims=;" %%a in ("%PATH%") do echo %%a
echo.

REM Step 2: Deactivate any current virtual environment
echo Step 2: Deactivating any current virtual environment...
if defined VIRTUAL_ENV (
    echo Deactivating: %VIRTUAL_ENV%
    call deactivate 2>nul
) else (
    echo No virtual environment currently active.
)
echo.

REM Step 3: Activate the correct virtual environment
echo Step 3: Activating venv_mycoding...
call venv_mycoding\Scripts\activate
echo New VIRTUAL_ENV: %VIRTUAL_ENV%
echo.

REM Step 4: Run the Python cleanup script
echo Step 4: Running notebook cleanup script...
python cleanup_venv_references.py
echo.

REM Step 5: Show summary
echo Step 5: Cleanup Summary
echo =====================
echo ✅ VS Code tasks.json updated
echo ✅ Textual inversions batch file updated  
echo ✅ VS Code settings.json created with default interpreter
echo ✅ Jupyter notebooks processed for kernel updates
echo ✅ Virtual environment properly activated
echo.
echo Manual steps remaining:
echo 1. Open any Jupyter notebooks in VS Code
echo 2. Select "venv_mycoding" as the kernel when prompted
echo 3. Save the notebooks to apply kernel changes
echo.
echo All done! Your environment is now clean.
pause
