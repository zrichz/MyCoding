@echo off
echo ========================================
echo    Image Stacking Tool Launcher
echo ========================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM Check if virtual environment exists
if not exist "focus_env\Scripts\activate.bat" (
    echo Virtual environment not found. Creating new environment...
    echo.
    
    REM Create virtual environment
    python -m venv focus_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please make sure Python is installed and accessible.
        pause
        exit /b 1
    )
    
    echo Virtual environment created successfully!
    echo.
)

echo Activating virtual environment...
call focus_env\Scripts\activate.bat

REM Check if requirements.txt exists and install dependencies
if exist "requirements.txt" (
    echo Installing/updating dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some packages may have failed to install.
        echo The GUI might still work, but some features may be limited.
        echo.
    ) else (
        echo Dependencies installed successfully!
        echo.
    )
) else (
    echo No requirements.txt found. Installing basic dependencies...
    python -m pip install --upgrade pip
    pip install opencv-python pillow numpy customtkinter matplotlib scipy imageio networkx
    if errorlevel 1 (
        echo WARNING: Some packages may have failed to install.
        echo.
    )
)

echo.
echo ========================================
echo      Launching Image Stacking Tool
echo ========================================
echo.

REM Launch the GUI
python main.py

REM Check if the GUI launched successfully
if errorlevel 1 (
    echo.
    echo ERROR: Failed to launch the GUI.
    echo Please check the error messages above.
    echo.
    pause
) else (
    echo.
    echo GUI closed successfully.
)

echo.
echo Deactivating virtual environment...
deactivate

echo.
echo Press any key to exit...
pause >nul
