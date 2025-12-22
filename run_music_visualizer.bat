@echo off
REM Music Visualizer Launcher for Windows
REM Uses the existing .venv in MyCoding directory

echo 🎵 Music Visualizer Launcher
echo ==============================

REM Get the directory of this script
set "SCRIPT_DIR=%~dp0"
set "VENV_PATH=%SCRIPT_DIR%.venv"

REM Check if virtual environment exists
if not exist "%VENV_PATH%" (
    echo ❌ Error: Virtual environment not found at %VENV_PATH%
    echo Please ensure the .venv directory exists in the MyCoding folder.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

echo 📦 Checking and installing required packages...

REM Install required packages
pip install librosa matplotlib opencv-python numpy scipy soundfile ffmpeg-python

echo.
echo ✅ Packages installed!
echo.
echo 🚀 Starting Music Visualizer...
python "%SCRIPT_DIR%music_visualizer.py"

echo Done.
pause