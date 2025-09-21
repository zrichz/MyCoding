@echo off
cd /d "%~dp0"
echo Starting SuperJPEG Encoder/Decoder...
echo.

REM Activate the virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found at .venv\Scripts\activate.bat
    echo Please ensure the virtual environment is set up correctly.
    pause
    exit /b 1
)

REM Run the SuperJPEG encoder
python superjpeg_encoder.py

REM Deactivate virtual environment
deactivate

echo.
echo SuperJPEG Encoder/Decoder closed.
pause
