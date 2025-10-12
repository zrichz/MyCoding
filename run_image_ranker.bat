@echo off
echo Starting Image Ranker...
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run VENV_SETUP.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Running Image Ranker...
python image_processors\image_ranker.py

echo.
echo Deactivating virtual environment...
call deactivate

pause