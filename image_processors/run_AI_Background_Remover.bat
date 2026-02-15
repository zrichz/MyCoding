
@echo off
echo ================================================
echo  AI Background Remover. firing up...
echo ================================================

cd /d "%~dp0" 
echo Current directory: %CD%
echo.

REM Change to the directory of the batch file
echo Activating virtual environment...
call ..\newvenv2026\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated successfully
echo.

echo Running Python script...
python AI_Background_Remover.py
if errorlevel 1 (
    echo ERROR: Python script failed
    pause
    exit /b 1
)

pause
exit