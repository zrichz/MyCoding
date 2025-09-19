@echo off
echo Starting Interactive L-System Fern Generator...
echo.

REM Activate virtual environment
call ..\venv_mycoding\Scripts\activate.bat

REM Run the interactive L-System fern generator
python iterative_L_System_fern.py

pause