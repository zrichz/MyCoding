@echo off
echo ================================================
echo  AI Background Remover
echo ================================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
call ..\newvenv2026\Scripts\activate.bat
python background_remover.py

pause
