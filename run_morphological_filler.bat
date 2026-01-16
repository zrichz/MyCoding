@echo off
REM Morphological Color Filler Launcher
REM Double-click this file to start the web interface

cd /d %~dp0

echo.
echo ====================================
echo   Morphological Color Filler
echo ====================================
echo.
echo Starting web interface...
echo A browser window will open automatically.
echo.
echo Press Ctrl+C to stop the server when done.
echo.

C:\MyCoding\newvenv2026\Scripts\python.exe morphological_color_filler.py

pause
