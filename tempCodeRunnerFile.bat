@echo off
REM DepthFlow Gradio Web Interface Launcher
REM Double-click this file to start the web interface

cd /d %~dp0

echo.
echo ====================================
echo   DepthFlow - Gradio Web Interface
echo ====================================
echo.
echo Starting web interface...
echo A browser window will open automatically.
echo.
echo Press Ctrl+C to stop the server when done.
echo.

C:\MyCoding\depthflow_env\Scripts\python.exe -m DepthFlow gradio

pause