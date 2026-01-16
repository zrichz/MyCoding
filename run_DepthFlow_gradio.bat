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
echo browser will auto-launch shortly...
echo.

C:\MyCoding\depthflow_env\Scripts\python.exe -m DepthFlow gradio

pause
