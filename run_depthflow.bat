@echo off
REM DepthFlow Launcher - Runs DepthFlow with depthflow_env environment
REM
REM Usage: 
REM   run_depthflow.bat              - Shows main menu
REM   run_depthflow.bat gradio       - Starts Gradio web interface
REM   run_depthflow.bat main         - Starts main application
REM   run_depthflow.bat --help       - Shows help

cd /d %~dp0

echo.
echo ====================================
echo   DepthFlow - 3D Parallax Creator
echo ====================================
echo.

C:\MyCoding\depthflow_env\Scripts\python.exe -m DepthFlow %*

pause
