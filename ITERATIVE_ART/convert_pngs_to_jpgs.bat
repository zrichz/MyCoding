rem copy this file to the folder with PNGs and run it

@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

rem Check for at least one PNG
dir /b *.png >nul 2>&1
if errorlevel 1 (
  echo No PNG files found in "%~dp0".
  pause
  exit /b 0
)

for %%I in (*.png) do (
  echo Converting "%%~fI" -> "%%~nI.jpg"
  ffmpeg -y -loglevel error -i "%%~fI" -q:v 1 "%%~nI.jpg"
)

echo Done.
pause
