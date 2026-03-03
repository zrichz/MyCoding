rem copy this file to the folder with PNGs, BMPs, GIFs, and TIFs and run it
rem Converts any PNG, BMP, GIF, and TIF files found, into high-quality JPG files

@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

set "found_files=0"

rem Check for PNG files
dir /b *.png >nul 2>&1
if not errorlevel 1 set "found_files=1"

rem Check for BMP files
dir /b *.bmp >nul 2>&1
if not errorlevel 1 set "found_files=1"

rem Check for GIF files
dir /b *.gif >nul 2>&1
if not errorlevel 1 set "found_files=1"

rem Check for TIF files
dir /b *.tif >nul 2>&1
if not errorlevel 1 set "found_files=1"

rem Check for TIFF files (alternative extension)
dir /b *.tiff >nul 2>&1
if not errorlevel 1 set "found_files=1"

if "%found_files%"=="0" (
  echo No PNG, BMP, GIF, or TIF files found in "%~dp0".
  pause
  exit /b 0
)

echo Converting image files to high-quality JPG...
echo.

rem Convert PNG files
for %%I in (*.png) do (
  echo Converting PNG: "%%~nxI" -> "%%~nI.jpg"
  ffmpeg -y -loglevel error -i "%%~fI" -q:v 1 "%%~nI.jpg"
)

rem Convert BMP files
for %%I in (*.bmp) do (
  echo Converting BMP: "%%~nxI" -> "%%~nI.jpg"
  ffmpeg -y -loglevel error -i "%%~fI" -q:v 1 "%%~nI.jpg"
)

rem Convert GIF files
for %%I in (*.gif) do (
  echo Converting GIF: "%%~nxI" -> "%%~nI.jpg"
  ffmpeg -y -loglevel error -i "%%~fI" -q:v 1 "%%~nI.jpg"
)

rem Convert TIF files
for %%I in (*.tif) do (
  echo Converting TIF: "%%~nxI" -> "%%~nI.jpg"
  ffmpeg -y -loglevel error -i "%%~fI" -q:v 1 "%%~nI.jpg"
)

rem Convert TIFF files (alternative extension)
for %%I in (*.tiff) do (
  echo Converting TIFF: "%%~nxI" -> "%%~nI.jpg"
  ffmpeg -y -loglevel error -i "%%~fI" -q:v 1 "%%~nI.jpg"
)

echo.
echo Done.
pause