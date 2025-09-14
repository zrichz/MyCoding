@echo off
REM Activate the existing venv and run image_quarteriser.py
cd /d %~dp0
call ..\..\activate_venv.bat
python image_quarteriser.py %*
