@echo off
echo Starting Gray-Scott Filter...
cd /d "%~dp0"
call "..\.venv\Scripts\activate.bat"
python GrayScott_filter.py
pause
