@echo off
echo Starting Gray-Scott Filter...
cd /d "C:\MyPythonCoding\MyCoding\image_processors\GrayScott_filter\src"
call "..\..\..\.venv\Scripts\activate.bat"
python GrayScott_filter.py
pause
