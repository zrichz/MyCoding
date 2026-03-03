1@echo off
echo ========================================
echo TI Changer SDXL
echo Textual Inversion Manipulation Tool
echo For Stable Diffusion XL
echo ========================================
echo.

echo Activating virtual environment...
call ..\newvenv2026\Scripts\activate.bat

echo Running TI Changer SDXL...
python TI_Changer_SDXL.py
