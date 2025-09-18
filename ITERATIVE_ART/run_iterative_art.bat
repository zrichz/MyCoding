@echo off
REM Activate the image_processors .venv and run iterative_art_1.py
set SCRIPT_DIR=%~dp0
set MYCODING_DIR=%SCRIPT_DIR%..\
call "%MYCODING_DIR%\image_processors\.venv\Scripts\activate.bat"
python "%SCRIPT_DIR%iterative_art_1.py"
