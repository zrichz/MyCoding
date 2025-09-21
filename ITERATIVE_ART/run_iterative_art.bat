@echo off
REM Activate the main .venv and run iterative_art_1.py
set SCRIPT_DIR=%~dp0
set MYCODING_DIR=%SCRIPT_DIR%..\
call "%MYCODING_DIR%\.venv\Scripts\activate.bat"
python "%SCRIPT_DIR%iterative_art_1.py"
pause
