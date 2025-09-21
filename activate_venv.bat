@echo off
title MyCoding Virtual Environment
echo Activating MyCoding virtual environment...
echo Location: C:\MyPythonCoding\MyCoding\venv_mycoding
echo.

cd /d "C:\MyPythonCoding\MyCoding"
call venv_mycoding\Scripts\activate

echo.
echo Virtual environment activated!
echo Python executable: %cd%\venv_mycoding\Scripts\python.exe
echo Current directory: %cd%
echo.
echo You can now run any Python scripts in this environment.
echo To deactivate, simply type: deactivate
echo.

cmd /k
