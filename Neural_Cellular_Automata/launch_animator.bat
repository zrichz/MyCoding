@echo off
echo Neural Cellular Automata - Animation Generator
echo ============================================
echo.

echo Launching NCA Animation Generator...

REM Try the local .venv first
if exist "..\\.venv\\Scripts\\python.exe" (
    echo Using local virtual environment...
    "..\\.venv\\Scripts\\python.exe" nca_animator.py
) else (
    echo Error: Virtual environment not found
    echo Please ensure .venv exists in the parent directory
    pause
    exit /b 1
)

echo.
echo Animation Generator closed.
pause
