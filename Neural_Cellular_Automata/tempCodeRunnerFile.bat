.exe" (
    echo Using local virtual environment...
    "..\\.venv\\Scripts\\python.exe" nca_animator.py
) else (
    echo Error: Virtual environment not found
    echo Please ensure .venv exists in the parent directory
    pause
    exit /b 1
)