# PowerShell script to run TI Changer Multiple with virtual environment
Write-Host "Running TI Changer Multiple with virtual environment..." -ForegroundColor Green
Write-Host "Virtual Environment: C:\MyPythonCoding\MyCoding\image_processors\.venv" -ForegroundColor Yellow
Write-Host ""

& "C:\MyPythonCoding\MyCoding\image_processors\.venv\Scripts\python.exe" "TI_CHANGER_MULTIPLE_2024_10_22.py"

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
