# PowerShell Environment Setup & Preflight Check for cGMP-QC-CoPilot
$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "cGMP-QC-CoPilot Environment Setup (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Locate Python Interpreter
$PYTHON_CMD = "python"
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $PYTHON_CMD = "py"
}

Write-Host "[*] Checking Python executable: $PYTHON_CMD"
& $PYTHON_CMD --version

# 2. Install or verify required packages
Write-Host "`n[*] Installing / Verifying requirements from requirements.txt..." -ForegroundColor Yellow
& $PYTHON_CMD -m pip install --upgrade pip
& $PYTHON_CMD -m pip install -r requirements.txt

# 3. Run Preflight Integrity Check
Write-Host "`n[*] Executing preflight_check.py..." -ForegroundColor Yellow
& $PYTHON_CMD preflight_check.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] cGMP-QC-CoPilot environment is fully configured and verified!" -ForegroundColor Green
    Write-Host "To launch the interactive dashboard, run:" -ForegroundColor Green
    Write-Host "    streamlit run app.py" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Preflight checks failed. Please inspect the output above." -ForegroundColor Red
}
