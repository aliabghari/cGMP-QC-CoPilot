#!/usr/bin/env bash
# Shell Environment Setup & Preflight Check for cGMP-QC-CoPilot
set -e

echo "======================================================================"
echo "cGMP-QC-CoPilot Environment Setup (POSIX Shell)"
echo "======================================================================"

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[*] Using Python: $($PYTHON_CMD --version)"

echo ""
echo "[*] Installing / Verifying requirements from requirements.txt..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

echo ""
echo "[*] Executing preflight_check.py..."
$PYTHON_CMD preflight_check.py

echo ""
echo "[SUCCESS] cGMP-QC-CoPilot environment is fully configured and verified!"
echo "To launch the interactive dashboard, run:"
echo "    streamlit run app.py"
