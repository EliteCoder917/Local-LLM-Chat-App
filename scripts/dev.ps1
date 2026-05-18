# Dev launcher — starts Python backend, Vite, and Electron together.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Resolve-Path "$here\..")

if (-not (Test-Path .\.venv)) {
  Write-Host "[dev] Creating .venv..."
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt | Out-Null

if (-not (Test-Path .\node_modules)) {
  Write-Host "[dev] Installing npm deps..."
  npm install
}

Write-Host "[dev] Launching..."
npm run dev
