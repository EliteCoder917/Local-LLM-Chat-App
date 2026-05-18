# Bundles the Python backend into a single-folder distribution using PyInstaller.
# Output: <repo>\python-dist\backend.exe (+ deps) — referenced by Electron in
# production via process.resourcesPath\python-dist\backend.exe.
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$here\.."
Set-Location $root

Write-Host "[build-python] Installing build deps..."
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install pyinstaller

Write-Host "[build-python] Running PyInstaller..."
$dist  = Join-Path $root "python-dist"
$build = Join-Path $root "build-python"
if (Test-Path $dist)  { Remove-Item -Recurse -Force $dist }
if (Test-Path $build) { Remove-Item -Recurse -Force $build }

pyinstaller `
  --name backend `
  --noconfirm `
  --noupx `
  --distpath $dist `
  --workpath $build `
  --specpath $build `
  --collect-all uvicorn `
  --collect-all fastapi `
  --collect-all pydantic `
  --collect-all httpx `
  --collect-all llama_cpp `
  --collect-binaries llama_cpp `
  --collect-data llama_cpp `
  --collect-submodules backend `
  --hidden-import websockets `
  --paths $root `
  $root\backend_entry.py

# PyInstaller writes to python-dist\backend\backend.exe — flatten one level.
$inner = Join-Path $dist "backend"
if (Test-Path $inner) {
  Get-ChildItem $inner -Force | Move-Item -Destination $dist -Force
  Remove-Item -Recurse -Force $inner
}

Write-Host "[build-python] Done: $dist\backend.exe"
