# Bundles the Python backend into a single-folder distribution using PyInstaller.
# Output: <repo>\python-dist\backend.exe (+ deps) -- referenced by Electron in
# production via process.resourcesPath\python-dist\backend.exe.
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$here\.."
Set-Location $root

# Prefer the project's venv Python so we bundle the custom CUDA-built
# llama-cpp-python that lives there, not a system Python that probably has
# a CPU-only wheel installed. Without this, the bundled exe gets the wrong
# DLL set and `(none of the CUDA DLLs found)` happens silently.
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
  $py = $venvPy
  Write-Host "[build-python] Using venv Python: $py"
} else {
  $py = "python"
  Write-Warning "[build-python] .venv\Scripts\python.exe not found; falling back to system 'python'. The bundled llama-cpp-python will be whatever that system Python has installed."
}

Write-Host "[build-python] Installing build deps..."
& $py -m pip install --upgrade pip
& $py -m pip install -r backend\requirements.txt
& $py -m pip install pyinstaller

Write-Host "[build-python] Running PyInstaller..."
$dist  = Join-Path $root "python-dist"
$build = Join-Path $root "build-python"
if (Test-Path $dist)  { Remove-Item -Recurse -Force $dist }
if (Test-Path $build) { Remove-Item -Recurse -Force $build }

# Discover llama_cpp's lib dir from the active Python (works whether the venv
# is at .venv\, .\venv\, or somewhere else). We need this because
# `--collect-binaries llama_cpp` silently skips the big CUDA DLLs
# (cublasLt64_12.dll alone is 660 MB and seems to hit some internal threshold).
# Force-include them with explicit --add-binary flags so the bundled app
# actually has GPU support.
$llamaLib = (& $py -c "import llama_cpp, os; print(os.path.join(os.path.dirname(llama_cpp.__file__), 'lib'))").Trim()
if (-not (Test-Path $llamaLib)) {
  Write-Error "Could not find llama_cpp lib dir at: $llamaLib"
  exit 1
}
Write-Host "[build-python] llama_cpp lib dir: $llamaLib"

# Sanity check: warn if the discovered lib dir isn't under our venv. That
# almost always means we're about to bundle the wrong llama-cpp-python.
if ($llamaLib -notlike "$root\.venv\*") {
  Write-Warning "[build-python] llama_cpp is NOT in the project venv -- bundling whatever the active Python has, which may be CPU-only."
}

# Explicit DLL list -- anything in $llamaLib that PyInstaller's collect-binaries
# might miss. Mostly the CUDA backend; we re-list the small CPU ones too in
# case --collect-binaries fails entirely. Build the --add-binary arg array.
$forceDlls = @(
  "cublas64_12.dll",
  "cublasLt64_12.dll",
  "cudart64_12.dll",
  "ggml-cuda.dll",
  "ggml-base.dll",
  "ggml-cpu.dll",
  "ggml.dll",
  "llama.dll",
  "mtmd.dll"
)
$addBinaryArgs = @()
foreach ($dll in $forceDlls) {
  $src = Join-Path $llamaLib $dll
  if (Test-Path $src) {
    $sizeMB = [math]::Round((Get-Item $src).Length / 1MB, 1)
    Write-Host "[build-python]   + $dll ($sizeMB MB)"
    # PyInstaller --add-binary syntax on Windows: "src;dest" (semicolon, not
    # colon -- colon is path-separator on Unix only). The dest is the relative
    # path inside the bundle, where llama-cpp-python's loader looks for them.
    $addBinaryArgs += @("--add-binary", "$src;llama_cpp/lib")
  }
}

& $py -m PyInstaller `
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
  @addBinaryArgs `
  $root\backend_entry.py

# PyInstaller writes to python-dist\backend\backend.exe -- flatten one level.
$inner = Join-Path $dist "backend"
if (Test-Path $inner) {
  Get-ChildItem $inner -Force | Move-Item -Destination $dist -Force
  Remove-Item -Recurse -Force $inner
}

Write-Host "[build-python] Done: $dist\backend.exe"
