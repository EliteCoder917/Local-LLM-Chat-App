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
# IMPORTANT: never let pip touch llama-cpp-python here. The venv holds our
# hand-built CUDA 12.8 / sm_120 wheel. Reinstalling from requirements.txt
# (which only pins `>=0.2.70`) risks either a 20-minute source rebuild OR a
# silent swap to a CPU-only PyPI wheel — both clobber the GPU build. So we
# install everything EXCEPT llama-cpp-python, then verify the custom build is
# still present below.
Write-Host "[build-python]   (skipping llama-cpp-python — using the venv's custom CUDA build)"

# Fast path: if every runtime dep + pyinstaller already imports, skip pip
# entirely. On a working dev venv this is the normal case, and it avoids
# pip's slow (sometimes minutes-long, output-free) network resolution against
# PyPI — which is what looks like a "stuck" build.
$probe = @"
import importlib.util
mods = ['fastapi', 'uvicorn', 'websockets', 'pydantic', 'httpx', 'PyInstaller']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print('MISSING:' + ','.join(missing))
"@
$probeOut = (& $py -c $probe).Trim()
$missing = ($probeOut -replace '^MISSING:', '').Split(',', [StringSplitOptions]::RemoveEmptyEntries)

if ($missing.Count -eq 0) {
  Write-Host "[build-python]   all deps already present — skipping pip install"
} else {
  Write-Host "[build-python]   installing missing deps: $($missing -join ', ')"
  # Only install what's actually missing. --timeout guards against a hung
  # network call rather than blocking forever with no output.
  $reqPath = Join-Path $root "backend\requirements.txt"
  $reqLines = Get-Content $reqPath | Where-Object { $_ -notmatch '^\s*llama-cpp-python' }
  $tmpReq = Join-Path $env:TEMP "requirements-no-llama.txt"
  $reqLines | Set-Content -Encoding ascii $tmpReq
  & $py -m pip install --disable-pip-version-check --timeout 60 -r $tmpReq
  if ($LASTEXITCODE -ne 0) { Write-Error "[build-python] pip install of deps failed."; exit 1 }
  if ($missing -contains 'PyInstaller') {
    & $py -m pip install --disable-pip-version-check --timeout 60 pyinstaller
    if ($LASTEXITCODE -ne 0) { Write-Error "[build-python] pip install of pyinstaller failed."; exit 1 }
  }
}

# Verify llama-cpp-python is actually installed (we did NOT install it above).
# Fail loudly rather than bundling an app with no inference engine.
$llamaVer = (& $py -c "import llama_cpp; print(llama_cpp.__version__)" 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $llamaVer) {
  Write-Error "[build-python] llama-cpp-python is not installed in this venv. Install the custom CUDA wheel into .venv first, then re-run."
  exit 1
}
Write-Host "[build-python] llama-cpp-python present: $($llamaVer.Trim())"

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

# Put the WMI-bypass sitecustomize on PYTHONPATH so it auto-loads in EVERY
# python process spawned from here — including the isolated subprocesses
# PyInstaller 6.x forks during analysis. Those subprocesses re-import
# PyInstaller's compat module (which calls platform.win32_ver() → WMI), so
# patching only the parent isn't enough; a wedged winmgmt service would still
# hang the analysis phase. The sitecustomize neutralises the WMI call globally
# for the build. (run_pyinstaller.py covers the parent as belt-and-suspenders.)
$wmiBypass = Join-Path $here "_wmi_bypass"
if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$wmiBypass;$($env:PYTHONPATH)"
} else {
  $env:PYTHONPATH = $wmiBypass
}
Write-Host "[build-python] WMI bypass on PYTHONPATH: $wmiBypass"

# Invoke PyInstaller through our wrapper (scripts/run_pyinstaller.py) rather
# than `-m PyInstaller`. The wrapper disables platform.win32_ver()'s WMI query,
# which hangs `import PyInstaller` indefinitely whenever the machine's winmgmt
# service is wedged — a stall that looks identical to a stuck build.
& $py "$here\run_pyinstaller.py" `
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
