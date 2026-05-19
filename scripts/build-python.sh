#!/usr/bin/env bash
# Bundles the Python backend into a single-folder distribution using PyInstaller.
# Output: <repo>/python-dist/backend (+ deps) — referenced by Electron in
# production via process.resourcesPath/python-dist/backend.
# Mirror of scripts/build-python.ps1 (Windows). Keep both in sync.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
cd "$root"

echo "[build-python] Installing build deps..."
python3 -m pip install --upgrade pip
python3 -m pip install -r backend/requirements.txt
python3 -m pip install pyinstaller

echo "[build-python] Running PyInstaller..."
dist="$root/python-dist"
build="$root/build-python"
rm -rf "$dist" "$build"

python3 -m PyInstaller \
  --name backend \
  --noconfirm \
  --noupx \
  --distpath "$dist" \
  --workpath "$build" \
  --specpath "$build" \
  --collect-all uvicorn \
  --collect-all fastapi \
  --collect-all pydantic \
  --collect-all httpx \
  --collect-all llama_cpp \
  --collect-binaries llama_cpp \
  --collect-data llama_cpp \
  --collect-submodules backend \
  --hidden-import websockets \
  --paths "$root" \
  "$root/backend_entry.py"

# PyInstaller writes to python-dist/backend/backend — flatten one level so the
# layout matches the Windows side (python-dist/backend instead of
# python-dist/backend/backend).
inner="$dist/backend"
if [ -d "$inner" ]; then
  # macOS mv refuses to merge dirs, so use a temp shuffle.
  tmp="$dist/_flatten"
  mv "$inner" "$tmp"
  shopt -s dotglob
  mv "$tmp"/* "$dist"/
  shopt -u dotglob
  rmdir "$tmp"
fi

echo "[build-python] Done: $dist/backend"
