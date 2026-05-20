# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = [('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\cublas64_12.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\cublasLt64_12.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\cudart64_12.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\ggml-cuda.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\ggml-base.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\ggml-cpu.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\ggml.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\llama.dll', 'llama_cpp/lib'), ('F:\\AI chat app\\.venv\\Lib\\site-packages\\llama_cpp\\lib\\mtmd.dll', 'llama_cpp/lib')]
hiddenimports = ['websockets']
datas += collect_data_files('llama_cpp')
binaries += collect_dynamic_libs('llama_cpp')
hiddenimports += collect_submodules('backend')
tmp_ret = collect_all('uvicorn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fastapi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pydantic')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('httpx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('llama_cpp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['F:\\AI chat app\\backend_entry.py'],
    pathex=['F:\\AI chat app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='backend',
)
