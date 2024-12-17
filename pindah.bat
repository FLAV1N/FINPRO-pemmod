@echo off
setlocal enabledelayedexpansion

:: Ubah "C:\Path\Ke\Folder\File" dengan path folder yang berisi file-file
set "source_folder=.\processed_images"

for %%f in ("%source_folder%\*_*.png") do (
    :: Ekstrak nama depan dari nama file
    for /f "tokens=1 delims=_" %%a in ("%%~nf") do (
        :: Buat folder tujuan jika belum ada
        md "%source_folder%\%%a" 2>nul
        :: Pindahkan file ke folder tujuan
        move "%%f" "%source_folder%\%%a"
    )
)

echo File-file telah dipindahkan.
pause