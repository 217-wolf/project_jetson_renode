# Skrypt PowerShell do uruchomienia testu ReID na pliku wideo
param(
    [string]$VideoPath = "test_wideo_reid.mp4",
    [switch]$Reset
)

# Wykrywanie środowiska Python (.jetson venv lub systemowy)
if (Test-Path ".jetson\Scripts\python.exe") {
    $PythonExe = ".jetson\Scripts\python.exe"
} else {
    $PythonExe = "python"
}

$ExtraArgs = @()
if ($Reset) {
    $ExtraArgs += "--reset"
    Write-Host ">> WLACZONO RESET ZAPISÓW TOŻSAMOSCI (ID rozpocznie sie od #1) <<" -ForegroundColor Magenta
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Uruchamianie ReID na pliku: $VideoPath" -ForegroundColor Green
Write-Host "  Interpreter: $PythonExe" -ForegroundColor Yellow
Write-Host "  Nacisnij 'q' lub 'Esc' w oknie, aby zakonczyc" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

& $PythonExe sourcer/main.py --mode camera --video $VideoPath $ExtraArgs