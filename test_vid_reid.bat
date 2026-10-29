@echo off
REM Skrypt uruchamiajacy test ReID na wideo srodowiska Windows
set VIDEO_PATH=%1
if "%VIDEO_PATH%"=="" set VIDEO_PATH=test_wideo_reid.mp4

if exist .jetson\Scripts\python.exe (set PYTHON=.jetson\Scripts\python.exe
) else (
    set PYTHON=python)

echo ==============================================
echo   Uruchamianie ReID na pliku: %VIDEO_PATH%
echo   Interpreter: %PYTHON%
echo   Nacisnij 'Q' lub 'Esc' w oknie, aby zakonczyc
echo ==============================================

%PYTHON% sourcer\main.py --mode camera --video "%VIDEO_PATH%" %2 %3 %4
