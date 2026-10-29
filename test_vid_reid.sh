#!/usr/bin/env bash

# Skrypt uruchamiający test śledzenia i ReID na pliku wideo
# Użycie:
#   ./test_vid_reid.sh [opcjonalna_sciezka_do_wideo] [--reset]
# --reset aby usunąć wszystkie zapisane embeddingi
# Domyślnie: test_wideo_reid.mp4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

VIDEO_PATH="test_wideo_reid.mp4"
EXTRA_ARGS=()

for arg in "$@"; do
    if [ "$arg" == "--reset" ]; then
        EXTRA_ARGS+=("--reset")
    elif [[ "$arg" == *.mp4 ]] || [[ "$arg" == *.avi ]] || [[ "$arg" == *.mkv ]]; then
        VIDEO_PATH="$arg"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

#wybranie .venv (.jetson) lub globalnego pythona

if [ -f ".jetson/bin/python" ]; then
    PYTHON=".jetson/bin/python"
elif [ -f ".jetson/Scripts/python.exe" ]; then
    PYTHON=".jetson/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "=============================================="
echo "  Uruchamianie ReID na pliku: $VIDEO_PATH"
echo "  Interpreter: $PYTHON"
echo "  Dodatkowe opcje: ${EXTRA_ARGS[*]}"
echo "  Naciśnij 'q' lub 'Esc' w oknie, aby zakończyć"
echo "=============================================="

exec "$PYTHON" sourcer/main.py --mode camera --video "$VIDEO_PATH" "${EXTRA_ARGS[@]}"