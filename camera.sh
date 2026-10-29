#!/usr/bin/env bash
# Skrypt uruchamiający GUI do zbierania embeddingów z kamery
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec "${PYTHON}" sourcer/camera_gui.py
