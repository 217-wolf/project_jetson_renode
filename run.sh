#!/bin/bash
# Skrypt uruchomieniowy dla Jetson Orin Nano

# Aktywacja środowiska (jeśli używasz venv/conda)
# source venv/bin/activate

# Sprawdź czy CUDA jest dostępne
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Uruchom system
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python3 sourcer/main.py 