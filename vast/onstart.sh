#!/bin/bash
# Remote entry. The rented image has no PyTorch. RTX 5090 needs a CUDA 12.8 build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
PY=python3
if ! $PY -m venv "$ROOT/.venv"; then
  $PY -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu128
  $PY -m pip install --break-system-packages -r requirements-train.txt
else
  "$ROOT/.venv/bin/python" -m pip install -U pip
  "$ROOT/.venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  "$ROOT/.venv/bin/python" -m pip install -r requirements-train.txt
  PY="$ROOT/.venv/bin/python"
fi
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'; print('cuda', torch.__version__, torch.cuda.get_device_name(0), flush=True)"
$PY src/train_voice.py
echo "weights: $ROOT/weights/bezimienny"
