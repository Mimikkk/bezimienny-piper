#!/bin/bash
# Train the Piper onnx for Bezimienny.
# piper-phonemize ships a Linux wheel for Python 3.11, not 3.12.
# Torch stays on CUDA 12.8 so the RTX 5090 can run it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends build-essential python3-dev software-properties-common
if ! command -v python3.11 >/dev/null 2>&1; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3.11 -m venv "$ROOT/.venv"
fi
export PATH="$ROOT/.venv/bin:$PATH"

python -m pip install -U pip wheel setuptools
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  'pytorch-lightning>=2.4' torchmetrics \
  'piper-phonemize~=1.1.0' 'librosa>=0.9.2,<1' \
  'cython>=0.29,<1' onnx onnxsim onnxscript onnxruntime 'numpy>=1.19'

if [[ ! -d "$ROOT/vendor/piper/.git" ]]; then
  git clone --depth 1 https://github.com/rhasspy/piper.git "$ROOT/vendor/piper"
fi
(
  cd "$ROOT/vendor/piper/src/python"
  bash build_monotonic_align.sh
)
python -m pip install -e "$ROOT/vendor/piper/src/python" --no-deps
python "$ROOT/vast/patch_piper_lightning.py" \
  "$ROOT/vendor/piper/src/python/piper_train/vits/lightning.py"
python -c "import torch; assert torch.cuda.is_available(); print('cuda', torch.__version__, torch.cuda.get_device_name(0), flush=True)"
python src/train_piper.py
echo "onnx: $ROOT/weights/piper/pl_PL-bezimienny-medium.onnx"
