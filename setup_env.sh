#!/usr/bin/env bash
# Linux / macOS: create the project virtual environment and register the Jupyter kernel.
#   bash setup_env.sh
set -euo pipefail
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then
  "$PY" -m pip install torch                     # macOS wheels include MPS (Apple Silicon) support
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected -> installing PyTorch with CUDA"
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  else
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi
"$PY" -m pip install -r requirements-dev.txt
"$PY" -m ipykernel install --user --name llm_edu_suite --display-name "Python (llm_edu_suite)"
"$PY" -m pytest -q
echo -e "\nListo. Activa el entorno con:  source .venv/bin/activate   y abre:  jupyter lab notebooks/"
