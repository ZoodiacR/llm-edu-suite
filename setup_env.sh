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
"$PY" -m pip install -r requirements.txt
"$PY" -m ipykernel install --user --name llm-edu-suite --display-name "Python (llm-edu-suite)"

# Local LLMs for notebooks 10-16 (Ollama). Install it from https://ollama.com if it is missing.
if command -v ollama >/dev/null 2>&1; then
  ollama pull qwen2.5:3b
  ollama pull nomic-embed-text
else
  echo "Ollama not found: install it from https://ollama.com, then run: ollama pull qwen2.5:3b && ollama pull nomic-embed-text"
fi
[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example (add your LANGSMITH_API_KEY for tracing)"; }
"$PY" -m pytest -q -m "not ollama"
echo -e "\nListo. Activa el entorno con:  source .venv/bin/activate   y abre:  jupyter lab notebooks/"
