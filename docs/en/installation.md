🇪🇸 [Versión en español](../es/instalacion.md)

# Installation and working environment

[← Back to the index](README.md)

This guide gets the project ready to run **everything**: the test suite, the command-line scripts and the 17 notebooks. Everything runs on **CPU** with no model downloads and no API keys, and an **NVIDIA GPU** is used automatically when present (much faster).

## Requirements

| Component | Minimum version | Verified with |
|---|---|---|
| Python | 3.10 | 3.11.4 (Windows 11) |
| PyTorch | 2.3 (2.5+ for the optional FlexAttention cell) | 2.15.0+cpu |
| NumPy | 1.24 | 2.4.6 |
| matplotlib | 3.7 | 3.11.2 |
| Disk space | ~1 GB (almost all of it is PyTorch) | |
| RAM | 4 GB | |

> **Why a dedicated virtual environment.** Installing into your global Python can drag in broken combinations.
> A real case found while testing this project: a `matplotlib 3.7` built against NumPy 1.x next to
> `numpy 2.4` fails on import with `ImportError: numpy.core.multiarray failed to import`. An isolated
> `.venv` avoids this kind of conflict.

## Option A — Automatic script (recommended)

**Windows (PowerShell):**

```powershell
cd llm-edu-suite
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

**Linux / macOS:**

```bash
cd llm-edu-suite
bash setup_env.sh
```

The script, in order:

1. Creates `.venv/` if it does not exist.
2. Installs PyTorch: the **CUDA** build if it detects an NVIDIA GPU (`nvidia-smi`), otherwise the **CPU** build (a small, universal wheel).
3. Installs `requirements-dev.txt` (numpy, matplotlib, pytest, jupyterlab, ipykernel, nbclient, requests).
4. Registers the environment as a Jupyter kernel named **"Python (llmsuite)"**.
5. Runs `pytest -q` to confirm everything works (all tests should pass; at the time of writing pytest collects 50 tests).

## Option B — Step by step

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python -m ipykernel install --user --name llmsuite --display-name "Python (llmsuite)"
pytest -q
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # on macOS: pip install torch
pip install -r requirements-dev.txt
python -m ipykernel install --user --name llmsuite --display-name "Python (llmsuite)"
pytest -q
```

> If PowerShell blocks `Activate.ps1`, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call
> `.\.venv\Scripts\python.exe` directly without activating the environment.

### Do I have an NVIDIA GPU?

Nothing in the project needs one, but it speeds training up a lot: the mini LLM in notebook 00 takes about **22 s on an RTX 3060 Laptop versus about 5 min on CPU**. The setup scripts detect the GPU and install CUDA PyTorch automatically, and the notebooks pick it up on their own (`llmsuite.utils.get_device()` returns `cuda` when available and `cpu` otherwise; Apple MPS only with `allow_mps=True`). On GPU, training uses **bf16** mixed precision (RTX 30xx or newer).

To install it by hand, or to switch a `.venv` that already has the CPU build:

```bash
nvidia-smi                                   # check the driver and the CUDA version it supports
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128     # or cu130 with recent drivers
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## What to run next

| What | Command | Approx. time (CPU) |
|---|---|---|
| Tests | `pytest -q` | ~10 s |
| Open the notebooks | `jupyter lab notebooks/` and pick the **Python (llmsuite)** kernel | — |
| Run every notebook without opening it | `python tools/run_notebooks.py` | see the [notebooks guide](notebooks_guide.md) |
| Run a single one | `python tools/run_notebooks.py 05_pretraining.ipynb` (or by number: `python tools/run_notebooks.py 05 14`) | — |
| Pretrain a small LM | `python scripts/train.py --steps 300 --out runs/smoke` | ~2.5 min |
| Generate tokens with it | `python scripts/sample.py --run runs/smoke --prompt "" --max-new-tokens 50` | seconds |
| Post-training SFT→DPO / GRPO | `python scripts/finetune.py --lora` | a few minutes (~5 min according to the script's docstring) |
| RAG over the sample documents | `python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json` | seconds |
| Regenerate the notebooks from code | `python tools/build_notebooks_part1.py && python tools/build_notebooks_part2.py && python tools/build_notebooks_part3.py` | seconds |

With `make` available (Linux/macOS/Git Bash): `make test`, `make smoke`, `make notebooks`, `make run-notebooks`, `make mini-llm`, `make mcp-server` (see [command-line scripts](cli_scripts.md#tools-and-makefile-targets)).

> **Important about the notebooks:** the `.ipynb` files are **generated** from `tools/build_notebooks_part*.py`.
> To change a notebook permanently, edit the generator and regenerate;
> if you edit the `.ipynb` directly, your changes will be lost the next time they are regenerated.

## Using a real LLM instead of `MockLLM`

Notebooks 10, 11 and 13 and `scripts/rag_demo.py` use `MockLLM` by default, a **deterministic,
rule-based** model that lets everything run offline. To plug in a real model, all you need is a
server that speaks the OpenAI API:

```python
from llmsuite.llm_client import OpenAICompatibleClient

# local vLLM:   vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
llm = OpenAICompatibleClient("Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")

# local Ollama: ollama serve  (then: ollama pull qwen2.5:7b)
llm = OpenAICompatibleClient("qwen2.5:7b", base_url="http://localhost:11434/v1")

# cloud API (OpenAI, OpenRouter, ...): the key is read from the OPENAI_API_KEY variable
llm = OpenAICompatibleClient("gpt-4o-mini", base_url="https://api.openai.com/v1")
```

```bash
python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?" \
    --model qwen2.5:7b --base-url http://localhost:11434/v1
```

> Never put API keys in code or notebooks: use environment variables
> (`OPENAI_API_KEY`) or a `.env` file, which is already in `.gitignore`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: numpy.core.multiarray failed to import` | matplotlib built for NumPy 1.x with NumPy 2.x installed | use the project's `.venv` (or `pip install -U matplotlib`) |
| `ModuleNotFoundError: llmsuite` in a notebook | Jupyter was started outside `notebooks/` | start it with `jupyter lab notebooks/`; each notebook adds the repo root (two levels up from `notebooks/<topic>/`) to `sys.path` |
| The notebook runs another Python | wrong kernel | menu *Kernel → Change kernel → Python (llmsuite)* |
| `RuntimeWarning: Proactor event loop...` on Windows | harmless `pyzmq` warning | safe to ignore; `tools/run_notebooks.py` already silences it |
| The FlexAttention cell is skipped | PyTorch < 2.5 or no CPU support | it is optional; the rest of notebook 03 works the same |
| Training is slow | everything runs on CPU | lower `--steps`/`--d-model`, or use a GPU |
