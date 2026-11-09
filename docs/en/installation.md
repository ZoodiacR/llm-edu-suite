🇪🇸 [Versión en español](../es/instalacion.md)

# Installation and working environment

[← Back to the index](README.md)

This guide gets the project ready to run **everything**: the test suite, the command-line scripts and the 17 notebooks.

The course has two halves with different needs:

| Notebooks | What they use | What they need |
|---|---|---|
| **00–06** | The from-scratch PyTorch model (package `llm_edu_suite`) | Only PyTorch. CPU works; an NVIDIA GPU is used automatically. |
| **07–09** | Hugging Face `transformers` + `peft` + `trl` + `optimum-quanto` on SmolLM2 | PyTorch and a one-time download of SmolLM2 from the Hugging Face Hub (~1 GB). A GPU makes them much faster. |
| **10–16** | LangChain, RAGAS, LangGraph, LangSmith, the official MCP SDK + `langchain-mcp-adapters` | A local **Ollama** server with `qwen2.5:3b` and `nomic-embed-text`. LangSmith is optional. |

No API key is required. Everything runs locally by default.

## Requirements

| Component | Minimum | Verified with |
|---|---|---|
| Python | 3.11 | 3.11.4 (Windows 11) |
| PyTorch | 2.3 (2.5+ for the optional FlexAttention cell in notebook 03) | 2.14.1+cu130 |
| NVIDIA GPU | optional | RTX 3060 Laptop GPU, 6 GB |
| [Ollama](https://ollama.com) | any recent version | 0.40.15 |
| Disk: virtual environment | ~4 GB with the CUDA build of PyTorch (PyTorch alone is ~2.9 GB); much less with the CPU build | 4.1 GB |
| Disk: Ollama models | ~2.2 GB (`qwen2.5:3b` 1.9 GB + `nomic-embed-text` 0.27 GB) | |
| Disk: Hugging Face cache | ~1 GB (SmolLM2-135M-Instruct 260 MB + SmolLM2-360M-Instruct 690 MB) | |

The frameworks are pinned with lower bounds in `requirements.txt`. These are the versions the course was verified with:

| Package | Version | Package | Version |
|---|---|---|---|
| transformers | 5.22.2 | langchain | 1.4.9 |
| peft | 0.21.2 | langchain-core | 1.7.6 |
| trl | 1.17.1 | langchain-community | 0.4.2 |
| optimum-quanto | 0.2.7 | langchain-ollama | 1.1.0 |
| ragas | 0.4.3 | langgraph | 1.2.20 |
| mcp | 1.30.0 (must stay `<2`) | langsmith | 0.14.17 |
| langchain-mcp-adapters | 0.3.2 | | |

> **Why a dedicated virtual environment.** Installing into your global Python can drag in broken combinations
> (for example a `matplotlib` built against NumPy 1.x next to NumPy 2.x fails with
> `ImportError: numpy.core.multiarray failed to import`). The course also pins `mcp<2`, because
> `langchain-mcp-adapters` does not support the 2.x SDK yet. An isolated `.venv` avoids these conflicts.

## Option A: setup script (recommended)

Install Ollama first (see [Ollama](#ollama)) so the script can pull the models. Then:

**Windows (PowerShell):**

```powershell
cd llm-engineering-suite
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

**Linux / macOS:**

```bash
cd llm-engineering-suite
bash setup_env.sh
```

The script does, in order:

1. Creates `.venv/` if it does not exist and upgrades `pip`.
2. Installs PyTorch: the **CUDA** build (`cu128` index) if `nvidia-smi` is found, otherwise the small **CPU** build. On macOS it installs the default wheel, which includes MPS support.
3. Installs `requirements-dev.txt`, which pulls in `requirements.txt` (NumPy, matplotlib, python-dotenv, LangChain, LangGraph, LangSmith, RAGAS, FAISS, rank-bm25, transformers, peft, trl, datasets, accelerate, optimum-quanto, mcp, langchain-mcp-adapters) plus pytest, JupyterLab, ipykernel, nbclient and nbformat.
4. Registers the environment as a Jupyter kernel named **"Python (llm-edu-suite)"**.
5. If the `ollama` command exists, runs `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`. If it does not, it prints how to install Ollama and continues.
6. Creates `.env` from `.env.example` if there is no `.env` yet.
7. Runs the fast tests: `pytest -q -m "not ollama"`.
8. (Windows script) prints the PyTorch version and the GPU it found.

## Option B: step by step

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128    # no NVIDIA GPU: .../whl/cpu
pip install -r requirements-dev.txt
python -m ipykernel install --user --name llm-edu-suite --display-name "Python (llm-edu-suite)"
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
Copy-Item .env.example .env
pytest -q -m "not ollama"
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128    # no NVIDIA GPU: .../whl/cpu ; macOS: pip install torch
pip install -r requirements-dev.txt
python -m ipykernel install --user --name llm-edu-suite --display-name "Python (llm-edu-suite)"
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
cp .env.example .env
pytest -q -m "not ollama"
```

> If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once,
> or call `.\.venv\Scripts\python.exe` directly without activating the environment.

To switch an existing `.venv` from the CPU build to the CUDA build:

```bash
nvidia-smi                                   # checks the driver and the CUDA version it supports
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128     # or cu130 with recent drivers
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Ollama

[Ollama](https://ollama.com) serves the chat model and the embedding model of notebooks 10–16 on your own machine, on the GPU if there is one.

1. **Install it.** Windows and macOS: download the installer from [ollama.com/download](https://ollama.com/download). Linux: `curl -fsSL https://ollama.com/install.sh | sh`.
2. **Start it.** The desktop app (Windows/macOS) and the Linux service start the server automatically. Otherwise run `ollama serve` in a separate terminal. It listens on `http://localhost:11434`.
3. **Pull the course models** (about 2.2 GB in total):

   ```bash
   ollama pull qwen2.5:3b          # chat model, 3.1B parameters, Q4_K_M, supports tool calling
   ollama pull nomic-embed-text    # embedding model for retrieval and memory
   ```

4. **Check it:**

   ```bash
   ollama list                               # both models should be listed
   curl http://localhost:11434/api/tags      # the server answers with JSON
   ollama ps                                 # models currently loaded, and whether they are on GPU or CPU
   ```

From Python, `llm_edu_suite.config.setup()` prints the provider, the models, whether Ollama answers and whether LangSmith is on. Every application notebook calls it in its first cell.

## The `.env` file

`llm_edu_suite/config.py` reads all its settings from environment variables, and loads the project's `.env` with `python-dotenv` (without overriding variables already set in your shell). Copy `.env.example` to `.env` (the setup scripts do it for you) and edit it.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai` or `anthropic`. |
| `LLM_MODEL` | `qwen2.5:3b` (Ollama), `gpt-4o-mini` (OpenAI), `claude-haiku-4-5` (Anthropic) | The chat model returned by `get_llm()`. |
| `EMBEDDING_MODEL` | `nomic-embed-text` (Ollama), `text-embedding-3-small` (OpenAI) | The embedding model returned by `get_embeddings()`. Anthropic has no embeddings API, so with `anthropic` the embeddings still come from Ollama. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama server is. |
| `LANGSMITH_TRACING` | `true` in `.env.example` | Send traces to LangSmith. It only takes effect if `LANGSMITH_API_KEY` is also set. |
| `LANGSMITH_API_KEY` | empty | Your LangSmith API key. |
| `LANGSMITH_PROJECT` | `llm-edu-suite` | The LangSmith project the traces go to. `setup(project=...)` overrides it per notebook, so each notebook's traces land in their own project. |
| `LANGSMITH_ENDPOINT` | (commented out) | Only for accounts in the EU region: `https://eu.api.smith.langchain.com`. |
| `OPENAI_API_KEY` | (commented out) | Needed only with `LLM_PROVIDER=openai`. |

With an empty `LANGSMITH_API_KEY`, tracing stays off and everything else works the same. Notebook 15 (observability) is the one that gets the most out of LangSmith.

### Creating a LangSmith API key

1. Sign in at [smith.langchain.com](https://smith.langchain.com) (there is a free tier).
2. Go to **Settings → API Keys → Create API Key**.
3. Copy the key (it starts with `lsv2_`) and paste it in `.env`: `LANGSMITH_API_KEY=lsv2_...`.
4. Keep `LANGSMITH_TRACING=true`. The next time you run a notebook, `setup()` prints `langsmith: on (project: ...)` and the traces appear under **Projects** in the LangSmith UI.

> **Never commit `.env`.** It is listed in `.gitignore`. Put keys only in `.env` or in environment
> variables, never in code or notebook cells. If a key leaks, revoke it in the same Settings page.

### Switching provider

Changing the provider is a `.env` change, not a code change:

```bash
# OpenAI
pip install langchain-openai
# .env:  LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...   (optional: LLM_MODEL=..., EMBEDDING_MODEL=...)

# Anthropic
pip install langchain-anthropic
# .env:  LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=sk-ant-...   (embeddings still come from Ollama)
```

`langchain-openai` and `langchain-anthropic` are not in `requirements.txt`, because the default setup does not need them. A hosted model costs money per call, and some notebook results (the numbers printed in the markdown) were obtained with `qwen2.5:3b`, so yours will differ. `scripts/rag_demo.py --ragas` always calls its judge through Ollama's OpenAI-compatible endpoint, so it needs Ollama whatever the provider.

## What to run next

Times measured on the verification machine (RTX 3060 Laptop GPU, Ollama running locally).

| What | Command | Time |
|---|---|---|
| Fast tests (no Ollama needed) | `pytest -q -m "not ollama"` | ~35 s (82 tests) |
| Ollama integration tests | `pytest -q -m ollama` | ~40 s (4 tests; they skip themselves if Ollama is not running) |
| All tests | `pytest -q` | the two above combined |
| Open the notebooks | `jupyter lab notebooks/` and pick the kernel **Python (llm-edu-suite)** | — |
| Run every notebook headlessly | `python tools/run_notebooks.py` | long; see the [notebooks guide](notebooks_guide.md) |
| Run some notebooks | `python tools/run_notebooks.py 05 14` (by number) or `python tools/run_notebooks.py 05_pretraining.ipynb` | — |
| Rebuild notebooks from their generators | `python tools/build_notebooks.py` (all) or `python tools/build_notebooks.py 05 14` | seconds |
| Pretrain a small LM | `python scripts/train.py --steps 300 --out runs/smoke` | see [CLI scripts](cli_scripts.md#trainpy) |
| SFT with LoRA on SmolLM2 | `python scripts/finetune.py sft` | ~1.5 min on GPU |
| RAG over the course docs | `python scripts/rag_demo.py --ask "What is GQA?"` | ~15 s |
| MCP server | `python scripts/mcp_server.py` | — |

The `.ipynb` files are **generated** from `tools/notebooks/nbNN_*.py` (and `scripts/mini_llm.py` from notebook 00's generator). To change a notebook permanently, edit its generator and rebuild; edits made directly in the `.ipynb` are lost on the next build. Building clears the outputs, so run `tools/run_notebooks.py` afterwards. With `make` available (Linux/macOS/Git Bash): `make test`, `make smoke`, `make notebooks`, `make run-notebooks`, `make mini-llm`, `make mcp-server` (see [CLI scripts](cli_scripts.md#tools-and-makefile-targets)).

## GPU notes

- Nothing requires a GPU, but training is much faster with one: the notebook 00 mini LLM takes about **22 s on an RTX 3060 Laptop vs about 5 min on CPU**. `llm_edu_suite.utils.get_device()` picks `cuda` when available (Apple MPS only with `allow_mps=True`). On RTX 30xx or newer, training uses **bf16** mixed precision.
- **VRAM budget on a 6 GB GPU.** Ollama keeps `qwen2.5:3b` loaded on the GPU for a few minutes after each call, using about **2.3 GB** of VRAM. The training notebooks cap their own PyTorch memory with `torch.cuda.set_per_process_memory_fraction`, at **1.5 GB by default** (environment variable `VRAM_BUDGET_GB`, e.g. `$env:VRAM_BUDGET_GB="3"` or `export VRAM_BUDGET_GB=3` to give them more).
- **Do not run the training notebooks (07–09) and the Ollama notebooks (10–16) at the same time on a 6 GB GPU.** When the two compete for VRAM, Ollama can hang instead of failing. Run them one after the other, and if you need the memory back, unload the models with `ollama stop qwen2.5:3b` (`ollama ps` shows what is loaded).
- If Ollama cannot fit a model on the GPU, it runs it (partly) on the CPU: it still works, only slower.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `!! Ollama is not running`, `Connection refused` on port 11434, or Ollama tests skipped | The Ollama server is not running | Start the Ollama app or run `ollama serve`; check with `curl http://localhost:11434/api/tags` |
| `model "qwen2.5:3b" not found, try pulling it first` (or the same for `nomic-embed-text`) | The models were not pulled | `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`; check with `ollama list` |
| A notebook or script hangs on the first LLM call | Ollama is out of VRAM because a training notebook holds the GPU | Stop the training kernel, `ollama stop qwen2.5:3b`, and run the notebooks one after the other (see [GPU notes](#gpu-notes)) |
| `LangChainDeprecationWarning` / a warning that `langchain-community` is being sunset | `FAISS` and `BM25Retriever` still live in `langchain-community` | Harmless; the notebooks and `rag_demo.py` filter it. Nothing to do |
| `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` when importing `ragas` | ragas 0.4.3 imports a module that langchain-community 0.4 removed | Already handled: notebook 10 and `rag_demo.py` register an empty stand-in module before importing ragas. Copy that shim (`_ragas_compat_shim` in `scripts/rag_demo.py`) if you import ragas in your own code |
| `UserWarning: huggingface_hub cache-system uses symlinks...` on Windows | Windows does not allow symlinks without Developer Mode | Harmless (the cache uses a bit more disk). Enable Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide it |
| MCP stdio client fails inside Jupyter with an error about `fileno` / stderr | Jupyter's `sys.stderr` has no OS file handle, and the SDK passes it to the server subprocess | Already handled in notebook 16, which sends the server's stderr to a log file. Not needed in normal scripts |
| `ModuleNotFoundError: llm_edu_suite` in a notebook | Jupyter was opened outside the repo, or with another kernel | Open with `jupyter lab notebooks/`; each notebook adds the repo root to `sys.path` |
| The notebook uses another Python | Wrong kernel | *Kernel → Change kernel → Python (llm-edu-suite)* |
| `RuntimeWarning: Proactor event loop...` on Windows | Harmless `pyzmq` warning | Ignore it; `tools/run_notebooks.py` already silences it |
| The FlexAttention cell is skipped | PyTorch < 2.5 or no CPU support | It is optional; the rest of notebook 03 works the same |
| `ImportError: numpy.core.multiarray failed to import` | matplotlib built for NumPy 1.x with NumPy 2.x installed | Use the project's `.venv` (or `pip install -U matplotlib`) |
| Slow training | Everything runs on CPU | Install the CUDA build of PyTorch, or lower `--steps` / `--d-model` |
