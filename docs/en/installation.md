🇪🇸 [Versión en español](../es/instalacion.md)

# Installation and working environment

[← Back to the index](README.md)

This guide gets the project ready to run **everything**: the test suite, the command-line scripts and the 32 notebooks.

The course has two halves with different needs:

| Notebooks | What they use | What they need |
|---|---|---|
| **00** | Hugging Face `transformers` on the base SmolLM2-135M, with its attention weights and hidden states | SmolLM2-135M (260 MB) from the Hugging Face Hub on the first run. No Ollama, no training; CPU or GPU. About 1.5 minutes. |
| **01–05, 08, 10** | The from-scratch PyTorch model (package `llm_edu_suite`) | Only PyTorch. CPU works; an NVIDIA GPU is used automatically. |
| **06** | The from-scratch model with `causal=False`, Hugging Face `transformers` + `datasets` (ModernBERT-base, Flan-T5-small, Banking77), `scikit-learn` and LangChain | PyTorch, a first-run download of ModernBERT-base (~600 MB), Flan-T5-small (~300 MB) and Banking77 from the Hugging Face Hub, and **Ollama** with `qwen2.5:3b` and `nomic-embed-text` for the comparison in section 2. About 15 minutes on a GPU. |
| **07** | The from-scratch `TransformerLM` (two trainings), Hugging Face `datasets` + `transformers` (FineWeb and FineWeb-Edu samples, the FineWeb-Edu classifier, the SmolLM2 tokenizer) and LangChain | PyTorch, about 100 MB of FineWeb samples streamed once and kept in `data/cache/`, the `HuggingFaceFW/fineweb-edu-classifier` (~440 MB) and **Ollama** with `qwen2.5:3b` for the synthetic-data section. About 30 minutes on a GPU (two trainings of 10–11 minutes and about 3 minutes of generation). |
| **09** | The from-scratch `TransformerLM`, `torch.profiler`, `torch.utils.checkpoint`, the `F.scaled_dot_product_attention` backends, `torch.compile` and `torch.distributed` | CUDA PyTorch and **an NVIDIA GPU** (it measures GPU memory and kernels; without one it stops at the start). No downloads, no Ollama. On Windows, FlashAttention-2 and Triton are missing and `torchrun` does not work (see [GPU notes](#gpu-notes)); the notebook detects this and carries on. About 5 minutes. |
| **11, 13, 14** | Hugging Face `transformers` + `peft` + `trl` + `bitsandbytes` + `optimum-quanto` on SmolLM2 | PyTorch and a one-time download of SmolLM2 from the Hugging Face Hub (~1 GB). A GPU makes them much faster; the QLoRA section of 13 needs an NVIDIA GPU (`bitsandbytes`). |
| **12** | `httpx` and `asyncio` against Ollama's HTTP API (streamed `/api/chat`, `/api/ps`), `subprocess` for a second `ollama serve`, Hugging Face `datasets` (ARC-Easy, GSM8K) and `config.setup` / `judge_model` | **Ollama** with `qwen2.5:3b`, `qwen2.5:7b` and the three formats of `qwen2.5:1.5b-instruct` (`q4_K_M`, `q8_0` and `fp16`: 986 MB, 1.6 GB and 3.1 GB); ARC-Easy (`allenai/ai2_arc`) and GSM8K (`openai/gsm8k`) from the Hugging Face Hub on the first run. **A GPU**: every result is a latency. The notebook starts and stops a second, temporary Ollama server on a free port (with `OLLAMA_NUM_PARALLEL=4`) and reads the models folder from the Ollama desktop app's `server.log`. No training. About 28 minutes on a GPU, most of it the 3B → 7B cascade on GSM8K. |
| **15** | LangChain with Ollama (`config.get_llm` and `get_judge_llm`), Hugging Face `datasets` (GSM8K) and Pydantic for the judge's structured output | **Ollama** with `qwen2.5:3b`, `deepseek-r1:1.5b` (1.1 GB) and, as verifier and pairwise judge, the optional `qwen2.5:7b`; the GSM8K test split from the Hugging Face Hub on the first run. No training. About 53 minutes, almost all of it LLM calls. |
| **16, 18, 19, 23, 25, 26, 28** | LangChain, RAGAS, LangGraph, LangSmith, the official MCP SDK + `langchain-mcp-adapters`; in 23, also FAISS and LightGBM | A local **Ollama** server with `qwen2.5:3b` and `nomic-embed-text` (and, optionally, `qwen2.5:7b` as notebook 18's judge and notebook 25's code-writing agent). Notebook 25 also uses `pandas`, which is installed as a dependency of `datasets`. LangSmith is optional. Notebook 23 downloads MovieLens-1M (6 MB) on the first run. |
| **17** | `sentence-transformers` (bi-encoders, cross-encoders, contrastive training, embedding quantization), `rank-bm25`, Hugging Face `datasets` and LangChain's `OllamaEmbeddings` through `config.get_embeddings` | SciFact and NFCorpus (`mteb/scifact`, `mteb/nfcorpus`) and four models (all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L6-v2, BAAI/bge-reranker-base) from the Hugging Face Hub on the first run, and **Ollama** with `nomic-embed-text`. A GPU is recommended (three short fine-tunings and two cross-encoders); not measured on CPU. About 18 minutes on a GPU. |
| **20** | DSPy 3.5 (signatures, modules, `LabeledFewShot`, `BootstrapFewShotWithRandomSearch`, `MIPROv2`, `GEPA`) with its own HTTP client (`engine="lm15"`) to Ollama's OpenAI-compatible `/v1` endpoint, Optuna (MIPROv2's Bayesian search), `sentence-transformers` (bge-small-en-v1.5) and Hugging Face `datasets` (Banking77) | **Ollama** with `qwen2.5:3b` and `qwen2.5:7b`; Banking77 and bge-small-en-v1.5 from the Hugging Face Hub on the first run (already cached if you ran 06 and 17). No litellm, no LangSmith traces. About 57 minutes on a GPU, most of it the optimizers and the 7B classifier. |
| **21** | `sentence-transformers` (all-MiniLM-L6-v2, bge-small-en-v1.5), `umap-learn`, `scikit-learn` (`KMeans`, `HDBSCAN`, `CountVectorizer`, the clustering metrics), Hugging Face `datasets` (Banking77) and LangChain with Ollama (`config.setup` and `get_llm`) to name the clusters | **Ollama** with `qwen2.5:3b`; Banking77 (`PolyAI/banking77`) and the two embedding models from the Hugging Face Hub on the first run (already cached if you ran 06 and 17). A GPU speeds up the encoding but is not needed: the models are small. No training. About 196 s on a GPU, 88 s of it the LLM naming the clusters. |
| **22** | Hugging Face `transformers` (CLIP) and `datasets` (Oxford-IIIT Pet), `scikit-learn` for the linear probe, the `ollama` Python client for the vision model, Pillow to render and alter images, Pydantic for the invoice schema, and `config.setup` | **Ollama** with `qwen2.5vl:3b` (3.2 GB); CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) and Oxford-IIIT Pet (`timm/oxford-iiit-pet`) from the Hugging Face Hub on the first run. **A GPU is recommended** (CLIP encodes the 3,680 training images and the test sample, and the VLM answers one image at a time). No training beyond a logistic regression. About 12 minutes on a GPU, most of it VLM calls. |
| **24** | NumPy and `llm_edu_suite.recsys` (`download_movielens_1m`, `load_movielens_1m`) for the recommender stream, and `sentence-transformers`, `scikit-learn` and Hugging Face `datasets` for the embedding drift tests; `config.setup` in the setup cell | MovieLens-1M through the course's own helper (the same `data/cache/` download as notebook 23), and Banking77 (`PolyAI/banking77`) and bge-small-en-v1.5 from the Hugging Face Hub on the first run (already cached if you ran 17 and 21). No new dependency and no LLM calls. A GPU only speeds up encoding Banking77. About 90 s on a GPU. |
| **27** | mini-swe-agent 2.4.6 (with litellm) in a **separate virtual environment**, `.venv-harness`, started by `scripts/harness_run.py` (which imports `scripts/harness_modern.py`); bash for the agent's commands (Git Bash on Windows); `pytest` for the hidden tests | The harness environment (see [The agent-harness environment](#the-agent-harness-environment-notebook-27)) and a model. By default **Claude Haiku 5.5** through the Anthropic API when `ANTHROPIC_API_KEY` is in `.env`: 366 agent runs, about **US$ 0.42** for the whole notebook (see [Optional API keys](#optional-api-keys-notebook-27)). Without the key it falls back to **Ollama** with `qwen2.5:7b`, which is much slower (in v2.8, local models took 64–216 s per task against about 7 s with Claude). `HARNESS_MODEL` picks any other litellm model. No downloads besides the model, no training, no LangSmith traces. |
| **29** | Inspect AI 0.3.289 and Phoenix 21.0.0 in a **separate virtual environment**, `.venv-eco` (its `inspect` and `phoenix` commands); in the course environment, `arize-phoenix-otel` (an OpenTelemetry exporter preconfigured for Phoenix) and `openinference-instrumentation-langchain`; LangChain with Ollama, and `httpx` for Phoenix's REST API | The Inspect and Phoenix environment (see [The Inspect and Phoenix environment](#the-inspect-and-phoenix-environment-notebook-29)) and **Ollama** with `qwen2.5:3b`; GSM8K from the Hugging Face Hub on the first run (already cached if you ran 15). Phoenix runs on your machine, on a free local port. About 23 minutes on a GPU, almost all of it the four Inspect passes over the 80 problems. |
| **30** | LangChain with Ollama (`config.get_llm` and `judge_model`), the `ollama` Python client (installed with `langchain-ollama`) for Llama Guard, Hugging Face `datasets` and Pydantic for the LLM judge's structured output | **Ollama** with `qwen2.5:3b`, `qwen2.5:7b` and `llama-guard3:1b` (1.6 GB); JailbreakBench (`JailbreakBench/JBB-Behaviors`), XSTest (`Paul/XSTest`) and XSTest's human-labelled replies (`natolambert/xstest-v2-copy`) from the Hugging Face Hub on the first run. No training. About 48 minutes on a GPU, almost all of it generation. |

No API key is required: everything can run locally. One key is optional and recommended: with `ANTHROPIC_API_KEY` in `.env`, notebook 27 runs its agent on Claude Haiku 5.5 (about US$ 0.42 per full run) instead of a much slower local model (see [Optional API keys](#optional-api-keys-notebook-27)).

## Requirements

| Component | Minimum | Verified with |
|---|---|---|
| Python | 3.11 | 3.11.4 (Windows 11) |
| PyTorch | 2.3 (2.5+ for the optional FlexAttention cell in notebook 04) | 2.14.1+cu130 |
| NVIDIA GPU | optional | RTX 3060 Laptop GPU, 6 GB |
| [Ollama](https://ollama.com) | any recent version | 0.40.15 |
| Disk: virtual environment | ~4 GB with the CUDA build of PyTorch (PyTorch alone is ~2.9 GB); much less with the CPU build | 4.1 GB |
| Disk: Ollama models | ~2.2 GB (`qwen2.5:3b` 1.9 GB + `nomic-embed-text` 0.27 GB); 4.7 GB more with the optional `qwen2.5:7b` (judge in 18, code agent in 25), 1.1 GB for `deepseek-r1:1.5b` (notebook 15), 1.6 GB for `llama-guard3:1b` (notebook 30), about 5.7 GB for the three formats of `qwen2.5:1.5b-instruct` (notebook 12) 3.2 GB for the vision model `qwen2.5vl:3b` (notebook 22) and, only to rerun v2.8's local comparison of notebook 27, 5.2 GB for `freehuntx/qwen3-coder:8b` (without an API key, notebook 27 uses `qwen2.5:7b`) | |
| Disk: downloaded data | MovieLens-1M, 6 MB zipped, and the FineWeb (12,000 documents) and FineWeb-Edu (1,000) samples, about 45 MB of JSONL, in `data/cache/` | |
| Disk: Hugging Face cache | ~1 GB (SmolLM2-135M-Instruct 260 MB + SmolLM2-360M-Instruct 690 MB), plus ~0.9 GB for notebook 06 (ModernBERT-base ~600 MB + Flan-T5-small ~300 MB, plus Banking77), and ~0.45 GB more for notebook 07 (`fineweb-edu-classifier` ~440 MB and the SmolLM2-135M tokenizer), plus the four models and two datasets of notebook 17, the three small benchmark datasets of notebook 30 and CLIP ViT-B/32 with Oxford-IIIT Pet for notebook 22 (notebooks 20, 21 and 24 reuse Banking77 and notebook 17's models), plus the base SmolLM2-135M (260 MB) for notebook 00 | |

The dependencies are declared once, in `pyproject.toml` (with lower bounds); `requirements.txt` only contains `-e .[dev]`, which installs the project from there with the `dev` extra (pytest and Jupyter). There is no `requirements-dev.txt` any more. The optional extras `openai` and `anthropic` add the cloud providers (see [Switching provider](#switching-provider)). v2.1 added two dependencies: `lightgbm>=4.5` (the LambdaMART ranker in notebook 23) and `bitsandbytes>=0.45` (the NF4 weights for QLoRA in notebook 13). Notebook 06 added `scikit-learn>=1.5` (logistic regression and the classification metrics), and notebook 17 added `sentence-transformers>=5.0` (bi-encoders, cross-encoder rerankers and contrastive training). Notebook 20 added `dspy>=3.5` and `optuna>=4` (the Bayesian search behind MIPROv2), and notebook 29 `arize-phoenix-otel>=0.17` and `openinference-instrumentation-langchain>=0.1.82` (the client side of tracing to Phoenix; Inspect AI and the Phoenix server live in their own environment, see [below](#the-inspect-and-phoenix-environment-notebook-29)). These are the versions the course was verified with:

| Package | Version | Package | Version |
|---|---|---|---|
| transformers | 5.22.2 | langchain | 1.4.9 |
| peft | 0.21.2 | langchain-core | 1.7.6 |
| trl | 1.17.1 | langchain-community | 0.4.2 |
| optimum-quanto | 0.2.7 | langchain-ollama | 1.1.0 |
| ragas | 0.4.3 | langgraph | 1.2.20 |
| mcp | 1.30.0 (must stay `<2`) | langsmith | 0.14.17 |
| langchain-mcp-adapters | 0.3.2 | faiss-cpu | 1.15.1 |
| bitsandbytes | 0.50.2 | lightgbm | 4.7.0 |
| scikit-learn | 1.9.1 | sentence-transformers | 6.2.0 |
| umap-learn | 0.6.0 | dspy | 3.5.0 |
| optuna | 5.0.0 | arize-phoenix-otel | 0.17.3 |
| openinference-instrumentation-langchain | 0.1.82 | | |

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
3. Installs `requirements.txt`, which installs the project with its `dev` extra from `pyproject.toml` (NumPy, matplotlib, python-dotenv, LangChain, LangGraph, LangSmith, RAGAS, FAISS, rank-bm25, LightGBM, scikit-learn, umap-learn, sentence-transformers, transformers, peft, trl, datasets, accelerate, bitsandbytes, optimum-quanto, mcp, langchain-mcp-adapters, DSPy, Optuna, arize-phoenix-otel, openinference-instrumentation-langchain) plus pytest, JupyterLab, ipykernel, nbclient and nbformat.
4. Registers the environment as a Jupyter kernel named **"Python (llm-edu-suite)"**.
5. If the `ollama` command exists, runs `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`. If it does not, it prints how to install Ollama and continues. It does not pull the optional `qwen2.5:7b` judge; do that by hand if you want it (see [Ollama](#ollama)).
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
pip install -r requirements.txt
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
pip install -r requirements.txt
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

[Ollama](https://ollama.com) serves the chat model and the embedding model of the application notebooks on your own machine, on the GPU if there is one.

1. **Install it.** Windows and macOS: download the installer from [ollama.com/download](https://ollama.com/download). Linux: `curl -fsSL https://ollama.com/install.sh | sh`.
2. **Start it.** The desktop app (Windows/macOS) and the Linux service start the server automatically. Otherwise run `ollama serve` in a separate terminal. It listens on `http://localhost:11434`.
3. **Pull the course models** (about 2.2 GB in total):

   ```bash
   ollama pull qwen2.5:3b          # chat model, 3.1B parameters, Q4_K_M, supports tool calling
   ollama pull nomic-embed-text    # embedding model for retrieval and memory
   ```

   **Optional: a stronger judge.** Notebook 18 uses `qwen2.5:7b` to grade the 3B model's answers with RAGAS, notebook 25 uses it for the agent that answers questions by writing pandas code (the 3B model rarely manages it), and `scripts/generate_item_keywords.py` uses it to write the movie keywords:

   ```bash
   ollama pull qwen2.5:7b          # 4.7 GB; the RAGAS judge (notebook 18) and the code agent (notebook 25)
   ```

   **Notebook 15** also needs a small reasoning model, and uses the 7B judge as its verifier and pairwise judge:

   ```bash
   ollama pull deepseek-r1:1.5b    # 1.1 GB; DeepSeek-R1-Distill-Qwen-1.5B (notebook 15)
   ```

   **Notebook 30** needs the guard model, and uses the 7B model both as the second model under test and as the candidate LLM judge:

   ```bash
   ollama pull llama-guard3:1b     # 1.6 GB; Llama Guard 3 1B, the safety classifier (notebook 30)
   ```

   **Notebook 12** compares the same 1.5B model in three GGUF formats, and uses the 7B model in its gateway and its 3B → 7B cascade:

   ```bash
   ollama pull qwen2.5:1.5b-instruct-q4_K_M   # 986 MB (notebook 12)
   ollama pull qwen2.5:1.5b-instruct-q8_0     # 1.6 GB (notebook 12)
   ollama pull qwen2.5:1.5b-instruct-fp16     # 3.1 GB (notebook 12)
   ```

   **Notebook 22** needs a vision-language model, which reads images as well as text:

   ```bash
   ollama pull qwen2.5vl:3b        # 3.2 GB; Qwen2.5-VL 3B (notebook 22)
   ```

   Everything works without the 7B judge: `config.judge_model()` falls back to the chat model and the 3B model grades itself. Notebook 18 explains why that is a bad idea: with the 3B judge, differences between configurations showed up that the 7B judge does not confirm. With the 7B model, RAGAS scoring takes about 17 minutes on a 6 GB GPU, against about 7 with the 3B. To use another judge, set `JUDGE_MODEL` in `.env`.

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
| `JUDGE_MODEL` | `qwen2.5:7b` if it is pulled in Ollama, otherwise the chat model | The model that acts as the judge (LLM-as-judge) in evaluations: RAGAS in notebook 18 and `config.get_judge_llm()`. It should be stronger than the model it grades. |
| `EMBEDDING_MODEL` | `nomic-embed-text` (Ollama), `text-embedding-3-small` (OpenAI) | The embedding model returned by `get_embeddings()`. Anthropic has no embeddings API, so with `anthropic` the embeddings still come from Ollama. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama server is. |
| `LANGSMITH_TRACING` | `true` in `.env.example` | Send traces to LangSmith. It only takes effect if `LANGSMITH_API_KEY` is also set. |
| `LANGSMITH_API_KEY` | empty | Your LangSmith API key. |
| `LANGSMITH_PROJECT` | `llm-edu-suite` | The LangSmith project the traces go to. `setup(project=...)` overrides it per notebook, so each notebook's traces land in their own project. |
| `LANGSMITH_ENDPOINT` | (commented out) | Only for accounts in the EU region: `https://eu.api.smith.langchain.com`. |
| `OPENAI_API_KEY` | (commented out) | Needed only with `LLM_PROVIDER=openai`. |
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HARNESS_MODEL`, `HARNESS_REPS`, `HARNESS_WORKERS` | (commented out / not set) | Optional, for the model of notebook 27; see [Optional API keys](#optional-api-keys-notebook-27). `ANTHROPIC_API_KEY` is also the key for `LLM_PROVIDER=anthropic`. |

With an empty `LANGSMITH_API_KEY`, tracing stays off and everything else works the same. Notebook 28 (observability) is the one that gets the most out of LangSmith.

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

`langchain-openai` and `langchain-anthropic` are optional extras (`pip install -e ".[openai]"` or `".[anthropic]"`), because the default setup does not need them. A hosted model costs money per call, and some notebook results (the numbers printed in the markdown) were obtained with `qwen2.5:3b`, so yours will differ. `scripts/rag_demo.py --ragas` always calls its judge through Ollama's OpenAI-compatible endpoint, so it needs Ollama whatever the provider.

### Optional API keys (notebook 27)

Nothing in the course needs a paid API, but notebook 27 is the one place where an API key changes the experience. Its agent harness runs 16 coding tasks, 3 times each, under several harness conditions: 366 agent runs in all. **With `ANTHROPIC_API_KEY` in `.env`, its default model is Claude Haiku 5.5** (`anthropic/claude-haiku-5-5`, called through litellm inside mini-swe-agent): the final run used 2.52 M prompt and 0.33 M completion tokens, **US$ 0.42** at US$ 0.10 / 0.50 per million input / output tokens, about 7 s per run. **Without a key** the notebook falls back to `qwen2.5:7b` through Ollama and prints that it is doing so. That works, but it is much slower: in v2.8 the local models took 64–216 s per task and solved 20–30% of tasks 01–10.

| Variable | What it does |
|---|---|
| `ANTHROPIC_API_KEY` | Makes Claude Haiku 5.5 the default model of notebook 27. It is the same variable `ChatAnthropic` reads with `LLM_PROVIDER=anthropic` (see [Switching provider](#switching-provider)). |
| `HARNESS_MODEL` | Picks the model, as a litellm id: `anthropic/claude-haiku-5-5`, `openai/...`, `gemini/...`, or an Ollama model name such as `qwen2.5:7b`. `scripts/harness_run.py` treats a model as an API model when its prefix is one of `gemini`, `anthropic`, `openai`, `openrouter`, `groq`, `mistral`, `deepseek` or `xai`; anything else goes to Ollama. The notebook's cost line only knows Claude Haiku 5.5's price. |
| `HARNESS_REPS` | Runs per task and condition (default `3`). One agent run is one draw; fewer runs are cheaper and noisier. |
| `HARNESS_WORKERS` | Runs in parallel (default `4` for API models, `1` for a local model, since two Ollama jobs at once can hang a 6 GB card). |
| `GEMINI_API_KEY` | Only used if you set `HARNESS_MODEL` to a Gemini model (for example `gemini/gemini-3.6-flash`). Gemini's free tier allows about 20 requests per day and model, too few for this notebook. |
| `HARNESS_MIN_INTERVAL` | Seconds to wait between model calls (for example `7`), for free tiers with a requests-per-minute limit. Default `0`; the notebook sets 7 for Gemini models. |

**Where the keys go.** Put them in `.env`, which is git-ignored (`.env.example` already lists `ANTHROPIC_API_KEY`, commented out), never in a notebook cell or on a command line. The notebook loads `.env` itself (`ANTHROPIC_API_KEY`, `HARNESS_MODEL`) and starts `scripts/harness_run.py` with its own environment, so litellm can read the key. **The key never reaches the agent's shell**: every command the agent runs gets a scrubbed environment with only `SYSTEMROOT`, `PATH`, `TEMP`, `TMP`, `HOME`, `USERPROFILE` and `LANG`, and the permission policy refuses commands that read `.env`. `harness_run.py` does not read `.env` itself: when you run it by hand with an API model, the key has to be set in the shell that starts it.

## The agent-harness environment (notebook 27)

Notebook 27 runs a real coding-agent harness, [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT), with Claude Haiku 5.5 by default or a local model through Ollama. mini-swe-agent lives in **its own virtual environment**, `.venv-harness`, next to `.venv`: its terminal UI needs a newer `rich` than the course environment allows, and keeping it apart means neither environment breaks the other. The notebook itself still runs in the course kernel; it starts `scripts/harness_run.py` with the harness environment's Python, one process per run (4 at a time for an API model), and runs the hidden tests with that Python's `pytest`.

**1. Create the environment** (from the repository root; `.venv-*/` is already in `.gitignore`):

```powershell
# Windows
python -m venv .venv-harness
.venv-harness\Scripts\python -m pip install mini-swe-agent==2.4.6 pytest
```

```bash
# Linux / macOS
python3 -m venv .venv-harness
.venv-harness/bin/python -m pip install mini-swe-agent==2.4.6 pytest
```

The notebook looks for exactly `.venv-harness/Scripts/python.exe` (Windows) or `.venv-harness/bin/python` (Linux/macOS) at the repository root, and stops with `Create the harness environment first` if it is not there. Version 2.4.6 is the one the notebook was run with (it brings litellm, which is what talks to the Anthropic API or to Ollama); the notebook prints the source of mini-swe-agent's functions, so another version can show different code.

**2. Bash.** The agent's only tool is a bash shell. On Linux and macOS the runner uses the `bash` on the `PATH`. On Windows it uses **Git Bash**, from [Git for Windows](https://git-scm.com/download/win), at `C:\Program Files\Git\bin\bash.exe`; if yours is somewhere else, set the `HARNESS_BASH` environment variable to the full path of `bash.exe` before starting Jupyter.

**3. The model.** With `ANTHROPIC_API_KEY` in `.env` there is nothing to download: the notebook uses Claude Haiku 5.5 (see [Optional API keys](#optional-api-keys-notebook-27)). Without it, or with `HARNESS_MODEL` set to a local model, it goes through Ollama (the same server as the rest of the course; the `OLLAMA_BASE_URL` environment variable points the runner at another one):

```bash
ollama pull qwen2.5:7b                  # 4.7 GB, the fallback when there is no API key
ollama pull freehuntx/qwen3-coder:8b    # 5.2 GB, optional: one of v2.8's three local models (with qwen2.5:3b and qwen2.5:7b)
```

For local models the runner asks Ollama for a **16k context** (`num_ctx=16384`), because an agent's history grows with every step. On a 6 GB card that is fine for the 3B and the 7B, but **the 8B model at 16k does not fit in VRAM**: `ollama ps` shows about 46 % of it on the CPU. In v2.8 (tasks 01–10) a run took 64 s with the 3B, 82 s with the 7B and 216 s with the 8B, against about 7 s with Claude Haiku 5.5. With a local model, run the notebook alone (see [GPU notes](#gpu-notes)).

**4. Check it** without the notebook: `-h` lists the arguments, and a folder that holds a `TASK.md` is enough for a real run (see [CLI scripts](cli_scripts.md#harness_runpy)):

```powershell
.venv-harness\Scripts\python scripts\harness_run.py -h
```

> The runner's sandbox (a scrubbed environment, a timeout, a deny list of commands, the task folder as working directory) is a teaching layer, like notebook 25's, not a security boundary. Point it only at throw-away task folders.

## The Inspect and Phoenix environment (notebook 29)

Notebook 29 uses two open-source tools that live outside the course environment: [Inspect AI](https://inspect.aisi.org.uk) (UK AI Security Institute, MIT), which needs a newer `rich` than the course environment allows (the same reason as `.venv-harness`), and the [Phoenix](https://github.com/Arize-ai/phoenix) server (Arize, Elastic License 2.0). Both go into **a third virtual environment**, `.venv-eco`, next to `.venv`. The notebook itself runs in the course kernel: it calls `.venv-eco`'s `inspect` command for the evaluations and starts its `phoenix` command as a separate server process, the way Phoenix would run next to an application in production. The client side of tracing, the OpenTelemetry exporter and the LangChain instrumentor, is in the course environment.

**1. Create the environment** (from the repository root; `.venv-*/` is already in `.gitignore`):

```powershell
# Windows
python -m venv .venv-eco
.venv-eco\Scripts\python -m pip install inspect-ai==0.3.289 arize-phoenix==21.0.0 datasets
```

```bash
# Linux / macOS
python3 -m venv .venv-eco
.venv-eco/bin/python -m pip install inspect-ai==0.3.289 arize-phoenix==21.0.0 datasets
```

`datasets` is there because the Inspect task loads GSM8K itself. These are the versions the notebook was run with, and its first cell prints them (`Inspect AI 0.3.289 | Phoenix 21.0.0`). The notebook looks for exactly `.venv-eco/Scripts/inspect.exe` (Windows) or `.venv-eco/bin/inspect` (Linux/macOS) at the repository root, and stops with `create .venv-eco first (see scripts/inspect_tasks/gsm8k_cot.py)` if it is not there.

**2. The client packages** go into the course's `.venv` and are already in `pyproject.toml`: `arize-phoenix-otel` (OpenTelemetry's SDK and an exporter preconfigured for Phoenix) and `openinference-instrumentation-langchain` (the spans for LangChain, in OpenInference format). If your `.venv` was created before notebook 29, run `pip install -r requirements.txt` again inside it.

**3. Where the data goes.** Inspect writes one JSON log per run to `data/cache/inspect_logs/` and Phoenix keeps its SQLite database and its `server.log` in `data/cache/phoenix/`; the notebook empties both folders at the start of each run. Phoenix listens on `127.0.0.1` only, on a free port chosen by the notebook, with its own telemetry turned off (`PHOENIX_TELEMETRY_ENABLED=false`), so no trace leaves the machine. While the notebook runs, the Phoenix UI is at the address its cell prints.

**4. Check it** without the notebook (the task's own docstring has the same commands; see [CLI scripts](cli_scripts.md#inspect_tasksgsm8k_cotpy)):

```powershell
.venv-eco\Scripts\inspect eval scripts\inspect_tasks\gsm8k_cot.py --model ollama/qwen2.5:3b --model-base-url http://localhost:11434/v1 --temperature 0 --max-tokens 512 --max-connections 1 -T n=5
.venv-eco\Scripts\inspect view          # browse the logs
```

## What to run next

Times measured on the verification machine (RTX 3060 Laptop GPU, Ollama running locally).

| What | Command | Time |
|---|---|---|
| Fast tests (no Ollama needed) | `pytest -q -m "not ollama"` | ~42 s (90 tests; with fake models, they also run attack code against notebook 25's sandbox and check that `mcp_server.py --require-token` rejects requests without the bearer token) |
| Ollama integration tests | `pytest -q -m ollama` | ~40 s (4 tests; they skip themselves if Ollama is not running) |
| All tests | `pytest -q` | the two above combined |
| Open the notebooks | `jupyter lab notebooks/` and pick the kernel **Python (llm-edu-suite)** | — |
| Run every notebook headlessly | `python tools/run_notebooks.py` | long; see the [notebooks guide](notebooks_guide.md) |
| Run some notebooks | `python tools/run_notebooks.py 05 14` (by number) or `python tools/run_notebooks.py 08_pretraining.ipynb` | — |
| Rebuild notebooks from their generators | `python tools/build_notebooks.py` (all) or `python tools/build_notebooks.py 05 14`; with `--keep-outputs` it keeps the outputs of code cells that did not change | seconds |
| Pretrain a small LM | `python scripts/train.py --steps 300 --out runs/smoke` | see [CLI scripts](cli_scripts.md#trainpy) |
| SFT with LoRA on SmolLM2 | `python scripts/finetune.py sft` | ~1.5 min on GPU |
| RAG over the course docs | `python scripts/rag_demo.py --ask "What is GQA?"` | ~15 s |
| MCP server | `python scripts/mcp_server.py` | — |

The `.ipynb` files are **generated** from `tools/notebooks/nbNN_*.py` (and `scripts/mini_llm.py` from notebook 01's generator). To change a notebook permanently, edit its generator and rebuild; edits made directly in the `.ipynb` are lost on the next build. Building clears the outputs, so run `tools/run_notebooks.py` afterwards; if you only changed prose (markdown, or anything that leaves the code untouched), `python tools/build_notebooks.py --keep-outputs 12` keeps the executed outputs of every code cell whose source did not change and saves you a re-run ([details](cli_scripts.md#toolsbuild_notebookspy---keep-outputs)). With `make` available (Linux/macOS/Git Bash): `make test`, `make smoke`, `make notebooks`, `make run-notebooks`, `make mini-llm`, `make mcp-server` (see [CLI scripts](cli_scripts.md#tools-and-makefile-targets)).

## Data downloaded at run time

- **SmolLM2** (notebooks 11, 13 and 14 and `finetune.py`): from the Hugging Face Hub into the Hugging Face cache, on first use.
- **ModernBERT-base, Flan-T5-small and Banking77** (notebooks 06, 21 and 24): `answerdotai/ModernBERT-base` (~600 MB), `google/flan-t5-small` (~300 MB) and the `PolyAI/banking77` dataset (CC BY 4.0) are downloaded from the Hugging Face Hub into its cache the first time the notebook runs. Notebook 21 treats Banking77's 3,080 test messages as unlabeled logs (the labels are only used to score the clustering) and uses its training split for the known-intent centroids of the monitoring section; it also uses `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`, the same downloads as notebook 17. Notebook 24 encodes Banking77's training split with `BAAI/bge-small-en-v1.5` for its drift tests.
- **FineWeb, FineWeb-Edu and the FineWeb-Edu classifier** (notebook 07): the notebook streams (`datasets.load_dataset(..., streaming=True)`) the first 12,000 documents of `HuggingFaceFW/fineweb` and 1,000 of `HuggingFaceFW/fineweb-edu` (config `sample-10BT`), about 100 MB of download, and stores them once in `data/cache/fineweb_12k.jsonl` and `data/cache/fineweb_edu_1k.jsonl` (about 45 MB). Later runs read those files offline. FineWeb is ODC-By licensed and the repository does not redistribute it. The `HuggingFaceFW/fineweb-edu-classifier` (~440 MB) and the `HuggingFaceTB/SmolLM2-135M` tokenizer go into the Hugging Face cache on first use.
- **GSM8K** (notebook 15): the test split of `openai/gsm8k` (config `main`, MIT license) is downloaded from the Hugging Face Hub into its cache the first time the notebook runs; the notebook uses 80 of its problems.
- **SciFact, NFCorpus and the retrieval models** (notebook 17): the BEIR copies of SciFact and NFCorpus on the Hugging Face Hub (`mteb/scifact` and `mteb/nfcorpus`, configs `corpus`, `queries` and `default` for the relevance labels) and the models `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2` and `BAAI/bge-reranker-base` are downloaded into the Hugging Face cache the first time the notebook runs. `nomic-embed-text` comes from Ollama, as in the application notebooks. The fine-tuned models are trained in memory and not saved.
- **JailbreakBench and XSTest** (notebook 30): the 100 harmful behaviours and their 100 benign twins of `JailbreakBench/JBB-Behaviors` (config `behaviors`, splits `harmful` and `benign`), the XSTest prompts of `Paul/XSTest` (the notebook samples 100 safe and 100 unsafe ones) and `natolambert/xstest-v2-copy`, the 900 GPT-4 and Llama 2 replies to XSTest with their human labels, used to validate the refusal judge. All three are downloaded from the Hugging Face Hub into its cache the first time the notebook runs. `llama-guard3:1b` comes from Ollama. The notebook scores the replies to harmful requests but never prints or saves them.
- **ARC-Easy and GSM8K** (notebook 12): the test split of ARC-Easy (`allenai/ai2_arc`, config `ARC-Easy`; the notebook samples 200 questions with a fixed seed) and the same 80 GSM8K problems as notebook 15 (`openai/gsm8k`, config `main`) are downloaded from the Hugging Face Hub into its cache the first time the notebook runs. The policy documents used as the cached system prompt are the course's own `data/sample_docs/`.
- **Oxford-IIIT Pet and CLIP** (notebook 22): the `timm/oxford-iiit-pet` copy of Oxford-IIIT Pet (Parkhi et al., 2012; 37 cat and dog breeds, 3,680 training images; the notebook samples 600 test images with a fixed seed) and the model `openai/clip-vit-base-patch32` are downloaded from the Hugging Face Hub into its cache the first time the notebook runs. `qwen2.5vl:3b` comes from Ollama. The invoices are synthetic, drawn by the notebook with Pillow, so no documents are downloaded.
- **Notebooks 00, 20 and 29**: 00 downloads the base model `HuggingFaceTB/SmolLM2-135M` (260 MB) into the Hugging Face cache; 20 reuses Banking77 (the same 231-message test sample as notebook 06) and `BAAI/bge-small-en-v1.5`, and writes its optimized program to `data/cache/dspy_banking77_program.json`, which it deletes after checking the reload; 29 reuses notebook 15's 80 GSM8K problems (loaded by the Inspect task from `.venv-eco` and by the notebook for the 20 traced requests), and writes the Inspect logs and Phoenix's database under `data/cache/`.
- **MovieLens-1M** (notebooks 23 and 24 and `scripts/generate_item_keywords.py`): `llm_edu_suite.recsys.download_movielens_1m()` fetches the official GroupLens zip (6 MB) and unzips it into `data/cache/ml-1m/` once. `data/cache/` is in `.gitignore`: the GroupLens license does not allow redistributing the dataset, so the repository does not include it and everyone downloads it from the original source. If you are offline, copy `ml-1m.zip` into `data/cache/` and the function will use it.
- What *is* in the repository, because it is the course's own work: `data/recsys/ml1m_llm_keywords.json` (the keywords `qwen2.5:7b` wrote for each movie; regenerating them takes an hour, see [`generate_item_keywords.py`](cli_scripts.md#generate_item_keywordspy)) and `data/recsys/conversational_eval.json` (20 requests with checkable constraints).

## GPU notes

- Nothing requires a GPU except notebook 09, but training is much faster with one: the notebook 01 mini LLM takes about **22 s on an RTX 3060 Laptop vs about 5 min on CPU**. `llm_edu_suite.utils.get_device()` picks `cuda` when available (Apple MPS only with `allow_mps=True`). On RTX 30xx or newer, training uses **bf16** mixed precision.
- **VRAM budget on a 6 GB GPU.** Ollama keeps `qwen2.5:3b` loaded on the GPU for a few minutes after each call, using about **2.3 GB** of VRAM. The training notebooks cap their own PyTorch memory with `torch.cuda.set_per_process_memory_fraction` (environment variable `VRAM_BUDGET_GB`, e.g. `$env:VRAM_BUDGET_GB="3"` or `export VRAM_BUDGET_GB=3` to give them more). The default is **2 GB in notebook 13**, which trains eleven models in a row and whose 4-bit QLoRA model leaves about 0.1 GB behind, and **1.5 GB in 08 and 09**.
- **Do not run the training notebooks (11, 13 and 14, and 17, which trains and calls Ollama) and the Ollama notebooks (16, 18, 19, 23, 25, 26 and 28) at the same time on a 6 GB GPU.** The same goes for 22, which keeps CLIP on the GPU while it calls the vision model. When the two compete for VRAM, Ollama can hang instead of failing. Run them one after the other, and if you need the memory back, unload the models with `ollama stop qwen2.5:3b` (`ollama ps` shows what is loaded).
- **Notebook 12 measures latency, so run it alone.** Any other job on the GPU (a training notebook, another Ollama client) changes every number. For its concurrency section it unloads the models from the default Ollama server, then starts a second server with `OLLAMA_NUM_PARALLEL=4` and `OLLAMA_MAX_LOADED_MODELS=1` on a free port and stops it when it is done, so the default server's settings are never changed. The second server has to find the same models: the notebook reads `OLLAMA_MODELS` from the desktop app's `%LOCALAPPDATA%\Ollama\server.log` and otherwise falls back to the `OLLAMA_MODELS` environment variable or `~/.ollama/models`.
- If Ollama cannot fit a model on the GPU, it runs it (partly) on the CPU: it still works, only slower.
- **Notebook 27 with a local model is one long chain of Ollama calls, so run it alone.** (With Claude Haiku 5.5, its default when `ANTHROPIC_API_KEY` is set, it does not use the GPU.) `freehuntx/qwen3-coder:8b` at the runner's 16k context needs more than 6 GB and runs about 46 % on the CPU, and a second Ollama job at the same time can make Ollama stall. In v2.8's local run one 3B task hit the notebook's safety timeout without a single step, an Ollama stall; the notebook counts such a run as failed and moves on, and it runs local models one at a time (`HARNESS_WORKERS=1`).
- **Notebooks 20 and 29 are long chains of Ollama calls too, so run them one at a time.** Notebook 20 makes thousands of calls (the optimizers alone 650–870 each) and loads `qwen2.5:7b` for its optimizers and its 7B classifier, with bge-small on the GPU next to it; notebook 29 sends 320 Inspect generations and 20 traced requests to `qwen2.5:3b`. Inspect runs with `--max-connections 1`, one request at a time, for the same reason.
- **Notebook 09 on Windows.** The Windows PyTorch wheels ship without FlashAttention-2 (SDPA then uses the memory-efficient or cuDNN kernels) and without Triton, so `torch.compile`'s default backend, `inductor`, fails with `TritonMissing` and the notebook uses `backend="cudagraphs"`. For FlashAttention-2, `torch.compile` with `inductor` or the `flash-attn` package, use Linux or WSL2. The notebook's DDP runs in two CPU processes with `gloo`, launched with `torch.multiprocessing.spawn` and connected through a shared file (`init_method="file://..."`, a `FileStore`), because `torchrun`'s TCP rendezvous needs libuv, which the Windows build does not include; NCCL, the GPU backend, exists only on Linux.
- **Slow instead of out of memory.** On Windows, when a process exceeds the VRAM, the driver can spill into system RAM ("shared GPU memory") instead of raising an error: in notebook 09, SDPA's *math* backend reported 8.1 GB on a 6 GB card and took 978 ms. If something suddenly slows down, check the shared GPU memory in Task Manager.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `!! Ollama is not running`, `Connection refused` on port 11434, or Ollama tests skipped | The Ollama server is not running | Start the Ollama app or run `ollama serve`; check with `curl http://localhost:11434/api/tags` |
| `model "qwen2.5:3b" not found, try pulling it first` (or the same for `nomic-embed-text`) | The models were not pulled | `ollama pull qwen2.5:3b` and `ollama pull nomic-embed-text`; check with `ollama list` |
| A notebook or script hangs on the first LLM call | Ollama is out of VRAM because a training notebook holds the GPU | Stop the training kernel, `ollama stop qwen2.5:3b`, and run the notebooks one after the other (see [GPU notes](#gpu-notes)) |
| `LangChainDeprecationWarning` / a warning that `langchain-community` is being sunset | `FAISS` and `BM25Retriever` still live in `langchain-community` | Harmless; the notebooks and `rag_demo.py` filter it. Nothing to do |
| `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` when importing `ragas` | ragas 0.4.3 imports a module that langchain-community 0.4 removed | Already handled: notebook 18 and `rag_demo.py` register an empty stand-in module before importing ragas. Copy that shim (`_ragas_compat_shim` in `scripts/rag_demo.py`) if you import ragas in your own code |
| `UserWarning: huggingface_hub cache-system uses symlinks...` on Windows | Windows does not allow symlinks without Developer Mode | Harmless (the cache uses a bit more disk). Enable Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide it |
| MCP stdio client fails inside Jupyter with an error about `fileno` / stderr | Jupyter's `sys.stderr` has no OS file handle, and the SDK passes it to the server subprocess | Already handled in notebook 26, which sends the server's stderr to a log file. Not needed in normal scripts |
| `ModuleNotFoundError: llm_edu_suite` in a notebook | Jupyter was opened outside the repo, or with another kernel | Open with `jupyter lab notebooks/`; each notebook adds the repo root to `sys.path` |
| `ModuleNotFoundError: No module named 'umap'` in notebook 21 | The environment was created before `umap-learn` was added to `pyproject.toml` | `pip install -r requirements.txt` again (or `pip install umap-learn`) inside `.venv` |
| The notebook uses another Python | Wrong kernel | *Kernel → Change kernel → Python (llm-edu-suite)* |
| `RuntimeWarning: Proactor event loop...` on Windows | Harmless `pyzmq` warning | Ignore it; `tools/run_notebooks.py` already silences it |
| The FlexAttention cell is skipped | PyTorch < 2.5 or no CPU support | It is optional; the rest of notebook 04 works the same |
| `ImportError: numpy.core.multiarray failed to import` | matplotlib built for NumPy 1.x with NumPy 2.x installed | Use the project's `.venv` (or `pip install -U matplotlib`) |
| Slow training | Everything runs on CPU | Install the CUDA build of PyTorch, or lower `--steps` / `--d-model` |
| Notebook 09 stops with `this notebook measures GPU memory and kernels; it needs a CUDA GPU` | No NVIDIA GPU, or the CPU build of PyTorch | Install the CUDA build of PyTorch; without a GPU the notebook has nothing to measure |
| `SDPA: FlashAttention-2: not available in this PyTorch build` or `inductor backend: unavailable here (TritonMissing)` in notebook 09 | The Windows PyTorch wheels do not include FlashAttention-2 or Triton | Harmless: the notebook compares the kernels that are available and uses CUDA graphs. To try the missing ones, run it on Linux or WSL2 |
| Notebook 12 stops with `the second Ollama server did not start` | The temporary `ollama serve` did not answer in time; the notebook discards its output, so the reason is hidden | Run `ollama serve` by hand in a terminal with `OLLAMA_HOST=127.0.0.1:<port>` to see the error, and check that `ollama` is on the `PATH` of the Jupyter kernel |
| In notebook 12's concurrency section, the second server answers `model ... not found` | The second server is looking for the models in another folder (for example, you moved them and do not use the desktop app, so there is no `server.log` to read) | Set `OLLAMA_MODELS` to the folder that holds your models before starting Jupyter |
| `bind: An attempt was made to access a socket in a way forbidden by its access permissions` when you start a second `ollama serve` on a fixed port (e.g. `OLLAMA_HOST=127.0.0.1:11435`) | Windows reserves ranges of TCP ports (for Hyper-V, WSL2 or Docker), and the port you picked falls inside one | List the reserved ranges with `netsh interface ipv4 show excludedportrange protocol=tcp` and choose a port outside them, or do as notebook 12 does and let the OS hand out a free one (`socket.bind(("127.0.0.1", 0))`) |
| Notebook 27 stops with `Create the harness environment first` | There is no `.venv-harness` at the repository root, or it has another name | Create it as in [The agent-harness environment](#the-agent-harness-environment-notebook-27); the notebook looks for that exact folder |
| In notebook 27 every run fails at its first command, with an exit status that starts with `error: FileNotFoundError` | The runner could not find bash (on Windows, Git Bash is not at `C:\Program Files\Git\bin\bash.exe`) | Install Git for Windows, or set `HARNESS_BASH` to the full path of your `bash.exe` and restart the kernel |
| Notebook 27 prints `falling back to a local model, which is much slower` | There is no `ANTHROPIC_API_KEY` in `.env` and no `HARNESS_MODEL` | Add the key to `.env` (about US$ 0.42 per full run) and restart the kernel, or let it run on `qwen2.5:7b` |
| In notebook 27 local runs take minutes each, or a run ends as `harness: TimeoutExpired` | A local model at 16k context spills onto the CPU (the 8B qwen3-coder took ~216 s per task in v2.8), or another job is using Ollama | Expected on a 6 GB card; check `ollama ps` and run nothing else on the GPU |
| Notebook 20 stops with `ModuleNotFoundError: No module named 'dspy'`, or notebook 29 with `No module named 'phoenix.otel'` or `'openinference'` | The `.venv` was created before these packages were added to `pyproject.toml` | `pip install -r requirements.txt` again inside `.venv`, and restart the kernel |
| Notebook 29 stops with `create .venv-eco first` | There is no `.venv-eco` at the repository root, or it has another name | Create it as in [The Inspect and Phoenix environment](#the-inspect-and-phoenix-environment-notebook-29); the notebook looks for that exact folder |
| Notebook 29 hangs at `Phoenix up at ...` or its first span query fails | The Phoenix server did not start in time, or exited | Read `data/cache/phoenix/server.log`, where the notebook sends the server's output; check that `.venv-eco` has `arize-phoenix==21.0.0` |
| `torchrun` fails on Windows with an error about libuv | `torchrun`'s TCP rendezvous needs libuv, which the Windows build lacks | Launch the processes with `torch.multiprocessing.spawn` and `init_method="file://..."`, as notebook 09 does, or use Linux or WSL2 |
