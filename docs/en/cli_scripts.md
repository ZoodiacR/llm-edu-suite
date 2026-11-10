🇪🇸 [Versión en español](../es/scripts_cli.md)

# Command-line scripts

> A guide to the eight scripts in `scripts/` and the Inspect AI task in `scripts/inspect_tasks/` (plus `tools/build_notebooks.py`, `tools/run_notebooks.py` and the `Makefile` targets): every real argument (taken from `argparse`), what each script does internally, usage examples and measured results.
>
> Related documents: [README](README.md) · [Installation](installation.md) · [Module reference](module_reference.md) · [Notebooks guide](notebooks_guide.md) · [Case study](case_study.md) · [Glossary](glossary.md)

| Script | What it is for | Built on | Needs |
|---|---|---|---|
| [`train.py`](#trainpy) | Pretrain a `TransformerLM` on a `.txt` file or a synthetic corpus | `llm_edu_suite` (`data`, `tokenizer`, `model`, `training`, `evaluation`, `utils`) | PyTorch |
| [`sample.py`](#samplepy) | Generate from a checkpoint (normal or speculative sampling) | `llm_edu_suite` (`model`, `tokenizer`, `inference`, `utils`) | PyTorch |
| [`mini_llm.py`](#mini_llmpy) | A complete mini GPT in one file: BPE → model → training → evaluation → sampling → checkpoint | nothing but PyTorch (self-contained) | PyTorch |
| [`finetune.py`](#finetunepy) | Post-training of SmolLM2-135M-Instruct: `sft`, `dpo` or `grpo` subcommands, each evaluated before and after | Hugging Face `transformers` + `peft` + `trl` | PyTorch, ~260 MB model download on first run; a GPU is strongly recommended |
| [`rag_demo.py`](#rag_demopy) | Hybrid RAG with citations over a folder of documents, plus retrieval metrics and optional RAGAS evaluation | LangChain (FAISS + BM25 + `EnsembleRetriever`), RAGAS | Ollama (`qwen2.5:3b`, `nomic-embed-text`) |
| [`mcp_server.py`](#mcp_serverpy) | An MCP server (`llm-edu-docs`) that exposes `data/sample_docs` to Claude Code, Claude Desktop or any MCP client | Official MCP Python SDK (`FastMCP`) + `rank_bm25` | Nothing else (no PyTorch, no Ollama) |
| [`generate_item_keywords.py`](#generate_item_keywordspy) | LLM-written keywords for every MovieLens-1M movie (used by notebook 12); resumable | `llm_edu_suite` (`config`, `recsys`) + LangChain | Ollama with `qwen2.5:7b`; downloads MovieLens-1M (6 MB) on first run |
| [`harness_run.py`](#harness_runpy) | Run the mini-swe-agent coding-agent harness on one task folder with a local model, in a sandboxed bash environment with a permission policy; writes the trajectory as JSON (used by notebook 27) | mini-swe-agent 2.4.6 + litellm | Its own `.venv-harness`, bash (Git Bash on Windows), Ollama with the model to run |
| [`inspect_tasks/gsm8k_cot.py`](#inspect_tasksgsm8k_cotpy) | Notebook 20's GSM8K chain-of-thought evaluation (80 problems, two scorers on the same generations) as an Inspect AI task, run with `inspect eval` (used by notebook 29) | Inspect AI 0.3.289 | Its own `.venv-eco`, Ollama with `qwen2.5:3b` (or any model Inspect can reach) |

---

## Before you start

Run every command from the **repository root** with the project's virtual-environment interpreter (see [installation](installation.md)). The exceptions are `harness_run.py`, which runs with the interpreter of its own environment, `.venv-harness`, and the Inspect task `inspect_tasks/gsm8k_cot.py`, which runs through the `inspect` command of `.venv-eco`. The scripts that import `llm_edu_suite` add the repo root to `sys.path` themselves, so there is no need to install the package with `pip install -e .`.

**Windows (PowerShell or cmd):**

```powershell
.venv\Scripts\python scripts\train.py --steps 300 --out runs\smoke
```

**Linux / macOS:**

```bash
.venv/bin/python scripts/train.py --steps 300 --out runs/smoke
```

Throughout this document each example is shown in both forms. When a long command is split across lines, the continuation character is `` ` `` in PowerShell, `^` in cmd and `\` in bash.

Every script with arguments accepts `-h` / `--help` (`mini_llm.py` has none: it is configured through its `CFG` dictionary). `train.py`, `finetune.py`, `rag_demo.py` and `generate_item_keywords.py` also print their docstring with examples in the help output.

Unless noted, the numbers in this document were measured on an RTX 3060 Laptop GPU (6 GB) with Ollama running locally; the `train.py` / `sample.py` numbers are from CPU.

---

## `train.py`

Pretrains a *decoder-only* language model with next-token prediction. Without `--data` it uses a synthetic corpus, which allows a *smoke test* with no downloads.

### Arguments

**Data and output**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data` | path | `None` | Training `.txt` file. If omitted, a 300,000-token synthetic corpus is used. |
| `--out` | path | `runs/default` | Output folder (created if missing). |
| `--vocab-size` | int | `1024` | With `--data`, size of the BPE vocabulary to train. Without `--data`, number of ids in the synthetic corpus. |
| `--tok-train-chars` | int | `500000` | Number of leading characters of the text used to *train* the tokenizer (the full text is encoded regardless). |

**Model** (mapped to [`ModelConfig`](module_reference.md#4-model) fields)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--d-model` | int | `256` | Residual dimension. |
| `--n-layers` | int | `4` | Number of blocks. |
| `--n-heads` | int | `8` | Query heads. |
| `--n-kv-heads` | int | `2` | K/V heads (GQA). Equal to `--n-heads` gives MHA and `1` gives MQA. |
| `--attn` | choice | `gqa` | `gqa`, `mla`, `swa`, `longformer`, `blocksparse` or `linear`. |
| `--hybrid-pattern` | str | `None` | Per-layer pattern, e.g. `"LLLG"`: `L` = the `--attn` attention, `G` = global GQA. |
| `--window` | int | `64` | Local window for `swa`, `longformer` and `blocksparse`. Only has an effect if it is smaller than `--block-size`. |
| `--qk-norm` | flag | off | Per-head RMSNorm on q and k. |
| `--moe` | flag | off | Mixture-of-Experts FFN in every layer (instead of dense SwiGLU). |
| `--n-experts` | int | `4` | Number of experts. |
| `--top-k` | int | `2` | Active experts per token. |
| `--moe-routing` | choice | `softmax` | `softmax` (auxiliary balancing loss) or `auxfree` (balancing bias, DeepSeek-V3). |

**Training** ([`TrainConfig`](module_reference.md#6-training) fields)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--steps` | int | `1000` | Optimization steps. |
| `--batch-size` | int | `32` | Sequences per micro-batch. |
| `--block-size` | int | `128` | Training context length. **Also sets the model's `max_seq_len`.** |
| `--lr` | float | `3e-3` | Peak LR (`max_lr`). |
| `--warmup` | int | `100` | Linear warmup steps. |
| `--schedule` | choice | `cosine` | `cosine` or `wsd` (Warmup-Stable-Decay). |
| `--grad-accum` | int | `1` | Micro-batches accumulated per step. |
| `--eval-every` | int | `100` | Validation and checkpoint frequency. |
| `--compile` | flag | off | Wraps the model in `torch.compile`. |
| `--seed` | int | `0` | Global seed. |
| `--device` | str | auto | `cpu`, `cuda`, `cuda:1`, `mps`… If omitted, picks `cuda` when available and `cpu` otherwise. |

Some `ModelConfig` fields are not exposed on the CLI and keep their defaults: `dropout=0`, `sandwich_norm`, the MLA sizes (`kv_lora_rank=32`, `qk_nope_dim=16`, `qk_rope_dim=8`, `v_dim=16`), `block=16`, `top_blocks=2`, `global_idx=(0,)`, `n_shared_experts=0`, `moe_every=1`, `aux_loss_coef=0.01`, `rope_scaling`. The same goes for these `TrainConfig` fields: `min_lr_ratio=0.1`, `weight_decay=0.1`, `grad_clip=1.0` and `eval_batches=10`. To vary them, use the Python API (see the notebooks).

### What it does, stage by stage

1. **Setup.** `seed_everything(--seed)`, picks the device and creates `--out`.
2. **Data.**
   - *With `--data`*: reads the file and trains a `ByteBPETokenizer` on the first `--tok-train-chars` characters (with `verbose=True`, it prints a line every 100 merges). Saves `<out>/tokenizer.json`, encodes **all** the text and prints the characters/token ratio. The effective vocabulary is the tokenizer's. The BPE is written in pure Python, so this step can take a while on a large text.
   - *Without `--data`*: `synthetic_structured_corpus(n_tokens=300_000, vocab=--vocab-size)`.
3. **Split**: contiguous 90 % train / 10 % val (`train_val_split`).
4. **Model.** Builds `ModelConfig` from the arguments with `max_seq_len = --block-size`, and prints total and active parameters (different only with `--moe`).
5. **Training loop** (`train_lm`). At each step it sets the LR from the schedule, runs `--grad-accum` micro-batches with `get_batch`, clips the gradient to norm 1.0 and applies AdamW (β = (0.9, 0.95), weight decay on matrices only). Every `--eval-every` steps, and at the last one, it estimates the validation loss over 10 batches, prints a log line and **saves a checkpoint** to `<out>/model.pt`. bf16 mixed precision turns on automatically on CUDA; on CPU training runs in fp32.
6. **Wrap-up.** Saves the final model (`model.save`), computes the perplexity over 20 random validation batches (`evaluation.perplexity`) and writes `<out>/train_log.json` with the model config, the training config, the validation curve, the final perplexity and the wall time.

**Log line format** (illustrative values):

```
step   100 | train 1.012 | val 1.004 | ppl     2.7 | lr 3.00e-03 | grad 0.41 | 47.3s
```

`grad` is the gradient norm **before** clipping. A sudden jump is usually the first sign of instability.

**Output files:**

| File | Contents |
|---|---|
| `<out>/model.pt` | `{"config": dict, "state_dict": ...}`, loadable with `TransformerLM.load`. |
| `<out>/tokenizer.json` | Only when `--data` was used: BPE merges + special tokens. |
| `<out>/train_log.json` | `config`, `train`, `val_losses` (`[[step, loss], ...]`), `final_val_ppl`, `wall_time_s`. |

### Examples

**Smoke test** (synthetic corpus, default model, CPU):

```powershell
.venv\Scripts\python scripts\train.py --steps 300 --out runs\smoke
```
```bash
.venv/bin/python scripts/train.py --steps 300 --out runs/smoke
```

**Measured on CPU:** **3.03M parameters**, final validation perplexity **2.65** in about **140 s**. Validation loss goes from 6.86 (step 0, ≈ ln 1024 = 6.93, i.e. chance) to 1.00 (step 100), 0.978 (step 200) and 0.974 (step 299).

Why such a low perplexity? The synthetic corpus is a lexicon of 60 "words" of 2 to 4 tokens separated by token 0. Only the first token of each word is genuinely uncertain (≈ 1 in 60); the rest of the word and the separator are determined. A model that memorizes the lexicon has little average uncertainty per token. The low perplexity shows that the *pipeline* learns, not that the model "knows language".

Where do the 3.03M parameters come from?

| Component | Computation | Parameters |
|---|---|---|
| Embedding (tied with `lm_head`) | 1024 × 256 | 262,144 |
| GQA attention per layer | wq 256·256 + wk 256·64 + wv 256·64 + wo 256·256 | 163,840 |
| SwiGLU per layer (`d_ff` = 688, Llama rule) | 3 · 256 · 688 | 528,384 |
| RMSNorm per layer | 2 · 256 | 512 |
| × 4 layers + final norm | | ≈ 2.77M + 256 |
| **Total** | | **≈ 3.03M** |

A faster variant is the one in the `Makefile` (`make smoke`):

```powershell
.venv\Scripts\python scripts\train.py --steps 100 --d-model 64 --n-layers 2 --out runs\smoke --eval-every 50
```
```bash
.venv/bin/python scripts/train.py --steps 100 --d-model 64 --n-layers 2 --out runs/smoke --eval-every 50
```

**Real text with GQA + MoE on GPU.** `data/input.txt` is not included in the repo: bring your own `.txt`, for example *tiny shakespeare*.

```powershell
.venv\Scripts\python scripts\train.py --data data\input.txt --vocab-size 2048 --d-model 384 --n-layers 6 --n-heads 6 --n-kv-heads 2 --moe --n-experts 4 --steps 3000 --batch-size 32 --block-size 256 --out runs\shakespeare
```
```bash
.venv/bin/python scripts/train.py --data data/input.txt --vocab-size 2048 --d-model 384 --n-layers 6 \
    --n-heads 6 --n-kv-heads 2 --moe --n-experts 4 --steps 3000 --batch-size 32 --block-size 256 \
    --out runs/shakespeare
```

**Compare architectures** with everything else held constant. Read the perplexities from each `train_log.json`.

```bash
.venv/bin/python scripts/train.py --steps 300 --attn mla                         --out runs/mla
.venv/bin/python scripts/train.py --steps 300 --attn swa --window 32             --out runs/swa32
.venv/bin/python scripts/train.py --steps 300 --attn linear --hybrid-pattern LLLG --out runs/hybrid
.venv/bin/python scripts/train.py --steps 300 --moe --moe-routing auxfree        --out runs/moe_auxfree
.venv/bin/python scripts/train.py --steps 300 --schedule wsd                     --out runs/wsd
```

On Windows, just replace `.venv/bin/python scripts/train.py` with `.venv\Scripts\python scripts\train.py`.

### Notes and pitfalls

- **`--window` must be smaller than `--block-size` to matter.** By default it is 64 against a `--block-size` of 128, so `--attn swa` already restricts attention. If you raise `--window` to `--block-size` (or beyond), the window covers the whole context and attention is equivalent to full causal attention.
- `--block-size` also limits later generation: `sample.py` cannot produce more than `max_seq_len` tokens in total.
- Only `gqa` and `mla` have a KV-cache. With `swa`, `longformer`, `blocksparse`, `linear` or a hybrid pattern that includes them, `sample.py` recomputes the whole sequence for each token and generation is slower.
- `--compile` needs a working compiler toolchain (Triton on GPU; a C++ compiler on CPU). On Windows and on CPU it often fails or is not worth it for such small models.
- The intermediate and final checkpoints are written to the same `<out>/model.pt`, so the latest save overwrites the previous ones.

---

## `sample.py`

Loads a `train.py` checkpoint and generates tokens, either with normal sampling (temperature, top-k, top-p, repetition penalty) or with **speculative decoding** using a *draft* model.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--run` | path | **required** | Folder with `model.pt` (and `tokenizer.json` if trained on text). |
| `--prompt` | str | `"\n"` | Initial text. **Ignored if the folder has no `tokenizer.json`** (see below). |
| `--max-new-tokens` | int | `200` | Maximum new tokens (also capped by `max_seq_len`). |
| `--temperature` | float | `0.8` | `<= 0` gives greedy decoding (argmax). |
| `--top-k` | int | `50` | Only sample among the k highest logits. |
| `--top-p` | float | `None` | *Nucleus sampling*: keep probability mass `p`. |
| `--repetition-penalty` | float | `1.0` | CTRL penalty on already-generated tokens (`1.0` = no penalty). |
| `--speculative` | path | `None` | Folder of a smaller *draft* model trained with the same tokenizer. Enables speculative decoding. |
| `--device` | str | auto | Same as in `train.py`. |

### What it does, stage by stage

1. **Load** `TransformerLM.load(<run>/model.pt)`. If `<run>/tokenizer.json` exists, it also loads the `ByteBPETokenizer`.
2. **Prompt.** With a tokenizer, `ids = tok.encode(--prompt)`. Without a tokenizer, or if the encoded prompt is empty (`--prompt ""`), `ids = [0]`: token 0, which in the synthetic corpus is the "word" separator.
3. **Generation.**
   - *Normal*: `model.generate(...)` with the sampling parameters. If every layer supports it (`gqa`/`mla`), it uses the KV-cache: one *prefill* of the prompt, then a single-token forward per step. It stops at `eos_id` (if there is a tokenizer), at `--max-new-tokens` or when `max_seq_len` is full.
   - *Speculative*: loads `<speculative>/model.pt` as the draft and calls `speculative_decode(target, draft, ids, max_new_tokens, k=4, temperature=max(T, 1e-3))`. Only the temperature is used: `--top-k`, `--top-p` and `--repetition-penalty` are ignored. It prints the **acceptance rate** and the **tokens per target pass**.
4. **Output.** Prints `[N tokens in X s = Y tok/s]` followed by the decoded text (with a tokenizer) or the **list of ids** (without one). The text includes the prompt.

### Why does the smoke test print numbers instead of text?

```powershell
.venv\Scripts\python scripts\sample.py --run runs\smoke --max-new-tokens 50
```
```bash
.venv/bin/python scripts/sample.py --run runs/smoke --max-new-tokens 50
```

**Measured on CPU:** about **200 tok/s**. The output is a list of integers shaped like

```
[50 tokens in 0.25s = 200.0 tok/s]

[0, 734, 12, 0, 455, 91, 803, 0, ...]
```

(the actual ids depend on sampling). This is expected, not a bug:

- `runs/smoke` was trained **without `--data`**, on the synthetic corpus. That corpus is made of integer ids with no text behind them, so `train.py` neither trains nor saves a tokenizer.
- Without `tokenizer.json`, `sample.py` cannot encode the prompt (so it ignores it and starts from `[0]`) or decode the output (so it prints the ids).
- What you *can* check is that the model learned the structure: groups of 2 to 4 ids repeat (the lexicon's "words"), separated by `0`.

To get text, train with `--data <file.txt>`: that produces `tokenizer.json` and `sample.py` decodes automatically.

### Examples

**Text with nucleus sampling and repetition penalty:**

```powershell
.venv\Scripts\python scripts\sample.py --run runs\shakespeare --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.1
```
```bash
.venv/bin/python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --max-new-tokens 200 \
    --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.1
```

**Greedy (deterministic):**

```bash
.venv/bin/python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --temperature 0
```

**Speculative decoding.** First train a small draft on **the same data with the same `--vocab-size`**. BPE is deterministic, so the resulting tokenizer is identical.

```powershell
.venv\Scripts\python scripts\train.py --data data\input.txt --vocab-size 2048 --d-model 128 --n-layers 2 --n-heads 4 --block-size 256 --steps 1500 --out runs\small_draft
.venv\Scripts\python scripts\sample.py --run runs\shakespeare --prompt "ROMEO:" --speculative runs\small_draft
```
```bash
.venv/bin/python scripts/train.py --data data/input.txt --vocab-size 2048 --d-model 128 --n-layers 2 \
    --n-heads 4 --block-size 256 --steps 1500 --out runs/small_draft
.venv/bin/python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --speculative runs/small_draft
```

The output looks like `[speculative] acceptance NN%, X.XX tokens/target pass` (no reference measurement in this repository). The closer the draft is to the target, the higher the acceptance and the more tokens you get per pass of the large model.

### Notes and pitfalls

- **Maximum length:** the total (prompt + generated) never exceeds `max_seq_len`, which is the training `--block-size`. With `runs/smoke` (128), the default `--max-new-tokens 200` stops at roughly 128 new tokens.
- **Speculative mode:** `--top-k`, `--top-p` and `--repetition-penalty` are **not applied**. Only the temperature is used, and `k = 4` proposals per round is hard-coded. The implementation recomputes without a cache and is batch-1, so on CPU it can be *slower* than normal generation with a KV-cache. Its purpose is educational: checking the acceptance rate and that the distribution is preserved.
- **Empty prompt:** `--prompt ""` (as in the README example) also works in a folder that has `tokenizer.json`: if the encoded prompt is empty, generation starts from id 0. Keep in mind that with a real BPE tokenizer, id 0 is the byte `\x00`, not a special token; for text, a prompt of at least one character is usually better (the default `"\n"` works).
- The tok/s figure includes the *prefill* and the first (warm-up) call. To measure TTFT and *throughput* rigorously, use `evaluation.benchmark_generation` (notebook 06).

---

## `mini_llm.py`

A **complete mini LLM in a single file** that depends only on PyTorch (it imports nothing from `llm_edu_suite`): byte-level BPE tokenizer → GPT with RoPE, RMSNorm and SwiGLU → training with AdamW and a cosine schedule → evaluation against the corpus's irreducible loss → sampling → checkpoint save and reload. It is notebook 00 (`notebooks/00_start_here/00_mini_llm_end_to_end.ipynb`) exported as a script: the whole pipeline at a glance, before the notebooks take each piece apart and the package scales it up.

> **Generated file.** `scripts/mini_llm.py` is produced from notebook 00 by its generator, `tools/notebooks/nb00_mini_llm_end_to_end.py`. To change it permanently, **edit the generator**, not the script: the next `make notebooks` overwrites it.

### Configuration

There is no `argparse`. Everything is controlled by the `CFG` dictionary at the top of the file:

| Key | Default | Description |
|---|---|---|
| `text_path` | `None` | A `.txt` file (e.g. *tiny shakespeare*). `None`, or a path that does not exist, uses the built-in synthetic corpus. |
| `vocab_size` | `512` | BPE vocabulary target (256 bytes + merges). Training stops early if there is nothing left to merge. |
| `block_size` | `128` | Context length in tokens. |
| `d_model`, `n_layers`, `n_heads` | `192`, `4`, `6` | Model size. |
| `dropout` | `0.0` | Attention dropout. |
| `batch_size`, `steps` | `32`, `800` | Sequences per batch and optimization steps. |
| `max_lr`, `min_lr`, `warmup` | `3e-3`, `3e-4`, `80` | Linear warmup, then cosine decay down to `min_lr`. |
| `weight_decay`, `grad_clip` | `0.1`, `1.0` | AdamW decay (matrices only) and gradient-norm clipping. |
| `eval_every`, `eval_batches` | `100`, `20` | Evaluation frequency and number of batches per estimate. |
| `seed` | `0` | Seed for `torch` and `random`. |

### What it does, stage by stage

1. **Corpus.** Without `text_path`, 30,000 synthetic sentences "*subject verb object place.*", each slot drawn from 8 options (e.g. "The old sailor found a golden key near the river.").
2. **Tokenizer.** A from-scratch byte-level BPE (GPT-2-style pre-tokenizer, merges applied in learned order) trained on the first 200,000 characters. It prints the number of merges, the real vocabulary size, the characters per token, a round-trip check and how a sample sentence is split.
3. **Data.** 90 / 10 contiguous split; random windows of `block_size` tokens with the target shifted by one (next-token prediction).
4. **Model.** `MiniGPT`: tied embeddings, pre-norm blocks with RMSNorm, causal attention with RoPE (`scaled_dot_product_attention`), a SwiGLU FFN (Llama's 8/3 rule rounded to 64), GPT-2 initialization with residual projections scaled by 1/√(2L). It prints the parameter count and the initial loss next to ln V (what a uniform guess would give).
5. **Training.** AdamW (β = (0.9, 0.95), decay on matrices only), warmup + cosine, clipping. A log line every `eval_every` steps: `step | train | val | ppl | lr | time`.
6. **Evaluation.** Final validation loss and perplexity. With the synthetic corpus it also computes the **irreducible loss**: each sentence carries exactly 4 · ln 8 nats of real uncertainty (four slots, eight choices each), so the floor per token is `4 · ln 8 · n_sentences / n_tokens`. It prints how far above that floor the model is. That is the honest yardstick: no model can go below it.
7. **Sampling.** Three generations from "The old sailor": greedy, T = 0.8 with top-k 40, and T = 1.0 with top-p 0.9.
8. **Checkpoint.** Saves `{config, state_dict, tokenizer merges}` to `mini_gpt.pt` in the system temporary folder (`tempfile.gettempdir()`), reloads it into a new model and tokenizer, and checks that logits and encodings are identical.

### Example

```powershell
.venv\Scripts\python scripts\mini_llm.py
```
```bash
.venv/bin/python scripts/mini_llm.py        # or: make mini-llm
```

On an NVIDIA GPU the script automatically trains in **bf16 autocast** (Ampere or newer) and lets matmuls use **TF32** (`torch.set_float32_matmul_precision("high")`); on CPU it runs in plain fp32. At the default 800 steps it takes about **3–5 minutes on CPU**. To try your own text, set `text_path` in `CFG` (the irreducible-loss line only appears with the synthetic corpus).

**Measured on the default 800-step run** (RTX 3060 Laptop GPU): **1.86M parameters**, vocabulary **455** after **199 BPE merges** (the synthetic corpus runs out of pairs to merge before reaching 512), **~22 s** of training, validation loss **0.778** (perplexity 2.18) against a corpus floor of **0.774** (perplexity 2.17): **0.6% above the floor**.

Where do the 1.86M parameters come from? Embedding (tied) 455 × 192 = 87,360; per layer, attention 4 · 192² = 147,456 + SwiGLU 3 · 192 · 512 = 294,912 + norms 384; × 4 layers + final norm ≈ 1.86M.

### Notes and pitfalls

- The vocabulary printed is the **real** one (`len(tok.vocab)`), which can be smaller than `vocab_size` on a small or repetitive corpus. The model is built with that real size.
- The checkpoint goes to the temporary folder, not to `runs/`. Copy it if you want to keep it.
- There is no KV-cache: generation recomputes the window (cropped to `block_size`) at every step. For the cached version, see [`model`](module_reference.md#4-model) and `sample.py`.

---

## `finetune.py`

Post-training with the Hugging Face stack (`transformers` + `peft` + `trl`) on the two small synthetic tasks of notebooks 07–08, using **SmolLM2-135M-Instruct** (downloaded from the Hugging Face Hub on first use, ~260 MB). One subcommand per technique:

| Subcommand | Task | What it does |
|---|---|---|
| `sft` | Booking → JSON extraction | LoRA (or full) SFT with TRL's `SFTTrainer` |
| `dpo` | Booking → JSON extraction | LoRA SFT first, then DPO (+ NLL by default) with TRL's `DPOTrainer` and a second LoRA adapter |
| `grpo` | Two-digit arithmetic | GRPO with TRL's `GRPOTrainer`, a LoRA adapter and a verifiable reward |

Every subcommand evaluates on a **held-out** set before and after training. The task builders, the scoring and the reward functions are the same ones notebooks 07–08 define inline, and `tests/test_finetune_hf.py` tests them.

### Arguments

```
finetune.py {sft,dpo,grpo} [options]
```

| Argument | Type | Default | Used by | Description |
|---|---|---|---|---|
| `stage` (positional) | choice | **required** | — | `sft`, `dpo` or `grpo`. |
| `--model` | str | `HuggingFaceTB/SmolLM2-135M-Instruct` | all | Any Hugging Face causal LM with a chat template. |
| `--full` | flag | off | `sft`, `dpo` | Full fine-tuning instead of LoRA for the SFT stage (needs more VRAM; uses the Adafactor optimizer). |
| `--rank` | int | `16` | all | LoRA rank `r`; `lora_alpha = 2 * rank`, `target_modules="all-linear"`. |
| `--n-train` | int | `300` | `sft`, `dpo` | SFT training examples. |
| `--epochs` | float | `2` | `sft`, `dpo` | SFT epochs. |
| `--lr` | float | `None` | all | Learning rate. Defaults: SFT `1e-3` (LoRA) or `1e-4` (`--full`); DPO `2e-4`; GRPO `1e-4`. In the `dpo` subcommand, `--lr` applies only to the DPO stage, and the SFT stage keeps its default. |
| `--nll-weight` | float | `1.0` | `dpo` | Weight of the SFT (NLL) loss on the chosen answer. `0` = plain DPO. |
| `--beta` | float | `0.1` | `dpo` | DPO β (how strongly the policy is tied to the reference). |
| `--reward` | choice | `exact` | `grpo` | `exact` (sparse: 1 if the last number in the reply is right) or `partial` (dense: up to 0.8 for being close, +0.2 for replying with only a number). |
| `--grpo-steps` | int | `40` | `grpo` | GRPO optimizer steps. |
| `--seed` | int | `0` | all | `transformers.set_seed`. |
| `--out` | path | `<temp dir>/llm_edu_finetune` | all | Trainer scratch directory. Nothing is saved there (`save_strategy="no"`). |

### What it does internally

**Common to all subcommands.** Loads the tokenizer (left padding, for batched generation) and the model in **bf16 on CUDA** or fp32 on CPU. Every trainer uses a micro-batch of 4 with 4 gradient-accumulation steps (16 sequences per step; GRPO uses 8 × 4), gradient checkpointing, logging every 5 steps, `report_to="none"` and no checkpoints. Evaluation uses greedy decoding with the chat template.

**`sft`**

1. **Data.** `booking_examples(--n-train, seed)` builds sentences such as *"We are six. Could we get a table at Casa Pepe on Friday around 8:30 pm? Booking for Ana."* with the target `{"name", "restaurant", "people" (int), "day" (lowercase), "time" (24-hour HH:MM)}`. Each one becomes a conversational **prompt-completion** row (system + user → assistant JSON). The 40 test examples use names and restaurants that never appear in training.
2. **Before.** Scores the base model: `valid` (the reply contains parseable JSON), `exact` (every field right) and `fields` (fraction of fields right).
3. **Training.** `SFTTrainer` with `SFTConfig(lr_scheduler_type="cosine", warmup_steps=5, max_length=256)` and a `LoraConfig` (or none with `--full`). With prompt-completion data, TRL computes the loss **only on the completion** (completion-only loss): the prompt tokens get label `-100`.
4. **After.** Merges the LoRA weights into the model (`merge_and_unload`) and scores it again.

**`dpo`**

1. Runs the `sft` stage above and prints the `SFT:` scores. With the defaults the SFT model is only partly trained, so DPO has room to help.
2. Builds 400 **preference pairs**: *chosen* = the right JSON, *rejected* = the same JSON with one realistic mistake (8 pm left as `08:00`, `people` as a string, a capitalized day, or the wrong name).
3. Trains a **new LoRA adapter on top of the merged SFT model** with `DPOTrainer`. The reference model is the same network with the adapter switched off, so no second copy is kept in memory. `loss_type=["sigmoid", "sft"]` with `loss_weights=[1, --nll-weight]` is DPO + NLL; with `--nll-weight 0` it is plain `["sigmoid"]`. One epoch, cosine schedule, 3 warmup steps.
4. Prints the log-probabilities of the chosen and rejected answers at the first and last logging step, then the scores after DPO. With `--nll-weight 0` you can see the chosen log-prob fall too (*likelihood displacement*).

**`grpo`** (starts from the base model, not from SFT)

1. **Data.** 2000 training prompts and 100 test prompts like *"What is 47 - 85?"* with the system prompt *"Answer with the number only."*.
2. **Before.** Exact accuracy of the base model on the test prompts.
3. **Training.** `GRPOTrainer` with a LoRA adapter: 8 completions per prompt (`num_generations=8`), 32 completions per optimizer step (4 prompts × 8), `max_completion_length=32`, `temperature=1.0`, KL coefficient `beta=0.04`, constant learning rate. The reward is `reward_exact` or `reward_partial`.
4. **After.** Exact accuracy again.

### Examples

```powershell
.venv\Scripts\python scripts\finetune.py sft                          # LoRA SFT
.venv\Scripts\python scripts\finetune.py sft --full                   # full fine-tuning
.venv\Scripts\python scripts\finetune.py dpo                          # SFT, then DPO + NLL
.venv\Scripts\python scripts\finetune.py dpo --nll-weight 0           # plain DPO: watch the chosen log-prob fall
.venv\Scripts\python scripts\finetune.py grpo --reward exact          # sparse, verifiable reward
.venv\Scripts\python scripts\finetune.py grpo --reward partial        # dense reward with partial credit
.venv\Scripts\python scripts\finetune.py grpo --grpo-steps 10         # a quick run
```
```bash
.venv/bin/python scripts/finetune.py sft
.venv/bin/python scripts/finetune.py sft --full
.venv/bin/python scripts/finetune.py dpo
.venv/bin/python scripts/finetune.py dpo --nll-weight 0
.venv/bin/python scripts/finetune.py grpo --reward exact
.venv/bin/python scripts/finetune.py grpo --reward partial
.venv/bin/python scripts/finetune.py grpo --grpo-steps 10
```

**Measured** (RTX 3060 Laptop GPU, 6 GB, Ollama models unloaded):

| Command | Before | After | Time |
|---|---|---|---|
| `finetune.py sft` | valid 0.00, exact 0.00, fields 0.00 | valid **1.00**, exact **0.375**, fields **0.855** | 73 s of training, 94 s in total |
| `finetune.py grpo --grpo-steps 10` | exact accuracy 0.04 | exact accuracy **0.42** | ~5 s per step, 69 s in total |

How to read them:

- **SFT.** The base model never answers with JSON for this prompt. After 38 optimizer steps on 300 examples every reply is valid JSON and 85 % of the fields are right, but only 37.5 % of the replies get *every* field right (time conversion is the hardest). That gap is what `dpo` targets.
- **GRPO.** Most of the jump in 10 steps comes from the format: the base model often reasons in words and runs out of its 32 tokens (75 % of the completions were cut off in the first five steps), and the reward teaches it to reply with just the number. In those first steps 65–70 % of the groups had the same reward for all 8 samples (`frac_reward_zero_std` in the logs): those groups give no gradient, which is the weakness of a sparse reward that `--reward partial` addresses.
- The full default runs (`dpo`, and `grpo` with 40 steps) were not timed for this document. Expect a few minutes each on a similar GPU, and much longer on CPU.

### Notes and pitfalls

- **Do not run it while Ollama holds the GPU** on a 6 GB card (see [installation, GPU notes](installation.md#gpu-notes)). Run `ollama stop qwen2.5:3b` first if you used Ollama in the last few minutes.
- `--full` changes only the SFT stage. DPO and GRPO always train a LoRA adapter.
- `--epochs` and `--n-train` have no effect on `grpo`; `--grpo-steps` and `--reward` have no effect on `sft` / `dpo`.
- On Windows, the first download may print a `huggingface_hub` warning about symlinks. It is harmless (see [troubleshooting](installation.md#troubleshooting)).
- Nothing is saved: the script is a demonstration that prints scores. To keep a model, see notebook 07, which saves the LoRA adapter with `save_pretrained`, reloads it and merges it.

---

## `rag_demo.py`

RAG over a folder of `.md` / `.txt` files with **LangChain**: hybrid retrieval (FAISS dense search + BM25, fused with Reciprocal Rank Fusion by an `EnsembleRetriever`) and an answer with numbered citations. It is the pipeline of notebook 10 packaged as a command-line tool. The models come from `llm_edu_suite.config` (Ollama `qwen2.5:3b` and `nomic-embed-text` by default; edit `.env` to switch).

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--docs` | path | `data/knowledge_base` | Folder with `.md` / `.txt` files (top level only, not recursive). |
| `--ask` | str | `None` | A question to answer with citations. |
| `--eval` | path (optional value) | `None` | An eval JSON file. Given **without** a value, it uses `data/eval/rag_eval.json`. |
| `--ragas` | flag | off | With `--eval`: also generate an answer for each question and score them with RAGAS. Slow: the judge is `config.judge_model()` (`qwen2.5:7b` if pulled, otherwise the chat model). |
| `--retriever` | choice | `hybrid` | `hybrid`, `dense` or `bm25`: the retriever the chain (and RAGAS) uses. |
| `--k` | int | `4` | Chunks per answer. |
| `--chunk-size` | int | `800` | Characters per chunk; the overlap is `chunk_size // 8`. |

`--ask` and `--eval` can be combined. With neither, the script only indexes and prints the summary line.

### What it does internally

1. **Setup.** `config.setup(project="llm-edu-suite-rag-demo")` loads `.env` (so with a LangSmith key every run is traced to that project). If the provider is Ollama and the server does not answer, the script exits with a message saying how to start it.
2. **Loading.** Every top-level `*.md` and `*.txt` in `--docs` becomes a LangChain `Document` with the file name in `metadata["source"]`.
3. **Splitting.** `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=chunk_size // 8, add_start_index=True)`.
4. **Indexing.**
   - *Dense*: `FAISS` over `get_embeddings()`, embedded in batches of 64 (one huge request can overload a local embedding server).
   - *Lexical*: `BM25Retriever` with a lower-cased word tokenizer (the default splits on whitespace only, so `Llama-3?` would not match `llama-3`).
   - *Hybrid*: `EnsembleRetriever([bm25, dense], weights=[0.5, 0.5])` (weighted RRF), cut to the top `k`.
   It prints `indexed N files -> M chunks in X s | retriever ... | llm ...`.
5. **The chain** (LCEL): `RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=prompt | llm | StrOutputParser())`. The documents travel with the answer, so they can be cited. The system prompt asks for one to three sentences, citations like `[1]` or `[2][3]`, and "I don't know" when the context does not contain the answer.
6. **`--ask`.** Prints `ANSWER` and then `SOURCES` as `[n]* file: first 100 characters`; the `*` marks the passages the answer actually cited.
7. **`--eval`.** Scores **all three retrievers** (whatever `--retriever` says) on every question:
   - *File level*: `file_hit@1`, `file_hit@k` and `file_mrr`: is a chunk from a correct source file in the top 1 / top k, and the mean inverse rank of the first one.
   - *Evidence level* (items with an `evidence` list): `evidence_hit@1`, `evidence_hit@k`, `evidence_mrr`: does a retrieved chunk actually contain one of the evidence strings? This is stricter: the right file is not enough if the chunk is the wrong part of it.
8. **`--ragas`.** Runs the chain with `--retriever` on every question (`batch`, 4 at a time), strips the `[n]` markers so the judge does not count them as unsupported claims, and scores the answers with four RAGAS metrics: **faithfulness**, **answer relevancy**, **context precision** and **context recall**. The judge is the configured chat model, called through Ollama's OpenAI-compatible endpoint (`OLLAMA_BASE_URL + /v1`). A metric that fails on one item is skipped (NaN) instead of stopping the run. Before importing ragas, the script registers an empty stand-in for `langchain_community.chat_models.vertexai`, a module that ragas 0.4.3 imports and langchain-community 0.4 removed.

### Eval file format

```json
[
  {"question": "What does Reciprocal Rank Fusion sum to combine several rankings?",
   "answer": "For each document RRF sums 1/(k + rank) over the rankings, with k = 60.",
   "sources": ["glossary.md", "module_reference.md"],
   "evidence": ["1/(60 + rank)"]}
]
```

The source can be given as `sources` (a list), `source` or `doc` (a single file name), and must match the file names in `--docs`. `answer` is the reference answer (needed by context precision and context recall); `evidence` is optional.

The repository includes two corpora:

- `data/knowledge_base/` (the default): five long documents about the course itself, with `data/eval/rag_eval.json` (20 questions with sources and evidence strings).
- `data/sample_docs/`: four short documents (`refund_policy.md`, `security.md`, `shipping.md`, `support.md`), with `data/sample_eval.json` (6 questions in the `doc` format).

### Examples

**Ask a question** (default corpus, hybrid retrieval):

```powershell
.venv\Scripts\python scripts\rag_demo.py --ask "Why must an MCP stdio server never print to stdout?"
```
```bash
.venv/bin/python scripts/rag_demo.py --ask "Why must an MCP stdio server never print to stdout?"
```

Measured output (~16 s, of which ~6 s indexing 358 chunks):

```
indexed 5 files -> 358 chunks in 6.4s | retriever hybrid | llm qwen2.5:3b

ANSWER: Because stdout is reserved for the protocol and a single stray `print()` corrupts the stream and the client fails to parse the next message [1].

SOURCES:
  [1]* module_reference.md: '- *stdio framing*. On the stdio transport the host launches the server as a subprocess, and each mes'
  [2]  cli_scripts.md: '- **Never print to stdout.** In an stdio server, stdout carries the protocol; a single stray `print('
  ...
```

**Retrieval evaluation** (fast, no generation):

```powershell
.venv\Scripts\python scripts\rag_demo.py --eval
```
```bash
.venv/bin/python scripts/rag_demo.py --eval
```

Measured (20 questions, `k = 4`, ~17 s in total):

| Retriever | file hit@1 | file hit@4 | file MRR | evidence hit@1 | evidence hit@4 | evidence MRR |
|---|---|---|---|---|---|---|
| dense | 0.95 | 1.00 | 0.97 | 0.65 | 0.85 | 0.73 |
| bm25 | 0.90 | 1.00 | 0.94 | 0.70 | **0.95** | 0.80 |
| **hybrid** | 0.95 | 1.00 | 0.97 | **0.80** | 0.90 | **0.84** |

At file level all three are near perfect: with five documents, finding the right *file* is easy. The evidence columns are the informative ones: the hybrid retriever puts a chunk that really contains the answer first in 80 % of the questions, against 65 % for dense search alone.

**Retrieval + RAGAS** on the small sample corpus:

```powershell
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --eval data\sample_eval.json --ragas
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json --ragas
```

Measured (6 questions, judge `qwen2.5:3b`, RAGAS part 47 s, ~55 s in total): retrieval 1.00 everywhere (each sample document is a single chunk, so with `k = 4` all four are always retrieved), **faithfulness 1.00, answer relevancy 0.75, context precision 1.00, context recall 1.00**. On the 20-question default set `--ragas` takes considerably longer, because every question needs one generation and several judge calls.

**Other options:**

```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?" --retriever dense
.venv/bin/python scripts/rag_demo.py --docs ./my_docs --ask "..." --k 6 --chunk-size 1200
```

### Notes and pitfalls

- The answers come from a 3B model running locally: they are usually right on this corpus, but they can paraphrase, cite the wrong passage number or over-answer (the dense `--ask` example above on `sample_docs` returned the whole refund document). This is what the RAGAS metrics are for.
- `--ragas` assumes Ollama for the judge, even if `LLM_PROVIDER` is `openai` or `anthropic`.
- A 3B judge is noisy (pull `qwen2.5:7b` to get the stronger judge; the measurement above used the 3B). Treat RAGAS scores from it as a relative signal (compare retrievers or chunk sizes with the same judge), not as absolute quality.
- The FAISS index lives in memory and is rebuilt on every run (a few seconds for this corpus).
- `langchain-community` prints a sunset warning for `FAISS` / `BM25Retriever`; the script filters it.

---

## `mcp_server.py`

An **MCP server** built with the official MCP Python SDK (`mcp.server.fastmcp.FastMCP`). Its name is **`llm-edu-docs`** and it exposes the small knowledge base in `data/sample_docs/` (or any folder of `.md` / `.txt` files) to any MCP client: Claude Code, Claude Desktop, the MCP Inspector, a Python client from the SDK, or a LangChain agent through `langchain-mcp-adapters`. It needs neither PyTorch nor Ollama.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--docs` | path | `data/sample_docs` | Folder with the `.md` / `.txt` files to expose (top level only). |
| `--http` | flag | off | Serve **Streamable HTTP** instead of stdio. |
| `--host` | str | `127.0.0.1` | HTTP host (only with `--http`). |
| `--port` | int | `8000` | HTTP port (only with `--http`). The endpoint is `http://<host>:<port>/mcp`. |
| `--log-level` | str | `WARNING` | Log level on stderr. `INFO` logs every request. |
| `--require-token` | flag | off | With `--http`: every request needs `Authorization: Bearer <token>`, with the token read from the `MCP_AUTH_TOKEN` environment variable (never from the command line, where other users can read it). Without the variable the server refuses to start. See [Requiring a bearer token](#requiring-a-bearer-token). |

### What it exposes

| Primitive | Name | Description |
|---|---|---|
| tool | `search_docs(query, k=3)` | BM25 search over the docs (`rank_bm25`). Each non-heading line of the sample docs is one self-contained sentence, so each line is one passage; the file name is added to the passage for matching. Returns the best `k` (clamped to 1–10) as `[file] passage` lines, or `no results`. Annotated `readOnlyHint: true`. |
| tool | `calculate(expression)` | Safe arithmetic: the expression is parsed with `ast` and only numbers and `+ - * / ** %` and parentheses are evaluated; anything else is a tool error (`isError: true`). Annotated `readOnlyHint: true`. |
| tool | `create_ticket(title, priority="normal")` | A simulated write action (`priority` ∈ `low`, `normal`, `high`). Returns the ticket (`id`, `title`, `priority`, `created`) as JSON. Annotated `destructiveHint: true`, so a well-behaved host **asks for confirmation**. |
| resource | `docs://index` | JSON list of every document with its URI and title. |
| resource template | `docs://{name}` | The full text of one document, e.g. `docs://shipping.md`. |
| prompt | `answer_with_sources(question)` | A RAG prompt: use `search_docs`, answer only from the passages, cite file names, say so if the answer is not there. |

The server also sends `instructions` in its `initialize` response (*"Use search_docs before answering policy questions and cite the file names."*). The SDK derives each tool's JSON Schema from its type hints and its description from the docstring.

**Measured** with an SDK client over stdio (`ClientSession` + `stdio_client`): the whole session (start the subprocess, initialize, list tools / resources / templates / prompts, call the three tools, read a resource) takes **about 2 s**. `search_docs("express shipping", k=2)` returns the two `shipping.md` lines; `calculate("(17 * 23) + 4")` returns `395`; `calculate("__import__('os')")` returns a tool error (`only numbers and + - * / ** % ( ) are allowed`). The server reports protocol version `2025-11-25` with SDK 1.30.0, and with `--http` it answers `initialize` on `/mcp` as an event stream with an `mcp-session-id` header.

### Running it

In stdio mode (the default) the server waits for JSON-RPC messages on stdin and prints nothing until a client connects; stop it with Ctrl+C.

```powershell
.venv\Scripts\python scripts\mcp_server.py                                # stdio
.venv\Scripts\python scripts\mcp_server.py --docs C:\path\to\your\docs
.venv\Scripts\python scripts\mcp_server.py --http --port 8000             # http://127.0.0.1:8000/mcp
```
```bash
.venv/bin/python scripts/mcp_server.py                                    # or: make mcp-server
.venv/bin/python scripts/mcp_server.py --docs path/to/your/docs
.venv/bin/python scripts/mcp_server.py --http --port 8000 --log-level INFO
```

### Connecting it to Claude Code

```bash
claude mcp add llm-edu-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py
```

Everything after `--` is the command Claude Code launches. Use the project's interpreter if the `python` on your `PATH` does not have the `mcp` and `rank_bm25` packages:

```powershell
claude mcp add llm-edu-docs -- D:\path\to\llm-engineering-suite\.venv\Scripts\python.exe D:\path\to\llm-engineering-suite\scripts\mcp_server.py
```
```bash
claude mcp add llm-edu-docs -- /path/to/llm-engineering-suite/.venv/bin/python /path/to/llm-engineering-suite/scripts/mcp_server.py
```

Then `claude mcp list` shows the server and its status, and `/mcp` inside a Claude Code session lists its tools. Ask, for example, *"How long does express shipping take? Use the llm-edu-docs server."*

To connect to the HTTP mode instead, start the server with `--http` and run `claude mcp add --transport http llm-edu-docs http://127.0.0.1:8000/mcp`.

### Connecting it to Claude Desktop

Add the server to `claude_desktop_config.json` (*Settings → Developer → Edit Config*), with absolute paths, and restart Claude Desktop:

```json
{
  "mcpServers": {
    "llm-edu-docs": {
      "command": "/ABSOLUTE/PATH/.venv/bin/python",
      "args": ["/ABSOLUTE/PATH/scripts/mcp_server.py"]
    }
  }
}
```

On Windows use `"command": "D:\\path\\to\\llm-engineering-suite\\.venv\\Scripts\\python.exe"` (backslashes doubled in JSON).

### Testing it with the MCP Inspector

The official web UI for trying servers (needs Node.js):

```bash
npx @modelcontextprotocol/inspector python scripts/mcp_server.py
```

It opens a browser page where you can list and call the tools, read the resources and render the prompt.

### From Python

With the official SDK client:

```python
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["scripts/mcp_server.py"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        res = await session.call_tool("search_docs", {"query": "express shipping", "k": 1})
        print(res.content[0].text)          # [shipping.md] Express shipping delivers in 1 to 2 business days ...

asyncio.run(main())
```

As LangChain tools for an agent, with `langchain-mcp-adapters`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({"llm-edu-docs": {"transport": "stdio", "command": sys.executable,
                                                "args": ["scripts/mcp_server.py"]}})
tools = await client.get_tools()            # search_docs, calculate, create_ticket as LangChain tools
```

Notebook 16 walks through both, and builds an agent on top of the tools with LangChain's `create_agent`.

### Requiring a bearer token

With `--http --require-token` the server acts as an OAuth 2.1 **resource server**, the role MCP gives remote servers. A request without the token, or with a wrong one, gets `401` with a `WWW-Authenticate: Bearer ... resource_metadata="..."` header; that URL (`/.well-known/oauth-protected-resource/mcp`) serves the protected-resource metadata of RFC 9728, which names the authorization server and the required scope (`docs:read`). With the right token, everything works as over stdio.

```powershell
$env:MCP_AUTH_TOKEN = "pick-a-long-random-token"
.venv\Scripts\python scripts\mcp_server.py --http --port 8000 --require-token
```
```bash
MCP_AUTH_TOKEN=pick-a-long-random-token .venv/bin/python scripts/mcp_server.py --http --port 8000 --require-token
claude mcp add --transport http llm-edu-docs http://127.0.0.1:8000/mcp --header "Authorization: Bearer pick-a-long-random-token"
```

From Python, pass the header to the SDK's HTTP client: `streamablehttp_client("http://127.0.0.1:8000/mcp", headers={"Authorization": f"Bearer {token}"})`. To reuse the server elsewhere, `build_server(docs_dir, log_level, token=..., host=..., port=...)` builds it with authentication; the check itself is the `StaticTokenVerifier` class, which compares one shared token in constant time.

A shared token is a teaching stand-in. A real deployment puts an authorization server in front (Auth0, Entra ID, Keycloak, Okta...) and validates short-lived JWTs or calls its introspection endpoint, checking the audience, and adds TLS, per-user scopes inside each tool, rate limits and `Origin` validation. Notebook 16, section 9, shows the whole flow on the wire. `tests/test_mcp_sdk.py` checks that requests without the token or with a wrong one get 401, that the right token works, and that `--require-token` refuses to start without `MCP_AUTH_TOKEN`.

### Notes and pitfalls

- **Never print to stdout.** In an stdio server, stdout carries the protocol; a single stray `print()` corrupts the stream and the client fails to parse the next message. Everything the server logs (the HTTP address, created tickets) goes to `sys.stderr`.
- Use **absolute paths** in the Claude Code and Claude Desktop configuration: the host launches the process from its own working directory.
- Tickets live in memory and disappear when the process ends.
- `FastMCP` here is the 1.x SDK class. The course pins `mcp<2` because `langchain-mcp-adapters` does not support the 2.x SDK yet; in 2.x the same class is `mcp.server.mcpserver.MCPServer` and the decorators work the same way.
- Inside Jupyter, an stdio client needs a workaround because Jupyter's `sys.stderr` has no OS file handle; notebook 16 already applies it. Plain scripts do not need it.

## `generate_item_keywords.py`

Uses an LLM to write a keyword profile for every MovieLens-1M movie. Notebook 12 uses the keywords as content text (for cold start) and as a ranker feature, and measures whether they help instead of assuming it (spoiler: on this dataset, they do not). It is the offline "LLM as feature generator" pattern: it runs once per catalog, not on every request.

The output is already in the repository, in `data/recsys/ml1m_llm_keywords.json` (3,706 of 3,706 movies), so the notebook does **not** need this script. You only need it to regenerate them, for example with another model or prompt.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `qwen2.5:7b` | Ollama model that writes the keywords. |
| `--out` | path | `data/recsys/ml1m_llm_keywords.json` | Output (and resume) JSON file. |
| `--batch` | int | `40` | Movies per batch; the file is saved after each batch. |
| `--concurrency` | int | `4` | Concurrent requests to Ollama within a batch. |

### What it does internally

1. **Data.** Loads MovieLens-1M with `recsys.load_movielens_1m(recsys.download_movielens_1m())`, which downloads it into `data/cache/` on the first run (6 MB; see [installation](installation.md#data-downloaded-at-run-time)).
2. **Resuming.** If `--out` exists, it reads it and skips the movies that already have a profile. You can stop it with Ctrl+C and start it again: at most you lose the batch in progress.
3. **Prompt.** For each movie, its title and genres, and a request for 8 to 12 comma-separated keywords (subgenre, tone, themes, setting, era, audience) with **no names of people**: small models invent directors and actors with complete confidence, and a wrong name in a profile is worse than none.
4. **Model.** `config.get_llm(model=..., num_predict=48, num_ctx=512)`: a short context window keeps the 7B model entirely on a 6 GB GPU (the prompts are about 60 tokens). Calls go through `llm.batch(..., config={"max_concurrency": ...})`.
5. **Output.** A `{title: "keyword, keyword, ..."}` dictionary sorted by title, in UTF-8, saved after every batch, with a `done/total seconds` progress line.

### Examples

```powershell
.venv\Scripts\python scripts\generate_item_keywords.py                          # qwen2.5:7b, every movie
.venv\Scripts\python scripts\generate_item_keywords.py --model qwen2.5:3b --out data\cache\kw_3b.json
```

```bash
.venv/bin/python scripts/generate_item_keywords.py                              # qwen2.5:7b, every movie
.venv/bin/python scripts/generate_item_keywords.py --model qwen2.5:3b --out data/cache/kw_3b.json
```

With `qwen2.5:7b` it takes about **an hour** on a 6 GB laptop GPU. `qwen2.5:3b` is faster and invents more details. Writing to another path (`--out`) avoids overwriting the file the notebook uses.

### Notes and pitfalls

- It needs Ollama running and the model pulled (`ollama pull qwen2.5:7b`, 4.7 GB).
- The JSON keys are the **titles** exactly as in `movies.dat`; the notebook joins on title.
- Keywords can be wrong for obscure films, and sometimes switch language: in the committed file, *The Ref* (1994) is described as a high-school sports film, with three keywords in Chinese. That is why the notebook measures before using them, and validation ended up rejecting them.
- Do not run it at the same time as a training notebook on a 6 GB GPU (see [installation](installation.md#gpu-notes)).

---

## `harness_run.py`

Runs the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) coding-agent harness on one task folder, with a local model through Ollama and a sandboxed bash environment, and writes the whole trajectory to a JSON file. Notebook 27 calls it once per task; you can also run it by hand on your own task folders.

Unlike the other scripts, it runs with the Python of the **harness environment**, `.venv-harness`, not the course's `.venv`, because mini-swe-agent's terminal UI needs a newer `rich` than the course environment allows. See [installation](installation.md#the-agent-harness-environment-notebook-27) for how to create it, for Git Bash on Windows and for the three models.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `task_dir` | path (positional) | — | Task folder. It must contain `TASK.md`, the instructions given to the agent; every command runs inside this folder. |
| `model` | str (positional) | — | Ollama model name, e.g. `qwen2.5:3b` (the script adds the `ollama_chat/` prefix for litellm). |
| `out_json` | path (positional) | — | Where to write the trajectory and its statistics. |
| `--steps` | int | `20` | Step limit: model calls before the run ends with `LimitsExceeded`. Notebook 27 uses 15. |
| `--agents-md` | flag | off | Append the folder's `AGENTS.md` to the task, the way harnesses load project memory files. Ignored if there is no `AGENTS.md`. |
| `--temperature` | float | `0.0` | Sampling temperature. Notebook 27 uses 0.3 for the sampled `AGENTS.md` runs. |
| `--seed` | int | `0` | Sampling seed passed to Ollama. |

Two environment variables change where it looks: `OLLAMA_BASE_URL` (default `http://localhost:11434`) and `HARNESS_BASH`, the full path of the bash executable (default: `C:\Program Files\Git\bin\bash.exe` on Windows, the `bash` on the `PATH` elsewhere).

### What it does internally

1. **Configuration.** Loads mini-swe-agent's own `mini_textbased.yaml` (the system and task templates, and the text-based action format: one fenced `mswea_bash_command` block per reply) and overrides the limits: `--steps` steps, no cost limit, 300 s of wall time.
2. **Model.** `LitellmTextbasedModel` with `ollama_chat/<model>`, the given temperature and seed, a 16k context (`num_ctx=16384`), at most 1,024 output tokens per reply (small models can ramble forever) and a 120 s timeout per call. Cost tracking is set to ignore errors, because local models have no price.
3. **Task.** Reads `TASK.md`; with `--agents-md`, appends `Project instructions (from AGENTS.md):` and the file.
4. **Environment.** `SandboxedBashEnvironment`, the script's own class, with the three methods mini-swe-agent expects from an environment (`execute`, `get_template_vars`, `serialize`):
   - every command runs as `bash -c` **inside the task folder**, with a 60 s timeout;
   - the environment is **scrubbed**: only `SYSTEMROOT`, `PATH`, `TEMP`, `TMP`, `HOME`, `USERPROFILE` and `LANG` are kept (plus `PYTHONIOENCODING=utf-8` and `cat` as pager), so no API key reaches the agent;
   - output is cut to its last 8,000 characters;
   - a **deny list** of regular expressions refuses network access (`curl`, `wget`, `ssh`, `scp`, `nc`, `Invoke-WebRequest`), `git push`, recursive deletes of `/`, `~` or `..`, reading secrets (`.env`, `id_rsa`, `.ssh/`, `printenv`, a bare `env`, `/proc/self/environ`) and `sudo` / `su`. A refused command is not run: the model gets `Permission denied by the harness policy (<reason>)` with return code 126, and the reason is recorded;
   - when a command's output starts with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and it succeeded, the run ends as `Submitted`.
5. **Run.** `DefaultAgent.run(task)`. Any exception is recorded as a failed run (`error: <type>: <message>`) instead of crashing the caller.
6. **Output.** `out_json` gets `exit_status` (`Submitted`, `LimitsExceeded`, `TimeExceeded`, `RepeatedFormatError` or `error: ...`), `steps`, `seconds`, `format_errors`, `denied` (the reasons of every refused command), `prompt_tokens`, `completion_tokens` and `messages` (role and content of every message, each cut to 4,000 characters). The same statistics, without the messages, are printed to stdout as one JSON line.

### Examples

A task folder needs only a `TASK.md` and the code to work on:

```text
my_task/
├── TASK.md        "The tests in test_stats.py fail. Fix the bug in stats.py so that `python -m pytest -q` passes. Do not change the tests."
├── stats.py
└── test_stats.py
```

```powershell
.venv-harness\Scripts\python scripts\harness_run.py my_task qwen2.5:3b my_task_run.json --steps 15
.venv-harness\Scripts\python scripts\harness_run.py my_task qwen2.5:7b run_memory.json --agents-md --temperature 0.3 --seed 1
```

```bash
.venv-harness/bin/python scripts/harness_run.py my_task qwen2.5:3b my_task_run.json --steps 15
.venv-harness/bin/python scripts/harness_run.py my_task qwen2.5:7b run_memory.json --agents-md --temperature 0.3 --seed 1
```

In notebook 27 (10 tasks, 15-step limit, temperature 0) a run took 64 s on average with `qwen2.5:3b`, 82 s with `qwen2.5:7b` and 216 s with `freehuntx/qwen3-coder:8b`, which at a 16k context runs about 46 % on the CPU of a 6 GB card.

### Notes and pitfalls

- **The agent edits the task folder in place.** Run it on a copy, never on a repository you care about: the sandbox is a teaching layer, like notebook 14's, not a security boundary. Production harnesses add OS-level isolation (containers, Seatbelt, Landlock, restricted Windows tokens).
- **`Submitted` does not mean solved.** It only means the model declared itself done; in notebook 27, 11 of 30 submitted runs failed the hidden tests. Check the result with your own tests afterwards.
- **`python` inside the agent's shell** is whatever is first on the `PATH`. Notebook 27 puts the harness environment's `Scripts` (or `bin`) folder first so that `python -m pytest` works; do the same, or activate `.venv-harness`, when a task asks the agent to run tests.
- The deny list matches command text with regular expressions: it is easy to get around (a script file, an alias, `python -c`) and it also refuses harmless commands that happen to match. It shows where a permission policy plugs into the loop, not how strong one should be.
- On start, mini-swe-agent prints its version and `Loading global config from ...` (its own `.env` under the user's application-data folder). That file is mini-swe-agent's, not the course's; the script does not need anything in it.
- One run at a time: the script makes one Ollama call per step, and two GPU jobs at once on a 6 GB card can make Ollama hang (see [installation](installation.md#gpu-notes)).

---

## `inspect_tasks/gsm8k_cot.py`

An [Inspect AI](https://inspect.aisi.org.uk) task with notebook 20's GSM8K evaluation: the same 80 test problems, the same chain-of-thought system message, and two scorers on the same generations, notebook 20's "last number" rule and Inspect's built-in numeric match. Notebook 29 runs it to check that the framework reproduces the hand-written loop before trusting it with anything else.

It is not run with `python`: Inspect loads the file through its `inspect eval` command, which lives in its own environment, `.venv-eco`, because Inspect needs a newer `rich` than the course environment allows. See [installation](installation.md#the-inspect-and-phoenix-environment-notebook-29) for how to create it.

### What it defines

| Piece | What it is |
|---|---|
| `COT` | The system message: `Solve the problem step by step, then give the final answer as a number on the last line.` |
| `load(n)` | The first `n` problems of GSM8K's test split (`openai/gsm8k`, config `main`) after `shuffle(seed=0)`, notebook 20's sample, as a `MemoryDataset` of `Sample(input=question, target=final number, id=i)`. |
| `final_number(text)` | The last number in the answer, after removing `$` and thousands separators (and anything before `</think>`, for reasoning models). |
| `last_number()` | A `@scorer` with the metrics `accuracy()` and `stderr()`: correct when the last number equals the reference to within 1e-6. It stores the extracted number as the score's `answer`, so the log shows what was read. |
| `gsm8k_cot(n=80)` | The `@task`: solver `[system_message(COT), generate()]`, scorers `[last_number(), match(location="end", numeric=True)]`. |

### Arguments

The task has one argument, passed with `-T`: `n` (int, default `80`), the number of problems. Everything else is an option of `inspect eval`; these are the ones notebook 29 uses (`inspect eval --help` lists them all):

| Option | In notebook 29 | What it does |
|---|---|---|
| `--model` | `ollama/qwen2.5:3b` | Inspect's `provider/model` id. |
| `--model-base-url` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint. |
| `--temperature` | `0` (greedy run), `0.7` (sampled run) | Sampling temperature. |
| `--max-tokens` | `512` | Maximum output tokens per answer. |
| `--max-connections` | `1` | Concurrent requests to the model. One, because a single Ollama on a 6 GB card serves one request at a time anyway. |
| `--epochs` | `3` (sampled run) | Runs every sample N times; the metrics are reduced over epochs (the mean by default) and every epoch stays in the log. |
| `--seed` | `0` (sampled run) | Random seed for generation. |
| `--log-format` | `json` | `json` (readable with `json.load`, which is what the notebook does) or Inspect's default `eval` format. |
| `--display` | `none` | No progress display, because the notebook runs it as a subprocess. |
| `--limit` | not used | Evaluate only some samples, for example `--limit 10`. |
| `--log-dir` / `INSPECT_LOG_DIR` | `INSPECT_LOG_DIR=data/cache/inspect_logs` | Where the logs go. |

### Examples

**Windows (PowerShell):**

```powershell
# the greedy run of notebook 29
.venv-eco\Scripts\inspect eval scripts\inspect_tasks\gsm8k_cot.py --model ollama/qwen2.5:3b `
    --model-base-url http://localhost:11434/v1 --temperature 0 --max-tokens 512 --max-connections 1
# three sampled epochs
.venv-eco\Scripts\inspect eval scripts\inspect_tasks\gsm8k_cot.py --model ollama/qwen2.5:3b `
    --model-base-url http://localhost:11434/v1 --temperature 0.7 --epochs 3 --seed 0 --max-tokens 512 --max-connections 1
# a quick check on 5 problems
.venv-eco\Scripts\inspect eval scripts\inspect_tasks\gsm8k_cot.py --model ollama/qwen2.5:3b `
    --model-base-url http://localhost:11434/v1 --temperature 0 --max-tokens 512 --max-connections 1 -T n=5
.venv-eco\Scripts\inspect view          # browse the logs in the browser
```

**Linux / macOS:**

```bash
.venv-eco/bin/inspect eval scripts/inspect_tasks/gsm8k_cot.py --model ollama/qwen2.5:3b \
    --model-base-url http://localhost:11434/v1 --temperature 0 --max-tokens 512 --max-connections 1
.venv-eco/bin/inspect eval scripts/inspect_tasks/gsm8k_cot.py --model ollama/qwen2.5:3b \
    --model-base-url http://localhost:11434/v1 --temperature 0.7 --epochs 3 --seed 0 --max-tokens 512 --max-connections 1
.venv-eco/bin/inspect view
```

In notebook 29 the greedy run took 307 s for the 80 problems and scored **0.850** with `last_number` (stderr 0.040), exactly notebook 20's 85.0 %, and 0.800 with `match` (stderr 0.045); its JSON log is 1.9 MB. The three sampled epochs took 918 s and scored 0.812 on average (stderr 0.043), with a 5.5 MB log.

### Notes and pitfalls

- **The two scorers grade the same generations, and they disagree.** On 4 of the 80 greedy answers, all ending in a LaTeX `\boxed{}`, the built-in `match(numeric=True)` read another number from the text and scored them wrong: 80.0 % against 85.0 %. Report which scorer a number comes from.
- **Run it from `.venv-eco`, not from `.venv`.** `inspect-ai` is not installed in the course environment.
- The first run downloads GSM8K's test split into the Hugging Face cache, through `.venv-eco`'s `datasets`.
- Keep `--max-connections 1` on a 6 GB card, and do not run it next to another GPU job: two Ollama jobs at once can make Ollama hang (see [installation](installation.md#gpu-notes)).
- Inspect records the generation settings in the log (model, temperature, max tokens, connections), so two runs can be compared afterwards without guessing how each was made.

---

## Tools and Makefile targets

**`tools/build_notebooks.py`** rebuilds the `.ipynb` files from their generators in `tools/notebooks/nbNN_*.py` (notebook 00's generator also writes `scripts/mini_llm.py`). Building **clears the outputs**.

```bash
python tools/build_notebooks.py                       # every notebook
python tools/build_notebooks.py 05 14                 # only some, by number
python tools/build_notebooks.py --keep-outputs 05     # prose-only edits: keep the executed outputs
```

### `tools/build_notebooks.py --keep-outputs`

Rebuilding from scratch means re-running the notebook, and some take a long time (12 about 25 minutes, 10 about 19). With `--keep-outputs`, the generator looks in the existing `.ipynb` for a code cell with exactly the same source (it matches by content, not by position, so inserting prose cells breaks nothing) and, when it finds one, **keeps its outputs** (text, tables and plots) and its execution count. Markdown cells are always rewritten; a code cell whose source changed comes out empty, and the script prints how many need a re-run (`N code cell(s) changed and need a re-run`) or `no code changed`. It also keeps the kernel metadata, so an unchanged notebook produces no diff. This is what you want when you only edited prose, for example when adding a theory section:

```bash
python tools/build_notebooks.py --keep-outputs 10 12    # rewrite the prose of 10 and 12 without losing the run
```

Use it only for changes that do not alter any result. If you change code, re-run the whole notebook with `tools/run_notebooks.py`: keeping the outputs of unchanged cells whose inputs did change (because an earlier cell is different) would leave outputs the code no longer produces.

**`tools/run_notebooks.py`** executes notebooks headlessly and **in place** (they keep their outputs), each with its own folder as the working directory, as Jupyter does. It searches `notebooks/` **recursively**, so it finds the notebooks in every topic folder. Arguments can be file names or number prefixes:

```bash
python tools/run_notebooks.py                       # every notebook, in number order
python tools/run_notebooks.py 05_pretraining.ipynb  # by file name, wherever it lives
python tools/run_notebooks.py 05 10 14              # by number (prefix)
```

It prints `[seconds] name: OK | FAIL` per notebook and `N/M notebooks OK` at the end, with a 1-hour timeout per notebook, and exits with code 1 if any fails, so it can be used in CI. An argument that matches zero or several notebooks stops with an error. Notebooks 10–16 need Ollama running; on a 6 GB GPU, avoid running the training notebooks (07–09) while Ollama has a model loaded (see [installation](installation.md#gpu-notes)).

**Makefile targets** (Linux/macOS, or Git Bash with `make`):

| Target | Command | What it does |
|---|---|---|
| `make setup` | `bash setup_env.sh` | Creates `.venv` and installs everything (on Windows, run `setup_env.ps1`). |
| `make test` | `pytest -q` | All tests. The 4 `ollama` tests skip themselves if Ollama is not running. Use `pytest -q -m "not ollama"` for the fast set only (90 tests, ~42 s). |
| `make notebooks` | `python tools/build_notebooks.py` | Rebuilds every `.ipynb` (and `scripts/mini_llm.py`) from the generators. **Clears outputs.** |
| `make run-notebooks` | `python tools/run_notebooks.py` | Executes every notebook in place. |
| `make smoke` | `train.py --steps 100 --d-model 64 --n-layers 2 ...` | A short pre-training smoke test. |
| `make mini-llm` | `python scripts/mini_llm.py` | Trains the self-contained mini GPT (notebook 00 as a script). |
| `make mcp-server` | `python scripts/mcp_server.py` | Starts the MCP server on stdio. |

---

## Summary of measured results

| Command | Hardware | Result |
|---|---|---|
| `train.py --steps 300 --out runs/smoke` | CPU | 3.03M parameters, validation perplexity **2.65**, ≈ 140 s |
| `sample.py --run runs/smoke --max-new-tokens 50` | CPU | ≈ **200 tok/s** (output as ids, because the corpus is synthetic) |
| `mini_llm.py` (800 default steps) | RTX 3060 Laptop GPU (~3–5 min on CPU) | 1.86M parameters, vocabulary 455 (199 merges), ~22 s of training, validation loss **0.778** vs a corpus floor of **0.774** (0.6 % above) |
| `finetune.py sft` | RTX 3060 Laptop GPU | exact 0.00 → **0.375**, fields 0.00 → **0.855**, valid JSON 0.00 → **1.00**; 94 s |
| `finetune.py grpo --grpo-steps 10` | RTX 3060 Laptop GPU | exact accuracy 0.04 → **0.42**; ~5 s per step, 69 s |
| `rag_demo.py --eval` | RTX 3060 Laptop GPU + Ollama | hybrid: file hit@1 **0.95**, evidence hit@1 **0.80**, evidence MRR **0.84** (dense: 0.65 / 0.73); ~17 s |
| `rag_demo.py --docs data/sample_docs --eval data/sample_eval.json --ragas` | RTX 3060 Laptop GPU + Ollama | faithfulness **1.00**, answer relevancy **0.75**, context precision **1.00**, context recall **1.00**; ~55 s |
| `rag_demo.py --ask "..."` | RTX 3060 Laptop GPU + Ollama | ~10–16 s including indexing |
| `mcp_server.py` via an SDK stdio client | CPU | full session (start, initialize, list everything, 4 tool calls, 1 resource read) ≈ 2 s |
| `pytest -q -m "not ollama"` / `pytest -q -m ollama` | RTX 3060 Laptop GPU + Ollama | 90 passed in ~42 s / 4 passed in ~40 s |
| `harness_run.py TASK MODEL OUT --steps 15` (through notebook 27, 10 tasks per model) | RTX 3060 Laptop GPU + Ollama | hidden tests passed: `qwen2.5:3b` **30 %**, `qwen2.5:7b` 20 %, `freehuntx/qwen3-coder:8b` 20 %; 64 / 82 / 216 s per task; 11 of 30 runs submitted as done while failing the hidden tests |
| `inspect eval scripts/inspect_tasks/gsm8k_cot.py --model ollama/qwen2.5:3b --temperature 0 ...` (through notebook 29) | RTX 3060 Laptop GPU + Ollama | 80 problems in 307 s; `last_number` **85.0 %** (the same as notebook 20's loop), built-in `match` 80.0 %; with `--temperature 0.7 --epochs 3`, 81.2 % in 918 s |

For the teaching context of each stage, see the [notebooks guide](notebooks_guide.md). For an end-to-end walkthrough with interpretation of the results, see the [case study](case_study.md).
