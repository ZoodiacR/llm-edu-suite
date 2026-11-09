🇪🇸 [Versión en español](../es/scripts_cli.md)

# Command-line scripts

> A guide to the six scripts in `scripts/` (plus `tools/run_notebooks.py` and the `Makefile` targets): every real argument (taken from `argparse`), what each stage does internally, usage examples and results measured on CPU.
>
> Related documents: [README](README.md) · [Installation](installation.md) · [Module reference](module_reference.md) · [Notebooks guide](notebooks_guide.md) · [Case study](case_study.md) · [Glossary](glossary.md)

| Script | What it is for | Modules it uses | Needs `torch` |
|---|---|---|---|
| [`train.py`](#trainpy) | Pretrain a `TransformerLM` on a `.txt` file or a synthetic corpus | `data`, `tokenizer`, `model`, `training`, `evaluation`, `utils` | yes |
| [`sample.py`](#samplepy) | Generate from a checkpoint (normal or speculative sampling) | `model`, `tokenizer`, `inference`, `utils` | yes |
| [`finetune.py`](#finetunepy) | Post-training demo: SFT (full or LoRA) → DPO and GRPO, with evaluation | `data`, `model`, `training`, `finetune`, `evaluation`, `utils` | yes |
| [`rag_demo.py`](#rag_demopy) | Index a folder of documents, answer questions and evaluate the RAG | `rag`, `llm_client`, `evaluation` | no |
| [`mini_llm.py`](#mini_llmpy) | A complete mini GPT in one file: BPE → model → training → evaluation → sampling → checkpoint | none (self-contained) | yes |
| [`mcp_server.py`](#mcp_serverpy) | A real MCP server over stdio that exposes `data/sample_docs` to Claude Code, Claude Desktop or any MCP client | `mcp`, `rag` | no |

---

## Before you start

Run every command from the **repository root** with the project's virtual-environment interpreter (see [installation](installation.md)). Each script adds the repo root to `sys.path`, so there is no need to install the package with `pip install -e .`.

**Windows (PowerShell or cmd):**

```powershell
.venv\Scripts\python scripts\train.py --steps 300 --out runs\smoke
```

**Linux / macOS:**

```bash
.venv/bin/python scripts/train.py --steps 300 --out runs/smoke
```

Throughout this document each example is shown in both forms. When a long command is split across lines, the continuation character is `` ` `` in PowerShell, `^` in cmd and `\` in bash.

Every script with arguments accepts `-h` / `--help` (`mini_llm.py` has none: it is configured through its `CFG` dictionary). `train.py` also shows its docstring with examples in the help output.

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

## `finetune.py`

An end-to-end *post-training* demo on the **"reverse the sequence"** toy task: given `[5, 6, 7]`, the assistant must answer `[7, 6, 5]`. It shows how *exact match* and reply perplexity change after each technique.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--steps` | int | `900` | SFT steps. 900 leaves the model *partially* trained, with room for DPO and GRPO to improve it. |
| `--lora` | flag | off | SFT with LoRA (r = 8, α = 16) **+ trainable embeddings** (`train_embeddings=True`) instead of full fine-tuning. |
| `--dpo-steps` | int | `300` | DPO steps. `0` disables the stage. |
| `--nll-weight` | float | `1.0` | Weight of the NLL anchor on the chosen answer (DPO + NLL / RPO). `0` = plain DPO, to observe *likelihood displacement*. |
| `--grpo-iters` | int | `150` | GRPO iterations. `0` disables the stage. |
| `--device` | str | auto | Same as in `train.py`. |

All other hyperparameters are fixed in the script (see the next table).

### What it does, stage by stage

| Stage | What happens | Fixed hyperparameters |
|---|---|---|
| 0. Format | `V = 512`; `ChatFormat.for_vocab(512)` reserves ids 507–511 as `SYS, USER, ASSIST, END, PAD`. | — |
| 1. Base model | Pretrains a `TransformerLM` (d_model 128, 4 layers, 8 heads, 2 KV, `max_seq_len` 128, ≈ 0.76M parameters) for **200 steps** on a 100,000-token synthetic corpus with vocabulary `V − 5`, so the role ids are never used. No logging. | batch 32, block 64, lr 3e-3, warmup 20 |
| 2. Data | **8000** SFT training examples (`seed=0`) and 200 test examples (`seed=99`) for the reply perplexity. Separately, 150 evaluation cases generated with another seed (length 2–5, tokens 1–400). | system = `[10]` |
| 3. `base` evaluation | `exact_match_rate` with `greedy_answer` over the 150 cases + `perplexity_on_examples` on the test set. | — |
| 4. SFT | The base is copied. *Without `--lora`*: `train_sft` on all parameters. *With `--lora`*: `apply_lora(r=8, alpha=16, train_embeddings=True)` and `train_sft` on `trainable_parameters` (adapters + `tok_emb`/tied `lm_head`: the role tokens never appeared in pretraining, so their embeddings must be learned). Then `merge_lora` and **unfreezes** every weight so DPO and GRPO can update the merged model. Evaluated. Logs every 300 steps. | full: lr 1e-3; LoRA: lr 2e-3; batch 32 |
| 5. DPO (if `--dpo-steps > 0`) | `train_dpo` from the SFT model (the reference is a frozen copy) with **3000** pairs (`seed=1`) and `nll_weight=--nll-weight`. *Chosen* = correct reversal, *rejected* = correct except for the last token (replaced by a different one). Evaluated as `SFT+DPO+NLL` (or `SFT+DPO` with `--nll-weight 0`). Logs every 100 steps (`loss`, `acc`, `margin`). | lr 5e-5, β 0.1, batch 16 |
| 6. GRPO (if `--grpo-iters > 0`) | `train_grpo` on **a copy of the SFT model** (not the DPO one). Each iteration: 8 prompts (length 2–5, tokens 1–400) × 8 sampled responses. **Dense, partial-credit** reward: fraction of positions that match the reversal, `− 0.1 ×` the length difference, and `+0.2` if it ends with `END`. Evaluated as `SFT+GRPO`. Logs every 50 iterations (`mean reward`, `pg`, `kl`). | lr 1e-4, β (KL) 0.04, max_new 8 |

```
base --SFT (full | LoRA)--> sft --DPO(+NLL)--> sft+dpo
                                +--GRPO-------> sft+grpo     (two independent branches off SFT)
```

Why a dense reward? With a binary reward (1 only if the reversal is exact), a partially trained model often gets the same reward on all 8 responses of a group, the relative advantage is 0 and GRPO receives no gradient. Per-position partial credit separates "almost right" responses from bad ones and gives signal in nearly every group.

**Sample output** (format only; the numbers depend on the machine and the arguments):

```
base model 758.91K params on cpu
base           exact match   0.0%  reply ppl   ...
sft step    0  loss ...
SFT (LoRA)     exact match  ...%  reply ppl   ...
dpo step    0  loss ...  acc 0.00  margin +0.00
SFT+DPO+NLL    exact match  ...%  reply ppl   ...
grpo iter   0  mean reward ...  pg ...  kl 0.0000
SFT+GRPO       exact match  ...%  reply ppl   ...
```

At the first DPO step, policy and reference are identical, so `margin` is 0 and `acc` is 0. With `--nll-weight 0` the loss is exactly `0.693` (= ln 2), a good sanity check; with the default (`1.0`) it is ln 2 plus the NLL of the chosen answer.

There are no measured numbers for this script with the current defaults yet.

### Examples

**What the README suggests** (LoRA SFT → DPO and GRPO, short version):

```powershell
.venv\Scripts\python scripts\finetune.py --lora --dpo-steps 200 --grpo-iters 40
```
```bash
.venv/bin/python scripts/finetune.py --lora --dpo-steps 200 --grpo-iters 40
```

**The full default recipe** (full SFT → DPO+NLL and GRPO; about 5 minutes on CPU according to the docstring):

```powershell
.venv\Scripts\python scripts\finetune.py
```
```bash
.venv/bin/python scripts/finetune.py
```

**Full SFT only** (the fastest run):

```bash
.venv/bin/python scripts/finetune.py --dpo-steps 0 --grpo-iters 0
```

**Compare LoRA and full fine-tuning** on the same budget:

```powershell
.venv\Scripts\python scripts\finetune.py --dpo-steps 0 --grpo-iters 0
.venv\Scripts\python scripts\finetune.py --dpo-steps 0 --grpo-iters 0 --lora
```

**See *likelihood displacement*** with plain DPO (no NLL anchor) and compare with the default output:

```powershell
.venv\Scripts\python scripts\finetune.py --nll-weight 0 --grpo-iters 0
```
```bash
.venv/bin/python scripts/finetune.py --nll-weight 0 --grpo-iters 0
```

### Notes and pitfalls

- **DPO and GRPO are not chained.** They are two independent branches off the SFT model. The output reports `SFT+DPO(+NLL)` and `SFT+GRPO`, not `SFT+DPO+GRPO`. To chain them, pass the model returned by `train_dpo` to `train_grpo` (see [notebook 08](notebooks_guide.md)).
- **Why DPO + NLL by default?** In this task the chosen and rejected answers differ by a single token. Plain DPO can widen the margin by pushing down the probability of *both* answers (*likelihood displacement*), and *exact match* gets worse even as `acc` rises. The NLL anchor (`--nll-weight 1.0`) keeps the probability of the correct answer high. See [reference §7.3](module_reference.md#7-finetune).
- **Why `train_embeddings=True` with `--lora`?** Ids 507–511 (role tokens) never appear in the pretraining corpus, so their embeddings remain random. LoRA only modifies the linear projections and cannot fix them; that is why the script also trains `tok_emb` (and the tied `lm_head`), like `modules_to_save` in `peft`.
- This is the heaviest script on CPU, especially GRPO: each iteration generates 64 responses token by token without a KV-cache. Lower `--grpo-iters` to iterate quickly.
- In this implementation GRPO takes a single update per sampled batch ($\mu = 1$), so the importance *ratio* is 1 and the *clip* never kicks in (see [reference, §7.4](module_reference.md#7-finetune)).
- The 150 evaluation cases are drawn at random from the same token range as training. They measure whether the *skill* (reversing) was learned, not whether specific examples were memorized, because the chance of exactly matching a training example is negligible.

---

## `rag_demo.py`

Indexes a folder of `.txt` / `.md` documents, answers questions with **hybrid retrieval (dense + BM25 + RRF)** and an LLM, and optionally evaluates retrieval and generation against a set of reference questions. **Does not need PyTorch.** Without `--model` it uses `MockLLM`, which is offline and deterministic.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--docs` | path | **required** | Folder with `.txt` and/or `.md` files (top level only, not recursive). |
| `--ask` | str | `None` | Question to answer: shows the sources and the answer. |
| `--eval` | path | `None` | JSON with a list of `{"question", "doc", "answer"}`. |
| `--k` | int | `4` | Retrieved chunks placed in the prompt. |
| `--chunk-size` | int | `120` | Chunk size in words. The overlap is `chunk_size // 4`. |
| `--model` | str | `None` | Model name on an OpenAI-compatible server. If omitted, `MockLLM` is used. |
| `--base-url` | str | `http://localhost:8000/v1` | Server URL (vLLM by default; Ollama: `http://localhost:11434/v1`). |

`--ask` and `--eval` can be passed together. If neither is passed, the script only indexes and prints the summary.

### What it does, stage by stage

1. **Load.** Finds `*.txt` and `*.md` in `--docs`. Each document's `doc_id` is its **file name** (without the path).
2. **Chunking.** `chunk_documents(docs, size=--chunk-size, overlap=--chunk-size // 4)`: overlapping word windows.
3. **Index.** `Retriever(chunks)` with the default configuration: a `HashedTfidfEmbedder` fitted on the chunks and a `VectorIndex` (exact cosine) for the dense side, plus `BM25` for the lexical side, no *reranker*.
4. **LLM.** Uses `OpenAICompatibleClient(--model, base_url=--base-url)` if `--model` is given; otherwise `MockLLM()`. Prints `indexed N docs -> M chunks | llm = ...`.
5. **`--ask`.** `retriever.retrieve(q, k=--k)` (dense top-20 and BM25 top-20, RRF fusion, top-k) → `build_rag_prompt` (numbered context + citation instructions) → `llm.complete`. Shows `SOURCES` (`[n] doc#chunk: first 100 characters...`) and `ANSWER`.
6. **`--eval`.** For each question:
   - *Retrieval*: the ordered list of **unique** `doc_id`s in the top-k against the relevant set `{doc}`. Aggregated with `evaluate_ranking(ks=(1, 3))`, which gives `recall@1`, `hit@1`, `ndcg@1`, `recall@3`, `hit@3`, `ndcg@3` and `mrr`.
   - *Generation*: `token_f1(answer, reference answer)` and `context_support(answer, retrieved context)`, a faithfulness proxy that measures how much of the answer is backed by the context.
   - Prints the retrieval metrics (rounded to 3 decimals) and the mean token-F1 and *context support*.

### Examples with the sample documents

The repository ships `data/sample_docs/` (four short documents: `refund_policy.md`, `security.md`, `shipping.md`, `support.md`) and `data/sample_eval.json` (six questions with their document and reference answer).

**Ask a question:**

```powershell
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --ask "What is the refund policy?"
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?"
```

**Evaluate:**

```powershell
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --eval data\sample_eval.json
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json
```

**Measured (with `MockLLM`):** `recall@1 = 1.0`, `MRR = 1.0`, `token-F1 = 0.887`.

How to read these numbers:

- **recall@1 = 1.0 and MRR = 1.0**: for all 6 questions, the right document comes *first*. Each sample document has about 60 words (fewer than `--chunk-size 120`), so it forms a single chunk and the index holds 4. With `--k 4` all four documents are always retrieved, so `recall@3` is trivially 1. The informative metric here is the **rank** (recall@1, MRR, nDCG).
- **token-F1 = 0.887 (not 1.0)**: `MockLLM` answers with the context sentence that shares the most words with the question, followed by the citation `[n]`. That sentence usually matches the reference almost exactly, but the citation adds a token and some sentences are longer than the expected answer (e.g. "… 8am to 8pm *Central European Time*."). Both effects lower the F1 precision.
- With a real LLM, F1 usually *drops* because the model paraphrases. That is why it pays to complement it with `context_support` and an LLM judge ([notebook 10](notebooks_guide.md)).

**Using your own documents:**

```bash
.venv/bin/python scripts/rag_demo.py --docs ./my_docs --ask "What is the return policy?" --k 3 --chunk-size 200
```

The `--eval` format is:

```json
[
  {"question": "How long does express shipping take?", "doc": "shipping.md",
   "answer": "Express shipping delivers in 1 to 2 business days."}
]
```

`doc` must match the file name inside `--docs` exactly. `answer` is optional: if missing, that question's F1 is 0.

**With a real LLM** through an OpenAI-compatible server. Requires `pip install requests`, which is commented out in `requirements.txt` and included in the `rag` extra of `pyproject.toml`.

```powershell
# vLLM (in another terminal / a GPU machine):  vllm serve Qwen/Qwen2.5-7B-Instruct
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --eval data\sample_eval.json --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

# Ollama:  ollama serve  (and first: ollama pull llama3.1)
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --ask "Are gift cards refundable?" --model llama3.1 --base-url http://localhost:11434/v1
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json \
    --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
```

For a remote provider, the key is read from the `OPENAI_API_KEY` environment variable. For example, `$env:OPENAI_API_KEY="sk-..."` in PowerShell or `export OPENAI_API_KEY=sk-...` in bash, with `--base-url https://api.openai.com/v1`.

### Notes and pitfalls

- The folder in the README example (`./my_docs`) **does not exist** in the repository. Use `data/sample_docs` or your own folder.
- The default embedder (`HashedTfidfEmbedder`) is lexical: it does not understand synonyms or other languages. A question in Spanish over English documents will retrieve poorly. For real embeddings, use `SentenceTransformerEmbedder` from Python (see [reference §10](module_reference.md#10-rag)); the script does not expose it on the CLI.
- `MockLLM` looks for the sentence with the most lexical overlap; it does not reason. It is there to check that the *pipeline* (retrieve → prompt → cite → evaluate) works end to end, not to measure generation quality.

---

## `mini_llm.py`

A **complete mini LLM in a single file** that depends only on PyTorch (it imports nothing from `llm_edu_suite`): byte-level BPE tokenizer → GPT with RoPE, RMSNorm and SwiGLU → training with AdamW and a cosine schedule → evaluation against the corpus's irreducible loss → sampling → checkpoint save and reload. It is notebook 00 (`notebooks/00_start_here/00_mini_llm_end_to_end.ipynb`) exported as a script: the whole pipeline at a glance, before the notebooks take each piece apart and the package scales it up.

> **Generated file.** `scripts/mini_llm.py` is produced from notebook 00 by `tools/build_notebooks_part3.py`. To change it permanently, **edit the generator**, not the script: the next `make notebooks` overwrites it.

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

## `mcp_server.py`

A real, runnable **MCP server over stdio**, built with [`llm_edu_suite.mcp`](module_reference.md#18-mcp) and no SDK. It exposes the sample knowledge base in `data/sample_docs/` (or any folder of `.md` / `.txt` files) to any MCP client: Claude Code, Claude Desktop, the MCP Inspector, or `MCPClient` + `StdioTransport` from Python. It does not need PyTorch.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--docs` | path | `data/sample_docs` | Folder with `.md` / `.txt` files to expose (top level only). |

### What it exposes

| Primitive | Name | Description |
|---|---|---|
| tool | `search_docs(query, k=3)` | Hybrid BM25 + dense retrieval (`rag.Retriever`, 80-word chunks with 20 of overlap). Returns `[file] passage` lines. `readOnlyHint: true`. |
| tool | `calculate(expression)` | Safe arithmetic: only digits, `+ - * / ( ) .` and spaces are accepted; anything else is a tool error. `readOnlyHint: true`. |
| tool | `create_ticket(title, priority="normal")` | Simulated write action (`priority` ∈ `low \| normal \| high`); returns the ticket as JSON (`structuredContent` too). Annotated `destructiveHint: true`, so a well-behaved client should **ask for confirmation**. |
| resources | `docs://<file>` | One resource per file (`text/markdown`), described by its first line. |
| prompt | `answer_with_sources(question)` | A RAG template: use `search_docs`, answer only from the passages, cite the file names, say so if the answer is not there. |

The server also sends `instructions` in its `initialize` response ("use `search_docs` before answering policy questions and cite the file names").

### Running and connecting it

The server waits for JSON-RPC messages on stdin (Ctrl+C to stop). A startup line goes to **stderr**: `[mcp_server] 3 tools, 4 resources, 1 prompts — waiting on stdin`.

```powershell
.venv\Scripts\python scripts\mcp_server.py
.venv\Scripts\python scripts\mcp_server.py --docs C:\path\to\your\docs
```
```bash
.venv/bin/python scripts/mcp_server.py          # or: make mcp-server
.venv/bin/python scripts/mcp_server.py --docs path/to/your/docs
```

**Claude Code:**

```bash
claude mcp add llm_edu_suite-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{"mcpServers": {"llm_edu_suite-docs": {"command": "python", "args": ["/ABSOLUTE/PATH/scripts/mcp_server.py"]}}}
```

**MCP Inspector** (official web UI for testing servers; needs Node.js):

```bash
npx @modelcontextprotocol/inspector python scripts/mcp_server.py
```

**From Python**, with the package's own client:

```python
import sys
from llm_edu_suite.mcp import MCPClient, StdioTransport, result_text

client = MCPClient(StdioTransport([sys.executable, "scripts/mcp_server.py"]))
client.initialize()
print(result_text(client.call_tool("search_docs", {"query": "express shipping", "k": 1})))   # [shipping.md] ...
print(result_text(client.call_tool("calculate", {"expression": "6*7"})))                     # 42
client.close()
```

### Notes and pitfalls

- **Never print to stdout.** In an stdio server, stdout carries the protocol; a single stray `print()` corrupts the stream and the client fails to parse the next message. Everything the server logs (startup, created tickets) goes to `sys.stderr`.
- Use **absolute paths** in the Claude Code and Claude Desktop configuration: the host launches the process from its own working directory. Use the interpreter of the project's virtual environment (e.g. `/ABSOLUTE/PATH/.venv/bin/python` or `.venv\Scripts\python.exe`) if `python` on the `PATH` does not have the dependencies (NumPy).
- Tickets live in memory and disappear when the process ends. `create_ticket` is a stand-in for a real write.
- The retriever is lexical (`HashedTfidfEmbedder` + BM25), as in [`rag_demo.py`](#rag_demopy): it does not understand synonyms or other languages.

---

## Tools and Makefile targets

**`tools/run_notebooks.py`** runs notebooks headlessly and **in place** (they keep their outputs), each with its own folder as the working directory, as Jupyter does. It searches `notebooks/` **recursively**, so it finds every notebook in the topic folders. Arguments can be file names or number prefixes:

```bash
python tools/run_notebooks.py                       # every notebook, in number order
python tools/run_notebooks.py 05_pretraining.ipynb  # by file name, wherever it lives
python tools/run_notebooks.py 05 10 14              # by number (prefix)
```

It prints `[seconds] name: OK | FAIL` per notebook and `N/M notebooks OK` at the end, and exits with code 1 if any fails, so it can be used in CI. An argument that matches zero or several notebooks stops the run with an error.

**Makefile targets** (Linux/macOS or Git Bash with `make`):

| Target | Command | What it does |
|---|---|---|
| `make setup` | `bash setup_env.sh` | Creates `.venv` and installs everything (on Windows, run `setup_env.ps1`). |
| `make test` | `pytest -q` | Unit and behavioural tests (~15 s). |
| `make notebooks` | `build_notebooks_part1.py`, `part2`, `part3` | Regenerates every `.ipynb` (and `scripts/mini_llm.py`) from the generators. **Clears outputs.** |
| `make run-notebooks` | `python tools/run_notebooks.py` | Runs every notebook in place (~40 min on CPU). |
| `make smoke` | `train.py --steps 100 --d-model 64 --n-layers 2 ...` | 30-second pretraining smoke test. |
| `make mini-llm` | `python scripts/mini_llm.py` | Trains the self-contained mini GPT (notebook 00 as a script). |
| `make mcp-server` | `python scripts/mcp_server.py` | Starts the demo MCP server on stdio. |

---

## Summary of measured results (CPU unless noted)

| Command | Result |
|---|---|
| `train.py --steps 300 --out runs/smoke` | 3.03M parameters, validation perplexity **2.65**, ≈ 140 s |
| `sample.py --run runs/smoke --max-new-tokens 50` | ≈ **200 tok/s** (output as ids because the corpus is synthetic) |
| `rag_demo.py --docs data/sample_docs --eval data/sample_eval.json` | recall@1 **1.0**, MRR **1.0**, token-F1 **0.887** (`MockLLM`) |
| `mini_llm.py` (default 800 steps, RTX 3060 Laptop GPU; ~3–5 min on CPU) | 1.86M parameters, vocabulary 455 (199 merges), ~22 s of training, validation loss **0.778** vs corpus floor **0.774** (0.6% above) |

For the teaching context of each stage, see the [notebooks guide](notebooks_guide.md). For an end-to-end walkthrough with interpretation of the results, see the [case study](case_study.md).
