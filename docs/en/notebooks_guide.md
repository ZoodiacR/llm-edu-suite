# Notebooks guide

🇪🇸 [Versión en español](../es/guia_notebooks.md) · [← Back to the index](README.md)

The 21 notebooks one by one: what you build, which tools you use, which concepts you take away, what gets measured and what results come out.

The course has two halves. **Notebooks 00–06** build a language model from scratch in PyTorch, with the course's own package `llm_edu_suite`, so you see every piece. **Notebooks 07–16** do the same jobs with the frameworks the industry uses: Hugging Face `transformers`, `peft`, `trl`, `bitsandbytes` and `optimum-quanto` on SmolLM2 (07–09), LangChain, LangGraph and LangSmith (10–15), and the official MCP SDK (16). The local model for 10–16 is `qwen2.5:3b` served by Ollama, with `nomic-embed-text` for embeddings. **Notebook 17** (encoders and encoder-decoders) mixes both halves and is read right after 04, before 05: it closes the architecture topic with the two transformer families that are not decoder-only. **Notebook 18** (data engineering) is read right before 05: deduplication, quality filtering, decontamination and synthetic data on a real FineWeb sample, and two `TransformerLM`s trained on raw and clean data to see what changes. **Notebook 19** (systems) is read right after 05: it measures on the course's GPU where the memory and the time of a training run go (weights, gradients, Adam and activations, bf16, activation checkpointing, attention kernels, profiling, CUDA graphs and MFU) and how training is split across several GPUs (real DDP in two processes and a ZeRO / FSDP calculator). **Notebook 20** (reasoning and test-time compute) is read right after 08: on GSM8K with `qwen2.5:3b` it measures what chain of thought, self-consistency, best-of-N with a verifier and a distilled reasoning model (DeepSeek-R1-Distill 1.5B) buy in accuracy and cost in tokens, and how far a pairwise LLM judge can be trusted.

**How to use this guide.** Read the entry before you open the notebook, run it cell by cell, then come back and compare your numbers with the ones here. Every number below comes from the run committed to the repository (a laptop with a 6 GB RTX 3060). Your numbers will move a little: small models are sensitive to each machine's arithmetic, and anything an LLM generates or judges changes between runs. Where a result is fragile, the notebook says so, and so does this guide. Every notebook also has theory sections ("Background: ...") before the code that uses them; the long version of that theory, with the equations, is in the [theoretical foundations](foundations.md).

**What each block needs**
- **00–06**: PyTorch and `llm_edu_suite` only. They also run on CPU.
- **07–09**: `transformers`, `peft`, `trl`, `bitsandbytes` (QLoRA in 07) and `optimum-quanto`. The first run downloads SmolLM2 from the Hugging Face Hub. Each notebook caps its own VRAM (`VRAM_BUDGET_GB`) so it fits on the GPU next to Ollama: 2 GB by default in 07, which trains eleven models and whose 4-bit model leaves ~0.1 GB behind, and 1.5 GB in 08 and 09.
- **10–16**: **Ollama must be running** with `qwen2.5:3b` and `nomic-embed-text` pulled. The first cell (`setup()` from `llm_edu_suite.config`) checks this. Notebook 10 also uses `qwen2.5:7b` as the RAGAS judge if it is pulled (`ollama pull qwen2.5:7b`, optional; otherwise the 3B model judges). In notebook 12, Ollama is only needed for the content embeddings and the LLM layer (requests and explanations); SASRec, the two-tower model and the LightGBM ranker are plain PyTorch and LightGBM, and MovieLens-1M is downloaded into `data/cache/` on the first run.
- **17**: `transformers` (ModernBERT-base and Flan-T5-small, downloaded from the Hugging Face Hub on the first run), `datasets` (Banking77), `scikit-learn` (logistic regression) and **Ollama running** with `qwen2.5:3b` and `nomic-embed-text` for section 2.
- **18**: `datasets` (FineWeb and FineWeb-Edu samples, about 100 MB streamed once and kept in `data/cache/`), `transformers` (the `HuggingFaceFW/fineweb-edu-classifier`, ~440 MB, and the SmolLM2 tokenizer), `llm_edu_suite.model` for the two models it trains, and **Ollama running** with `qwen2.5:3b` for section 7 (synthetic data). About 30 minutes on the RTX 3060: two trainings of 10–11 minutes and about 3 minutes of LLM generation.
- **19**: PyTorch and `llm_edu_suite.model` only, no downloads and no Ollama, but it **needs a CUDA GPU** (it measures GPU memory and kernels; without one it stops with a message at the start). The Windows PyTorch wheels ship without FlashAttention-2 and without Triton (the `inductor` backend of `torch.compile`): the notebook detects this and carries on with what is available, and DDP runs in two CPU processes with `gloo`. About 5 minutes on the RTX 3060.
- **20**: **Ollama running** with `qwen2.5:3b`, `deepseek-r1:1.5b` (`ollama pull deepseek-r1:1.5b`, 1.1 GB) and, as the verifier and pairwise judge, `qwen2.5:7b` (if it is not pulled, `config.judge_model()` falls back to the chat model and the 3B model judges itself); `datasets` downloads the GSM8K test split from the Hugging Face Hub on the first run. No training and no PyTorch model: it is all LLM calls. About 53 minutes on the RTX 3060, of which about 25 go to sampling five chains of thought for each of the 80 problems, about 10 to the verifier and about 10 to the reasoning model.
- **LangSmith is optional.** With `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`, every LangChain/LangGraph call in 10–16 is traced. Without a key everything still runs, and notebook 15 skips the cells that need the LangSmith server and prints that it skipped them. The committed run had LangSmith on.

| Topic | Notebooks | GPU time of the final run |
|---|---|---|
| Start here | [00](#00) | 32 s |
| Architecture | [01](#01) · [02](#02) · [03](#03) · [04](#04) · [17](#17) | 5 s · 31 s · 180 s · 48 s · ≈ 15 min (≈ 19.5 min) |
| Pre-training and inference | [18](#18) · [05](#05) · [19](#19) · [06](#06) · [09](#09) | ≈ 30 min · 35 s · ≈ 5 min · 61 s · 53 s (≈ 37.5 min) |
| Post-training | [07](#07) · [08](#08) · [20](#20) | ≈ 25 min · 645 s · ≈ 53 min (≈ 89 min) |
| Applications | [10](#10) · [12](#12) · [13](#13) | ≈ 19 min · ≈ 25 min · 140 s (≈ 46 min) |
| Agents and MCP | [11](#11) · [14](#14) · [16](#16) | 56 s · 119 s · 65 s (≈ 4 min) |
| LLMOps | [15](#15) | 129 s |

Notebooks 07, 10 and 12 were re-run for v2.1 and their times are approximate: 07 now also trains QLoRA, 10 uses a 7B judge (about 17 of its 19 minutes are RAGAS scoring) and 12 includes ~11 minutes of ranker feature building. In total, about an hour and a half for notebooks 00–16, plus about 15 minutes for 17, about 30 for 18, about 5 for 19 and about 53 for 20, on the GPU, with Ollama not shared with other jobs. When something else is using the GPU, the times for 10–16 grow a lot (the LLM calls queue up). CPU times were not measured for this run: 00–06 work fine on CPU, but expect 07–09 to be much slower without a GPU. Notebook 19 does not run without a CUDA GPU.

---

<a id="00"></a>
## 00 · A complete mini LLM in one notebook
`notebooks/00_start_here/00_mini_llm_end_to_end.ipynb` · also as a script: `python scripts/mini_llm.py`

**What you build.** A whole GPT without importing anything from the project, just PyTorch: a byte-level BPE tokenizer, a modern transformer (embeddings, RoPE, RMSNorm, causal attention, SwiGLU, tied output head), the training loop (AdamW, warmup + cosine, clipping), evaluation, sampling and saving.

**Key concepts**
- How all the pieces fit together before you study them separately.
- The sanity check: the initial loss should be ≈ ln(V).
- **Irreducible loss**: since the corpus is generated by a known grammar, you can compute the theoretical minimum (4·ln 8 nats per sentence) and measure how far the model is from it.
- Why BPE stops merging once every word is already a token.

**Results.** 1.86M parameters, a 455-token vocabulary (199 merges). Initial loss 6.144 against ln V = 6.120. The 800 training steps take **~19 s on the GPU**. Validation loss 0.778 against a theoretical floor of 0.774: **0.6% above the optimum**. **82%** of the sampled sentences (40/49) are grammatical. Greedy decoding shows the typical repetition loop.

**Exercises.** Train it on tiny Shakespeare (`text_path`), remove RoPE and watch what happens to the grammar, or add a KV-cache to `generate`.

---

<a id="01"></a>
## 01 · Tokenization
`notebooks/01_architecture/01_tokenization.ipynb` · module `tokenizer`

**What you build.** A byte-level BPE with training, encoding, decoding, special tokens and save/load.

**Key concepts.** Why sub-words; why byte-level BPE never produces `<UNK>`; how vocabulary size trades sequence length against embedding-table size; **fertility** (tokens per word) as a measure of how well a tokenizer covers a domain or language.

**Results.** 255 merges, a 514-token vocabulary including 3 special tokens. The example sentence is 52 bytes and encodes to 7 tokens (7.43 bytes/token), with an exact round-trip. Fertility **1.00 in the training domain versus 3.90 outside it** (code and Spanish): this is why under-represented languages cost more tokens.

---

<a id="02"></a>
## 02 · Attention, RoPE and normalization
`notebooks/01_architecture/02_attention_rope_norm.ipynb` · modules `layers`, `model`

**What you build.** RMSNorm (checked against `torch.nn.RMSNorm`), RoPE with Llama-3.1-style scaling, attention written by hand (checked against `F.scaled_dot_product_attention`), MHA → GQA → MQA → MLA, and the KV-cache.

**Key concepts.** RoPE depends only on relative distance (the notebook checks it numerically). GQA shares K/V heads to shrink the cache. MLA (DeepSeek) compresses K/V into a latent vector. Incremental generation with the cache must give exactly the same result as the full forward pass (max difference ~1e-7 for GQA and MLA).

**Results.** Training the same network with each variant: **MHA 2048 B/token of cache versus MLA 160 B/token, at the same perplexity (2.65)**. GQA-2 and MQA sit in between (512 and 256 B/token, perplexity 2.64). In DeepSeek-V3 the per-layer reduction is about 57×.

---

<a id="03"></a>
## 03 · Efficient attention for long contexts
`notebooks/01_architecture/03_efficient_attention.ipynb` · module `layers`

**What you build.** Sliding window (SWA), Longformer, FlexAttention (if your PyTorch has it), content-selected block-sparse attention, linear attention (with its constant-state recurrent form) and hybrids.

**Key concepts.** O(T²) versus O(T·w) versus O(T) cost; the receptive field of stacked SWA; why linear attention has a fixed-size state; **choosing the right metric**: overall perplexity is useless on this task, copy accuracy is not.

**Results.** A copy task (a 12-token span repeated after a 30-token gap, so ~42 tokens apart), 1500 steps:

| Variant | Copy accuracy |
|---|---|
| dense (GQA) | **99.9%** |
| block-sparse | **88.7%** |
| hybrid (1 global layer out of 4) | 48.4% |
| sliding window w=16 | 0.6% |
| linear | 0.3% |

With only 300 steps even the dense model is at 0.3%: ***induction heads*** appear suddenly during training (a phase transition).

---

<a id="04"></a>
## 04 · FFN and Mixture of Experts
`notebooks/01_architecture/04_ffn_moe.ipynb` · module `layers`

**What you build.** SwiGLU with Llama's sizing rule, an MoE layer with top-k routing, shared experts, balancing with an auxiliary loss and *aux-free* balancing (DeepSeek-V3).

**Key concepts.** Total versus active parameters; why the router collapses without balancing; why the aux-free bias update must be small; a fair comparison at equal active compute.

**Results.** With top-2 of 8 experts, the layer has 3× (2.99×) more total than active parameters. Aux-free balancing takes an unbalanced router **from 4.0× to 1.15×** with `bias_lr=0.01`, while 0.1 oscillates (it only gets to 1.68×; the plot shows it). On the toy corpus MoE does not beat the dense model: dense ppl 3.24, MoE with auxiliary loss 3.31, MoE aux-free 3.24, with the same active parameters. The notebook explains why MoE pays off at scale.

**Next:** [17](#17), which completes the architecture topic, then [18](#18) (the data) and 05.

---

<a id="17"></a>
## 17 · Encoders and encoder-decoders: the other two transformer families
`notebooks/01_architecture/17_encoders_and_encoder_decoders.ipynb` · modules `model`, `data`, `training` · `transformers`, `datasets`, `scikit-learn`, LangChain (Ollama)

Read it after 04 and before 05. Notebooks 00–06 build a single kind of transformer, the decoder-only one; this notebook covers the other two branches of the original transformer: **encoders** (BERT, ModernBERT), with bidirectional attention, and **encoder-decoders** (T5, Flan-T5), with *cross-attention*.

**What you build.** Three parts. (1) **From scratch:** the same `TransformerLM` with `ModelConfig(causal=False)` (bidirectional attention), trained with BERT's MLM objective, against a causal decoder of the same size; both for 800 steps (~30 s each) on the synthetic corpus of 05, followed by a fill-in-the-blank test. (2) **In practice:** classifying the real messages of Banking77 (77 intents) three ways: fine-tuning an encoder (ModernBERT-base), frozen embeddings + logistic regression, and asking an LLM (`qwen2.5:3b`) zero-shot and few-shot. (3) **Encoder-decoder:** Flan-T5-small, how its parameters are split and what its cross-attention reads.

**Tools.** `ModelConfig(causal=False)` and `train_lm` from `llm_edu_suite`; a `mask_tokens` function with BERT's 80/10/10 recipe; `datasets.load_dataset("PolyAI/banking77")` with a stratified validation split; `AutoModelForSequenceClassification` + `Trainer` (`learning_rate=5e-5`, warm-up, linear decay, bf16, `DataCollatorWithPadding` for dynamic padding); `AutoModel` with mean pooling; `OllamaEmbeddings(nomic-embed-text)` with the `classification:` prefix; `scikit-learn` (`StandardScaler` + `LogisticRegression`, `C` chosen on validation, and `f1_score`); `with_structured_output(..., method="json_schema" | "function_calling", include_raw=True)` with a `Literal` of the 77 intents; `evaluation.bootstrap_ci` and `paired_bootstrap_diff`; `AutoModelForSeq2SeqLM` with `attn_implementation="eager"` to get the attention weights.

**Key concepts**
- All three families stack the same block; what changes is **the mask**. An encoder is the same attention layer called with `is_causal=False`.
- **Masked language modeling**: a bidirectional model cannot be trained to predict the next token (it would be looking at it), so 15% of the positions are hidden (80% `[MASK]`, 10% a random token, 10% unchanged). It only learns from that 15%; ELECTRA fixes this by classifying every position.
- A sentence representation from `[CLS]` or from the mean of the tokens, with a linear **classification head** on top that is fine-tuned together with the encoder.
- ModernBERT: what decoders learned (RoPE, GeGLU, pre-normalization, global layers alternating with local windows, *unpadding*, an 8,192-token context) applied to an encoder.
- **Linear probe**: a linear classifier on frozen representations measures how much useful information they hold.
- **Constrained decoding** versus function calling when the enum is large.
- Cross-attention: the queries come from the output being written; the keys and values from the encoder states.
- Protocol: the official test set untouched, a stratified validation split to choose settings, macro-F1 next to accuracy, and a paired sample to compare against the LLM.

**Results**

From scratch, hiding one token of held-out text (989 first-token-of-a-word cases and 2069 inside-a-word cases):

| Model | First token of a word | Inside a word | All |
|---|---|---|---|
| decoder (left context) | 2.9% | 95.4% | 73.6% |
| encoder, left context only | 2.5% | 95.9% | 73.8% |
| **encoder, both sides** | **100%** | 98.3% | 99.1% |

Banking77 (13,083 messages, CC BY 4.0; train 9,002 · validation 1,001 · test 3,080):

| Method | Labels | Test accuracy (3,080) | Paired 231-message sample [95% CI] | Latency per message |
|---|---|---|---|---|
| ModernBERT fine-tuned, 8 per intent | 616 | 69.6% (102 s) | 0.649 [0.589, 0.710] | 40 ms |
| ModernBERT fine-tuned, 32 per intent | 2,464 | 89.3% (128 s) | 0.861 [0.818, 0.905] | 40 ms |
| ModernBERT fine-tuned, all (~117 per intent) | 9,002 | **93.0%** (163 s) | 0.922 [0.883, 0.957] | 40 ms |
| `nomic-embed-text` + logistic regression | 9,002 | **93.3%** | 0.926 [0.892, 0.957] | **18 ms** |
| ModernBERT frozen (mean-pooled) + logistic regression | 9,002 | 89.2% | 0.896 [0.857, 0.935] | – |
| `qwen2.5:3b` few-shot (1 per intent, `json_schema`) | 77 | – | 0.619 [0.554, 0.680] | ~250 ms |
| `qwen2.5:3b` zero-shot (`json_schema`) | 0 | – | 0.489 [0.424, 0.554] | ~250 ms |
| `qwen2.5:3b` zero-shot (`function_calling`) | 0 | – | 0.152 [0.104, 0.195], 181 of 231 outputs invalid | ~310 ms |

For every trained method, macro-F1 equals accuracy to three decimals.

Flan-T5-small (77M): a 19M encoder stack, a 25M decoder stack (with cross-attention) and a 16M shared embedding; a decoder block is self-attention → cross-attention → feed-forward.

**Honest findings**
- **Bidirectional context is the encoder's advantage, and it comes from the right-hand side.** At the start of a word, from the left you only know that a new word begins (2.9%); with both sides, the rest of the word gives it away (100%). The control row is what makes the comparison fair: the *same* encoder with the right side cut off falls to 2.5%, the decoder's level. The gain does not come from a better model.
- **With labels, a fine-tuned encoder or a linear classifier beats a small LLM by more than 30 points** on a 77-way task. Many intents are near-duplicates ("pending_transfer" vs "transfer_not_received_by_recipient") that only data teaches.
- **Embeddings + logistic regression match fine-tuning** (0.926 against 0.922 on the paired sample, a difference inside the noise), with no network training and half the latency. Frozen ModernBERT, never trained for similarity, is four points behind: for a "general" representation, the pre-training objective matters.
- **The encoder is data-hungry at first**: with 8 examples per intent it stays at 69.6%, still above the LLM, and it improves with every label, while prompting does not.
- **How the output is constrained decides between usable and useless.** With `function_calling` the model mostly did not call the tool with a 77-value enum (181 of 231 outputs invalid); with `json_schema` (grammar-constrained decoding) every output was valid. This is the contrast with [11](#11), where function calling was the more reliable method on a small schema.
- **Flan-T5-small** classifies the stolen-card message correctly ("lost card") and writes a reasonable summary, but its translation is wrong: "Einzahlung" means *deposit*, not *purchase*. The last layer's cross-attention map is dominated by two *attention sinks* (the "e" of "Translate" and `</s>`), with a faint alignment underneath. The last layer of a small model is a poor place to look for alignment.
- The notebook ends with "Results" and an "In production" section with a decision guide (no labels: an LLM with constrained decoding; hundreds of labels: embeddings + logistic regression; for the last points: fine-tune an encoder), what to monitor and the typical failure modes (label drift, over-confidence out of distribution, train/test leakage).

**Next:** [18](#18), what happens to text before pre-training, and then 05.

---

<a id="18"></a>
## 18 · Data engineering for LLMs: deduplication, quality filtering, decontamination
`notebooks/02_pretraining_and_inference/18_data_engineering.ipynb` · modules `model`, `config` · `datasets`, `transformers`, LangChain (Ollama)

Read it right before 05: this is what happens to text before a model ever sees it. The most repeated sentence in LLM engineering is that the data matters more than the model, and the open models that caught up with closed ones in 2023–2024 did it largely through better data (FineWeb, DCLM, Llama 3's data pipeline). This notebook builds the main steps of that pipeline on a real sample, measures what each one removes and why, and finally checks whether cleaning actually changes the trained model.

**What you build.** On a **FineWeb** sample (12,000 documents, 6.2M words, median 297 per document; streamed once into `data/cache/`, ODC-By license, not redistributed) and a **FineWeb-Edu** reference (1,000 documents):
1. A **simulated raw crawl** with known duplicates, because FineWeb is already deduplicated: 10,000 base documents plus 600 exact copies (a mirror), 50 documents copied 10 times each (syndicated content) and 600 near-duplicates of each of four kinds (another template with the text unchanged, and another template with 1 %, 3 % or 10 % of the words replaced). 13,500 documents in total. The last 2,000 FineWeb documents stay out of everything as a clean held-out set.
2. **Exact deduplication**: normalize (lowercase, collapse whitespace), SHA-1, keep the first document of each hash.
3. **MinHash + LSH from scratch**: word 5-gram shingles hashed to 32 bits with CRC-32; 128 affine hash functions $h_i(x) = (a_i x + b_i) \bmod p$ with the Mersenne prime $p = 2^{61} - 1$; 16 bands of 8 rows; a candidate is accepted if its signatures agree on at least 70 % of positions, and accepted pairs are merged into clusters with union-find. Recall is measured per duplicate kind and compared with the theoretical curve.
4. **Quality filters**: a compact version of the Gopher and C4 rules that records which rule removes each document (a per-rule funnel), then the **FineWeb-Edu classifier** (a 110M BERT encoder, like the ones in [17](#17)) scoring a sample of the survivors.
5. **Decontamination** with 13-grams: 200 held-out documents act as a "benchmark", the first 300 words of 50 of them are pasted into training documents, and training documents sharing any 13-gram with the benchmark are flagged.
6. **Does it change the model?**: two identical `TransformerLM`s (the one from 05, 4 layers of width 256, ~16M parameters, SmolLM2's 49k-token BPE tokenizer), one on the raw crawl and one on the clean corpus, with the same number of tokens.
7. **Synthetic data**: `qwen2.5:3b` writes "textbook" passages on 20 topics (3 samples per topic at temperature 0.8), treated like any other data: near-duplicates with MinHash, diversity (distinct bigrams) and the classifier's score.

**Tools.** `datasets.load_dataset("HuggingFaceFW/fineweb" / "HuggingFaceFW/fineweb-edu", name="sample-10BT", streaming=True)` with `itertools.islice` to read only the first documents; `hashlib.sha1` and `zlib.crc32`; NumPy for the signatures (the 128 hash functions vectorized over all of a document's shingles); `collections.defaultdict` as the per-band bucket table; union-find with path compression; `AutoModelForSequenceClassification` with `HuggingFaceFW/fineweb-edu-classifier` (truncated to 512 tokens, bf16); `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")`; `ModelConfig` and `TransformerLM` from `llm_edu_suite.model` with AdamW (`lr=1e-3`, `betas=(0.9, 0.95)`, `weight_decay=0.1`) and warm-up + cosine; `config.get_llm(temperature=0.8)` with `.batch(..., max_concurrency=4)`.

**Key concepts**
- The pre-training corpus pipeline: crawl → text extraction → language ID → URL and safety filters → quality filters → deduplication (exact and near-duplicate) → decontamination → mixing → tokenization. FineWeb kept a filter only if a small model trained on the filtered data scored better; the notebook copies that method at small scale.
- Why duplicates hurt: they waste compute, cause **memorization** (a privacy and copyright risk) and leak evaluations.
- **Jaccard similarity** between shingle sets and **MinHash**: $\Pr[\min h(A) = \min h(B)] = J(A, B)$, with standard error $\sqrt{J(1-J)/k}$ for $k$ hash functions.
- **Banded LSH**: $\Pr[\text{candidate}] = 1 - (1 - J^r)^b$, an S-curve with a threshold near $(1/b)^{1/r}$; with $b = 16$, $r = 8$, $J \approx 0.71$. Bands and rows set the threshold, and the threshold defines what a "duplicate" is.
- Heuristic rules (Gopher, C4) versus **model-based filters** (FineWeb-Edu, DCLM): the former remove what is not text; the latter judge what the text is worth, with the bias of whoever labelled their data.
- **Decontamination** with long n-grams (8 to 13): cheap and reliable for verbatim copies, blind to paraphrases.
- Synthetic data: controllable and cheap, but less diverse, and training recursively on model outputs narrows the distribution (***model collapse***).

**Results**

| Step | Measured | What it shows |
|---|---|---|
| exact dedup | 13,500 → 12,400 documents; all 1,100 planted copies removed, nothing else | cheap, exact, and blind to the smallest edit |
| MinHash + LSH (k = 128, 16 × 8, threshold 0.7) | 1,693 candidate pairs, 1,633 accepted, 12,400 → 10,767 in 12 s; recall 100 % (template only and +1 %), 72.2 % (+3 %), 0 % (+10 %); 0 merges of different base documents | the measured detection rate sits on the theoretical curve $1 - (1 - J^8)^{16}$ |
| Gopher + C4 rules | 10,767 → 9,472 (12.0 % removed); 1,173 by C4's terminal-punctuation rule, then length (66), non-alphabetic words (29), bullet lines (14), stop words (11) | rules remove menus, forum threads and fragments, not bad prose |
| FineWeb-Edu classifier | mean 1.20 and 2.0 % scored ≥ 3 on FineWeb after the rules; 1.00 and 0.8 % on what they rejected; 3.04 and 47.6 % on FineWeb-Edu | model-based filtering is far more selective; spam in good prose survives the rules |
| decontamination (13-grams) | 50 of 50 planted leaks found in 6 s, plus 29 natural overlaps (syndicated news) | contamination happens on its own when the web repeats itself |
| training on raw vs clean (same 12.3M tokens) | held-out loss 5.547 vs 5.537 (perplexity 257 vs 254); loss on the 43 documents repeated 10 times 4.350 vs 5.281 | at this scale, dedup's measurable effect is less memorization at 27 % fewer tokens |
| synthetic data (`qwen2.5:3b`, 60 passages in 196 s) | classifier score 4.15 (100 % ≥ 3) vs 3.14 (57 %) for human text; distinct bigrams 0.60 vs 0.72 | synthetic text games the quality classifier and is less diverse |

**Honest findings**
- **The threshold decides what a duplicate is.** One edited word breaks up to five 5-grams, so replacing 1 % of the words leaves a Jaccard around 0.9 (always caught), 3 % leaves 0.72–0.78, right on the threshold (72.2 %), and 10 % drops it to about 0.42, where LSH by design does not look (0 %). That the measured rates sit on the theoretical curve is the best check that the implementation is right. FineWeb uses 5-grams and a threshold near 0.75: it removes templated copies and light rewrites but keeps documents that merely quote each other.
- **Rules and classifier measure different things.** What the rules reject scores 1.00 on average, close to what they keep (1.20), and the lowest-scored survivor is the terms-and-conditions page of an escort site: well-formed prose that passes every rule. Heuristics remove what is not text; a classifier is needed to judge what the text is worth. The cost: keeping only scores of 3 or more would throw away 98 % of the sample, and the classifier encodes one model's idea of "educational", which favours textbook prose over conversation or code. (The 47.6 % for FineWeb-Edu is not identical to theirs: they round the score before thresholding, and here the input is truncated at 512 tokens.)
- **Nobody copies the test set on purpose.** The 29 flagged documents that were not planted share 13-grams with the benchmark because the web repeats itself: Associated Press stories republished by different sites. A benchmark built from web text has to be checked against the training corpus.
- **Cleaning does not improve held-out loss at this scale, and the notebook says so.** Cleaning removed 27 % of the tokens (9.5M → 6.9M) and both models trained on the same 12.3M (1,500 steps of 32 × 256), so the clean model saw its data more times. 5.547 against 5.537 is within noise: FineWeb measured the gains of filtering on models trained for billions of tokens and on benchmarks, and a 16M model trained for a few minutes is far from that regime.
- **Memorization does show.** On the 43 documents repeated 10 times that survived cleaning (the clean model saw each once), the raw model's loss is 4.350, well below its loss on unseen text (5.547), and the clean model's is 5.281, close to its own. Ten copies are enough for a small model to start memorizing, the effect Lee et al. (2022) and Carlini et al. (2023) measured on large models: duplication buys memorization, not knowledge.
- **Synthetic text games the filter that judges it.** The classifier scores the `qwen2.5:3b` passages above human FineWeb-Edu text (4.15 vs 3.14) because it rewards what *looks* educational. They are not near-duplicates of each other (same-topic pairs share only 0.09 of their 3-grams, none above 0.5); the narrowing is subtler: the same structure and phrasing patterns across topics, visible in the distinct bigrams (0.60 vs 0.72). Synthetic data is evaluated by training on it, not by the classifier, and lexical diversity is the cheap early warning of collapse.
- The notebook ends with "Results" and an "In production" section: use a framework built for this (`datatrove`, used for FineWeb, or NVIDIA NeMo Curator) and keep each step's decision per document; defaults (exact dedup, then MinHash on 5-grams with a threshold around 0.75, within each crawl snapshot first; Gopher and C4 as a baseline with a model score on top, its threshold chosen by small training ablations; decontaminate against every benchmark you report, with 8 to 13-grams; for synthetic data, varied prompts, personas and seeds, and measure diversity); what to monitor (each source's and filter's share over time, duplicate rates per source, the classifier's score distribution and a random sample read by a person at every release); and failure modes (a threshold that silently removes a whole domain such as code, other languages or dialects; over-aggressive dedup that removes legitimate repetition; contamination discovered after publication; synthetic data that inflates the metrics used to judge it).

**Next:** [05](#05), where text like this becomes a trained model.

---

<a id="05"></a>
## 05 · Pre-training
`notebooks/02_pretraining_and_inference/05_pretraining.ipynb` · modules `training`, `data`

**What you build.** The full pipeline: text → tokenizer → model → loop with AdamW, warmup + cosine, clipping, checkpoints and a dashboard (loss, gradient norm, LR); then generation, saving and loading.

**Key concepts.** Sanity checks (initial loss ≈ ln V, a test that the model cannot see the future); how to read the dashboard; **irreducible loss** (the *E* term of Chinchilla); a schedule ablation (cosine versus WSD); a scaling sanity check.

**Results.** Validation perplexity 2.79 against a theoretical floor of 2.75 (**1.5% above the floor** in loss). At the same number of steps the 1.67M model learns much faster than the 125K one (loss 1.054 versus 1.259 at step 50). Cosine and WSD end up practically the same (1.024 versus 1.027).

**Next:** [19](#19), which measures what a training run like this costs (memory, precision, kernels and parallelism), then [06](#06).

---

<a id="19"></a>
## 19 · Systems for training LLMs: memory, precision, kernels and parallelism
`notebooks/02_pretraining_and_inference/19_systems_memory_and_parallelism.ipynb` · modules `model`, `utils` · `torch.profiler`, `torch.utils.checkpoint`, `torch.nn.attention`, `torch.compile`, `torch.distributed`

Read right after 05. Notebook 05 trains a model; this one looks at **what that costs and why**: the part of LLM engineering that decides whether a run fits on your hardware, how fast it goes and how it scales to many GPUs. Everything is measured on the course's GPU (an RTX 3060 Laptop, 6 GB), mostly with a 29.4M-parameter `TransformerLM` (width 512, 8 layers, vocabulary 8,192, context 512) on random tokens: what matters here is the cost, not what it learns.

**What you build.**
1. **Memory anatomy**: `torch.cuda.memory_allocated()` after moving the model, after forward + backward and after the optimizer step, against the textbook formula (in fp32, 4 bytes per parameter for the weights, 4 for the gradients and 8 for Adam's two moments), and the activation peak from batch 2 to 16 in fp32 and in bf16.
2. **bf16 and activation checkpointing**: the same training step (batch 8) in fp32 and with bf16 autocast, with and without `torch.utils.checkpoint.checkpoint` around every block; time, peak memory and the loss after 30 steps on the same batches.
3. **Attention kernels**: one causal attention layer (8 heads of 64, batch 1, bf16), forward and backward, from 512 to 16,384 tokens. The naive version materializes the $T \times T$ matrix; against it, the backends of `F.scaled_dot_product_attention` forced one at a time (math, FlashAttention-2, memory-efficient and cuDNN), after checking which ones this PyTorch build can run.
4. **Profiling** one bf16 training step (batch 8) with `torch.profiler`: the operators with the most GPU time, the share that goes into matrix multiplications and the number of kernels launched.
5. **Launch overhead and CUDA graphs**: a small model (6 layers of width 256) running the forward pass on 64 tokens at batch 1, as in decoding, in eager mode and captured with `torch.compile(backend="cudagraphs")`.
6. **MFU**: tokens per second of full bf16 training steps at batch 16, converted into useful FLOP/s with $6N + 12LdT$ FLOPs per token and divided by the peak measured with a large bf16 matmul.
7. **Data parallelism for real**: `DistributedDataParallel` in two CPU processes on a small `TransformerLM` (2 layers of width 64), checking that two processes with batch 4 end up with the same weights as one process with batch 8.
8. **A ZeRO / FSDP calculator**: per-GPU memory for the model states ($16\Psi$ bytes) under DDP and the three ZeRO stages, for 7B and 70B with 8 and with 64 GPUs.

**Tools.** `torch.cuda.memory_allocated`, `max_memory_allocated` and `reset_peak_memory_stats`; `torch.autocast("cuda", dtype=torch.bfloat16)`; `torch.utils.checkpoint.checkpoint`; `torch.nn.attention.sdpa_kernel` with `SDPBackend`; `torch.profiler.profile` with `ProfilerActivity.CUDA`; `torch.compile(backend="cudagraphs")`; `torch.distributed` with `gloo` and `init_method="file://..."` (a `FileStore`), `DistributedDataParallel` and `torch.multiprocessing.spawn`; `ModelConfig` and `TransformerLM` from `llm_edu_suite.model`.

**Key concepts**
- **Compute accounting**: training costs $C \approx 6ND$ FLOPs (2 per parameter per token forward, 4 backward), plus an attention term of about $12LdT$ per token that only matters when $T$ is large compared with $d$.
- **16 bytes per parameter**: with Adam and mixed precision, 2 (bf16 weights) + 2 (gradients) + 12 (fp32 master weights and Adam's two moments); in plain fp32 it is also 16 (4 + 4 + 8). A 7B model needs about 112 GB before the first activation.
- **Activations**: they grow with batch × sequence × width × layers, plus a $T^2$ term if the attention kernel stores the score matrix. For long sequences and large batches they dominate.
- **Mixed precision**: compute in bf16 on the tensor cores with fp32 master weights. bf16 has fp32's exponent range and needs no *loss scaling*; fp16 does.
- **Activation checkpointing**: keep only each block's input and recompute its internals during the backward pass, at the price of one extra forward pass (~30 % more compute).
- **Arithmetic intensity**: FLOPs per byte moved. Large matmuls run near peak; element-wise operations, normalizations and softmax are memory-bound. That is why fusing kernels helps, and why FlashAttention wins mostly by moving fewer bytes.
- **FlashAttention**: attention computed in tiles in the chip's SRAM with an online softmax. It never writes the $T \times T$ matrix to HBM, it is exact and its memory is linear in $T$.
- **Kernel fusion and CUDA graphs**: every PyTorch operation launches a kernel with a fixed cost of a few microseconds. `torch.compile` fuses operations (its default backend generates Triton); a CUDA graph records a sequence of kernels and replays it with a single launch.
- **MFU** (*model FLOPs utilization*): the share of the GPU's peak that is useful work; well-tuned large training runs reach 40–60 %.
- **Parallelism**: data (DDP, an all-reduce of the gradients), ZeRO-1/2/3 and FSDP (shard the optimizer states, the gradients and the weights), tensor (each matmul across GPUs), pipeline (the layers in stages), sequence and expert. A ring **all-reduce** of $P$ bytes moves $2\frac{k-1}{k}P$ bytes per GPU, almost independent of the number of GPUs $k$.

**Results**

| Question | Measured on the RTX 3060 Laptop (6 GB) | Lesson |
|---|---|---|
| model memory, 29.4M parameters, fp32 + Adam | weights 0.111 GB, gradients 0.118 GB, Adam 0.219 GB (predicted 0.110 / 0.110 / 0.219) | the 16-bytes-per-parameter rule holds |
| activations | 0.43 → 3.39 GB from batch 2 to 16 (fp32); 1.71 GB at batch 8 versus 0.45 GB of model states | activations decide what fits |
| bf16 autocast | step 2.2× faster (275.6 → 123.6 ms), peak 2.18 → 1.66 GB | always on, with fp32 master weights |
| activation checkpointing | peak −57 % in fp32 (2.18 → 0.93 GB) and −51 % in bf16 (1.66 → 0.81 GB) for ~22 % more time | the standard trade when memory is the limit |
| attention kernels | naive: 4.1 GB at 8k tokens, out of memory at 16k; fused: 0.22 GB at 16k, 51.9 ms with cuDNN | fused kernels make long contexts possible; FlashAttention-2 is missing from the Windows wheels |
| profile of one step | 226.9 ms of GPU time, 47 % in matmuls, 1,136 kernels | in a small model, element-wise ops and launches weigh a lot |
| CUDA graphs | 8.68 → 2.11 ms per forward (4.1×), same output | launch overhead is real and removable |
| efficiency | 36,608 tokens/s, 7.38 TFLOP/s, MFU 32 % (measured peak 23.0 TFLOP/s) | a reasonable baseline; large runs reach 40–60 % |
| data parallelism (2 processes) | weights identical to one process with the global batch (max difference 7.45e-9); 0.53 MB all-reduced per step | DDP is large-batch training, plus an all-reduce |
| ZeRO / FSDP calculator | 7B: 112 GB (DDP) → 14 GB per GPU (FSDP, 8 GPUs), 1.8 GB with 64; 70B: 140 GB with 8, 17.5 GB with 64 | sharding is what makes large models trainable |

**Honest findings**
- **The textbook arithmetic holds almost to the megabyte, and it is not what weighs most.** Weights 0.111 GB against 0.110 predicted, Adam 0.219 against 0.219; the gradients take 0.118 GB because of a few small autograd buffers. The activations dwarf it all: at batch 8 the forward + backward peak is 1.82 GB, about 1.71 GB of activations against 0.45 GB of model states, and an easy part to forget is the logits (batch × 512 × 8,192 numbers) kept for the loss. bf16 trims the activations by only about a quarter (2.47 GB at batch 16), because some of the saved tensors stay in fp32.
- **bf16 and checkpointing combine.** bf16 makes a step 2.2× faster thanks to the tensor cores; checkpointing costs ~22 % more time (123.6 → 150.9 ms in bf16), close to the "one extra forward pass" of the theory. The loss is 9.062 in all four variants, but the data are random tokens: this only checks that the numerics agree, not that the model learns. Absolute step times move between runs on a laptop GPU (power and temperature limits); the ratios are what carry over.
- **Knowing which kernel your build uses is part of the job.** The Windows PyTorch wheels do not include FlashAttention-2, so the fused kernels compared are memory-efficient and cuDNN, built on the same IO-aware idea. The naive version grows with the square of the sequence (0.035 GB at 512 tokens, 4.1 GB at 8,192, out of memory at 16,384); memory-efficient needs 0.118 GB at 8,192 (about 35× less) and 0.220 GB at 16,384, and cuDNN runs forward and backward at 16,384 tokens in 51.9 ms. At short lengths all of them are fast and the differences are noise.
- **A Windows trap: slow instead of failing.** SDPA's unfused *math* backend reported 8.1 GB at 8,192 tokens on a 6 GB card and took 978 ms: the Windows driver spilled into system RAM ("shared GPU memory") instead of raising an out-of-memory error. If a job suddenly slows down instead of crashing, check whether it is paging into shared memory.
- **Matmuls are only 47 %.** One bf16 step is 226.9 ms of GPU time and 1,136 kernels; outside the matmuls are the element-wise operations (`mul`, `copy_`, the RMSNorm and SwiGLU arithmetic), the fused attention backward (3.6 %) and AdamW (3.4 %). At width 512 each matmul is short and the memory-bound work weighs more: that is what kernel fusion (`torch.compile`, Triton, Liger kernels) targets.
- **`torch.compile` without Triton.** The default backend, `inductor`, is unavailable (`TritonMissing`: Triton is not supported on Windows). With `backend="cudagraphs"` the small model's forward goes from 8.68 to 2.11 ms (4.1×) with identical output: the GPU does the same arithmetic, and what disappears is the CPU cost of launching hundreds of small kernels, the caveat of [06](#06). That is why vLLM, TensorRT-LLM and llama.cpp capture the decode step in CUDA graphs.
- **An MFU of 32 %.** 36,608 tokens/s at 202 MFLOPs per token is 7.38 TFLOP/s of useful work against a measured peak of 23.0 TFLOP/s. That is reasonable for 29M parameters on a laptop; what is missing is the time outside matmuls and the launches. For perspective: at this speed, even this small model would need 4.5 hours to see its Chinchilla-optimal 588M tokens.
- **DDP is large-batch training, split up.** After 5 steps, the weights of two processes with batch 4 and of one process with batch 8 differ by at most 7.45 × 10⁻⁹, floating-point rounding. Each step all-reduces 0.53 MB of gradients (the small model's 133k fp32 parameters). NCCL, the GPU backend, needs Linux and several GPUs, so `gloo` on CPU is used here; and instead of `torchrun`, whose TCP rendezvous needs libuv (which this PyTorch build lacks), the processes are launched with `torch.multiprocessing.spawn` and meet through a shared file (`init_method="file://..."`).
- **Plain DDP does not scale memory.** It replicates the 16 bytes per parameter on every GPU: 112 GB per GPU for a 7B model, however many GPUs you add. ZeRO-1 brings it to 38.5 GB with 8 GPUs, ZeRO-2 to 26.2 GB and ZeRO-3 / FSDP to 14.0 GB (1.8 GB with 64). For 70B, from 1,120 GB to 140 GB per GPU with 8, still too much for an 80 GB H100, and 17.5 GB with 64. ZeRO-3 pays for that memory in communication (an all-gather of the weights before each layer); that is why the largest models combine tensor parallelism inside a node, over NVLink, with FSDP across nodes.
- The notebook ends with "Results" and an "In production" section: defaults (bf16 with fp32 master weights; a fused attention kernel, checking the backend, which on Linux is FlashAttention-2/3; AdamW with weight decay, clipping and checkpointing only where memory forces it; a batch that fills memory and gradient accumulation for the effective batch); how to scale out (DDP while the 16 bytes per parameter fit on one GPU, FSDP with `fully_shard` or DeepSpeed ZeRO-3 when they do not, tensor parallelism inside a node for the largest models; `torchtitan`, Megatron-LM / NeMo and DeepSpeed as references, launched with `torchrun` on Linux with NCCL over NVLink or InfiniBand); measuring before optimizing (`torch.profiler` or Nsight Systems, MFU from $6N$ FLOPs per token against the real peak, tokens/s and memory per GPU on a dashboard, and a drop in MFU treated like any other regression), and the failure modes (out of memory from activations, silent fallbacks, fp16 without loss scaling, communication-bound scaling and nondeterminism between runs).

**Next:** [06](#06), where the same ideas (memory-bound decoding, KV-cache, batching) apply to serving.

---

<a id="06"></a>
## 06 · Inference and serving
`notebooks/02_pretraining_and_inference/06_inference_serving.ipynb` · modules `inference`, `model` · `transformers` (section 6)

**What you build.** Sampling strategies, KV-cache, **speculative decoding from scratch** (with a check that it preserves the distribution), batching and TTFT/throughput measurement.

**Key concepts.** Temperature, top-k and top-p as transformations of the distribution (plot and entropy); diversity (distinct-2); why speculative decoding is exact; decoding is memory-bound; and the *overhead* of launching GPU kernels for small models.

**Results.** At the start of a word, the entropy is 4.25 nats at T=1.0 and drops to 2.89 with T=0.7 + top-k 20. Greedy produces the same continuation 16 times out of 16. Speculative decoding **accepts 98% of the draft's tokens, ~4.6 tokens per pass of the large model**. Batch 16 gives ~16× the tokens/s of batch 1 (2180 versus 137). On the GPU, with a model this small, the KV-cache barely speeds things up (1.2×); the notebook explains why and when that changes.

**Section 6: from our code to real weights.** SmolLM2-135M-Instruct's published weights are loaded into our `TransformerLM`. It takes a mapping of tensor names and a permutation of the rows of `q_proj` and `k_proj`, because Hugging Face uses the *rotate_half* RoPE convention and our code the interleaved one. Result in fp32: 272 tensors copied, the same 134,515,008 parameters, logits equal to within 3e-5, the same token at 100% of positions (KL 2.4e-8) and **the same 76 generated tokens** as `transformers`. Without the permutation the model loads and runs but silently gets worse: it agrees on only 68% of tokens and answers something fluent but unrelated to the question. Lesson: always check a port numerically.

---

<a id="07"></a>
## 07 · Fine-tuning with Hugging Face: SFT, LoRA and distillation
`notebooks/03_post_training/07_finetuning_sft_lora.ipynb` · `transformers`, `peft`, `trl` · also as a script: `python scripts/finetune.py sft`

**What you build.** You fine-tune `HuggingFaceTB/SmolLM2-135M-Instruct` to turn a booking request into strict JSON (`name`, `restaurant`, `people` as an integer, lowercase `day`, `time` in 24h). 300 labelled examples; 40 test requests with names and restaurants never seen in training. Few-shot baseline → full SFT → LoRA → QLoRA on a 4-bit base → the "new special token" trap → rank ablation → distillation from a 360M teacher → model merging: a second adapter for another task (converting a time to 24-hour format) is merged with the booking adapter in four ways and compared with one multi-task adapter trained on both datasets.

**Tools.** `AutoTokenizer.apply_chat_template`, `AutoModelForCausalLM`; `trl.SFTConfig` / `SFTTrainer` with a prompt/completion dataset (loss on the completion only); `peft.LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")`, `get_peft_model`, `PeftModel.from_pretrained`, `merge_and_unload()`; `tokenizer.add_tokens` + `resize_token_embeddings`, and `LoraConfig(modules_to_save=["embed_tokens"], ensure_weight_tying=True)`; for QLoRA, `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`, `peft.prepare_model_for_kbit_training` and `optim="paged_adamw_8bit"`; for merging, `PeftModel.load_adapter` (two adapters on one base) and `add_weighted_adapter(combination_type="linear" | "cat" | "ties_svd" | "dare_ties_svd")`.

**Key concepts**
- The chat template and loss masking: prompt labels are `-100`, and the loss covers the answer **including `<|im_end|>`** (in the example, 42 of 110 tokens). The notebook checks that TRL builds exactly the same labels.
- Memory: AdamW costs ~16 bytes per trainable parameter (2.2 GB for 135M, ~112 GB for 7B). Full fine-tuning only fits the notebook's budget (2 GB) with bf16 weights + Adafactor.
- LoRA starts identical to the base model (`B = 0`) and merges back for free.
- **QLoRA**: the frozen weights are stored in NF4 (16 levels at the quantiles of a normal distribution), with double quantization of the scales and a paged optimizer; the adapters stay in bf16 and every matmul dequantizes first. It saves weight memory, not activation memory, and the adapter is merged into a bf16 copy of the base.
- **The new-token trap**: with tied embeddings and mean initialization, a new token can never be the most likely one if the embeddings are frozen. Fix: train the embedding.
- Training loss does not tell you task accuracy; per-field metrics tell you where the errors are (here, almost always `time`).
- **Model merging**: a fine-tune is a *task vector* $\tau = \theta_t - \theta_0$, and fine-tunes of the same base can be added (task arithmetic) because they stay in one basin of the loss and unrelated updates are nearly orthogonal. TIES and DARE deal with interference when the vectors overlap. With LoRA there is a trap: summing the $A$ and $B$ factors separately (`linear`) adds the cross terms $B_1A_2 + B_2A_1$; `cat` and `svd` give the exact sum.

**Results** (40 test requests; each one is 2.5 points)

| Model | Valid JSON | Exact match | Fields | `time` | Trainable | Train s | Peak VRAM |
|---|---|---|---|---|---|---|---|
| base, zero-shot | 5% | 0% | 2% | 0% | – | – | – |
| base, few-shot (4 examples) | 100% | 17.5% | 73% | 25% | – | – | – |
| full SFT | 100% | 35% | 85.5% | 37.5% | 134.5 M | 72 | 1.19 GB |
| LoRA r=2 | 100% | 20% | 80.5% | 30% | 0.6 M | 58 | 1.46 GB |
| LoRA r=16 | 100% | 32.5% | 86.5% | 32.5% | 4.9 M | 63 | 1.41 GB |
| QLoRA r=16 (NF4 base) | 100% | 37.5% | 86.5% | 45% | 4.9 M | 72 | 1.34 GB |
| LoRA r=64 | 100% | 45% | 89% | 45% | 19.5 M | 109 | 1.25 GB |
| teacher 360M, LoRA r=16 | 100% | **87.5%** | 97.5% | 92.5% | 8.7 M | 230 | 1.27 GB |
| student 135M, human + teacher labels | 100% | **55%** | 91% | 55% | 4.9 M | 87 | 1.54 GB |

- LoRA r=16 trains 3.5% of the parameters and lands within one request of full SFT. The adapter is 19.6 MB (the model in bf16 is 269 MB). Merged in fp32 it changes the logits by at most 1e-4 and keeps the same greedy token at every position.
- **QLoRA** loses no quality: 37.5% exact match against 32.5% for LoRA, with the same 86.5% of fields. That is two requests out of 40, noise, so the honest reading is "the same", not "better". The weights in memory drop from 269 MB to 110 MB (210 linear layers in 4 bits; the tied 28M-parameter embedding stays in bf16), but the training peak barely moves (1.41 → 1.34 GB): at 135M the activations and the 49k-token output layer dominate. Each step is a little slower (72 s against 63 s) because it dequantizes. The decisive saving comes with large models: the frozen weights of a 7B model go from 14 GB to 3.6 GB, a 13B from 26 to 6.7 GB and a 70B from 140 to 36.1 GB.
- **New token `<|json|>`**: with frozen embeddings it is never emitted (0% of answers; rank ~24,000 as the first token, probability ≈ 0, training loss 0.30) and the model degenerates. With a trainable embedding (33.2M parameters) it starts 100% of the answers (p ≈ 0.93) and exact match is back to 32.5%. `trainable_token_indices` fails inside `SFTTrainer` on a tied `lm_head` with the versions used (trl 1.17, peft 0.21).
- **Distillation**: the teacher labels 600 new requests in 104 s, 90% of them exactly right; the student goes from 32.5% to 55%. Honest caveats: the student also took 57 optimizer steps instead of 38, and it learns from labels that are 10% wrong.
- The biggest levers were model size and more data (even machine-labelled); rank came next.

Model merging (both tasks on the same model; booking = exact match on the 40 requests, time = accuracy on 40 time conversions):

| Model | Booking exact | Booking fields | Time conversion |
|---|---|---|---|
| base, no adapter | 0% | 1% | 30% |
| booking adapter only | 32.5% | 86.5% | 27.5% |
| time adapter only | 0% | 0% | 97.5% |
| merge `linear` | 27.5% | 83.5% | 87.5% |
| merge `cat` (task arithmetic) | **32.5%** | 85% | **90%** |
| merge `ties_svd` | 27.5% | 83% | 80% |
| merge `dare_ties_svd` | 30% | 84% | 87.5% |
| multi-task LoRA (both datasets) | **55%** | 89.5% | **97.5%** |

- **The two task vectors barely overlap**: across the 210 adapted matrices the mean cosine similarity is +0.012 (the largest 0.095), and where both updates are large their signs disagree 49% of the time, what unrelated directions give. That is the condition under which plain addition works.
- **Task arithmetic (`cat`) is the best merge**: it keeps the booking task where the booking adapter alone had it (32.5% exact) and converts times at 90%, against 97.5% for the specialist and 30% for the base model. The four merges took 19 s of arithmetic, with no data and no training.
- `linear` loses on both tasks: its update is 91% away from the exact sum (the cross terms). **TIES and DARE did not help** here (their updates are 37% and 50% away from the sum): there was no interference to resolve, so keeping half the coordinates only threw signal away.
- **No skill transfer**: the merged models keep the booking `time` field at 32.5–37.5%, no better than the booking adapter. Merging puts two behaviours in one model; it does not combine them into a new one.
- **Multi-task training wins when you have both datasets**: 55% exact, 70% on the `time` field and 97.5% on conversion, better than every merge on both tasks, with 76 optimizer steps (as many as the two specialists together).
- The notebook ends with "Results" and an "In production" section: the recipe (a held-out set before training, a few-shot baseline, LoRA r = 16 on all linear layers, QLoRA only when the bf16 base does not fit, full fine-tuning only when LoRA plateaus); data beats knobs (add examples of exactly what fails, distil from a stronger model and spot-check its labels); serving adapters (merge into a bf16 base for one task; one base with many adapters in vLLM, SGLang or LoRAX for many; version each adapter with its base, data and evaluation); to combine adapters you cannot retrain together, merge with task arithmetic (`cat` / `svd` in `peft`, or `mergekit` for full models) and run every task's evaluation on the merge; what to monitor (frozen test set, JSON validity, general benchmarks for forgetting, input drift), and the failure modes (LR too low for LoRA, training on the prompt, an untrained new token, merging into the wrong base or merging LoRA factors with `linear`, a teacher's errors copied by the student, trusting the training loss).

---

<a id="08"></a>
## 08 · Preference tuning and RL with TRL: DPO and GRPO
`notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb` · `trl`, `peft` · also as a script: `python scripts/finetune.py dpo` / `grpo`

**What you build.** On notebook 07's booking task, a partially trained SFT model (LoRA, 30% exact), then DPO and DPO + NLL on 400 preference pairs (256 of them with the SFT model's own mistake as the rejected answer). Then GRPO with verifiable rewards on two-digit addition and subtraction, starting from the instruct model, with a **sparse** reward (exact) and a **dense** one (partial credit + format bonus).

**Tools.** `trl.DPOConfig(beta=0.1, loss_type="sigmoid")` / `DPOTrainer` with a `peft_config` (the reference is the same network with the adapter off); DPO + NLL as `loss_type=["sigmoid", "sft"]`, `loss_weights=[1.0, 1.0]` (formerly `rpo_alpha`); `trl.GRPOConfig(num_generations=8, max_completion_length=32, temperature=1.0, beta=0.04)` / `GRPOTrainer(reward_funcs=[reward_exact, reward_partial])` with `reward_weights=[1, 0]` or `[0, 1]`, so both rewards are logged and only one drives the update.

**Key concepts**
- DPO as classification over implicit rewards; the best pairs are your model's own mistakes.
- ***Likelihood displacement***: DPO can lower the probability of the good answer too. Watch `logps/chosen`, not just preference accuracy.
- The NLL anchor on the chosen answer as a cheap, robust default.
- GRPO: group-relative advantage instead of a critic; groups where all samples get the same reward carry no gradient.
- Reward design and **Goodhart's law**: the dense reward is a proxy, not the metric.

**Results**

Booking extraction (40 held-out requests):

| Method | Valid JSON | Exact match | Fields | `time` |
|---|---|---|---|---|
| SFT (start) | 100% | 30% | 86% | 30% |
| + DPO | 95% | **7.5%** | 77.5% | 10% |
| + DPO + NLL | 100% | 32.5% | 86.5% | 32.5% |

- Plain DPO **hurts**: log π(chosen) falls from −1.91 to −5.63 (rejected: −4.81 → −12.74), even though its training metrics look good. With NLL the chosen answer stays put (−1.91 → −2.01) while the rejected one still falls (−4.81 → −8.22).
- Honest: DPO + NLL gains a single request (30% → 32.5%), which is noise. The model lacks a *rule* (12h → 24h), not a preference; more SFT data was the stronger lever in 07.

Arithmetic (100 held-out problems, 40 GRPO steps):

| Model | Exact match | Mean dense reward | Median abs. error | Time |
|---|---|---|---|---|
| instruct, no RL | 5% | 0.090 | 46 | – |
| GRPO, sparse reward | **46%** | 0.553 | 2 | 200 s |
| GRPO, dense reward | 42% | 0.566 | 2 | 194 s |

- Without a single labelled answer, GRPO goes from 5% to 42–46% in about 3.5 minutes. Answers get shorter (~27 → ~13 tokens): the policy learns that a short `96 - 20 = …` is rewarded more often than an explanation that gets cut off.
- The dense reward gives a signal to more groups, but it did not produce a better model: a bit higher on its proxy, a bit lower on exact match. That is the direction Goodhart predicts, but 4 problems out of 100 are within noise.
- Peak VRAM was at most 1.40 GB for everything (0.88 GB for GRPO).

**Next:** [20](#20), where reasoning of the kind GRPO trains is put to work and measured at inference time, then [09](#09).

---

<a id="20"></a>
## 20 · Reasoning and test-time compute: chain of thought, self-consistency, verifiers, reasoning models
`notebooks/03_post_training/20_reasoning_test_time_compute.ipynb` · modules `config`, `evaluation` · LangChain (Ollama), `datasets`, Pydantic

Read right after 08. GRPO in 08 is how reasoning is *trained*; this notebook is about how it is *used* and what it costs. Since 2024 the strongest models (o1/o3, DeepSeek-R1, Claude with extended thinking, Gemini thinking, Qwen3) spend extra tokens thinking before they answer. Here that idea is taken apart lever by lever on 80 problems from the GSM8K test set, with `qwen2.5:3b` served by Ollama, and every lever is measured in accuracy (with bootstrap intervals), output tokens and seconds per problem.

**What you build.**
1. **Direct answer vs chain of thought**: the same model with greedy decoding, once told to give only the final number (16 tokens allowed) and once told to solve step by step and put the number on the last line (512 tokens).
2. **Self-consistency**: five chains of thought per problem at temperature 0.7; for every $k$ from 1 to 5, **maj@k** (majority vote among the first $k$, ties to the earliest answer) and **pass@k** (at least one of the $k$ is right, the ceiling of a perfect verifier).
3. **Best-of-5 with a verifier**: the judge (`qwen2.5:7b` if it is pulled) scores each of the five solutions from 1 to 10 and the top one wins, on the first 40 problems, against the vote and pass@5 on the same 40.
4. **A reasoning model**: `deepseek-r1:1.5b` (Qwen2.5-Math-1.5B fine-tuned on DeepSeek-R1's reasoning traces), which thinks inside `<think>...</think>`, with up to 2,048 tokens.
5. **The cost-accuracy picture**: every strategy as a point on a log axis of output tokens.
6. **Comparative evaluation**: a Bradley-Terry ranking of the strategies from per-problem "matches", with bootstrap intervals, and a pairwise LLM judge asked to compare the chain-of-thought and R1 solutions of 30 problems in both orders.

**Tools.** `datasets.load_dataset("openai/gsm8k", "main", split="test")` (shuffled with seed 0, first 80); a `final_number` function that takes the last number after any `</think>` block (extraction is part of the evaluation); `config.get_llm(num_predict=..., temperature=...)` and `get_llm(model="deepseek-r1:1.5b", num_predict=2048)` with `.batch(..., config={"max_concurrency": 4})`, reading the output tokens from `usage_metadata`; `config.get_judge_llm(num_predict=600).with_structured_output(..., method="json_schema", include_raw=True)` with Pydantic models (`Verdict` with a bounded `reasoning` and a 1–10 `score`; `Choice` with `Literal["A", "B"]`), so a malformed verdict comes back as data instead of an exception; `evaluation.bootstrap_ci` and `paired_bootstrap_diff`; Bradley-Terry fitted in NumPy with minorization-maximization updates and 300 bootstrap resamples.

**Key concepts**
- **Chain of thought as serial computation**: a transformer does a fixed amount of work per token, so writing the intermediate steps as tokens is what lets later steps read earlier results. More tokens buy more serial computation.
- **Test-time compute**: parallel sampling (self-consistency, best-of-N), sequential refinement (self-refine, *budget forcing*) and trained reasoning are all ways of spending more compute per problem at inference time; Snell et al. (2024) treat it as a scaling axis of its own.
- **maj@k vs pass@k**: the gap between them is the most a verifier could recover. When it is small, voting already extracts nearly everything.
- **Verifiers**: an *outcome* reward model scores the final answer, a *process* reward model every step. A verifier only helps if it tells right from wrong better than the vote.
- **Reasoning models and distillation**: R1 learned long reasoning with RL on verifiable rewards (08); R1-Distill copies it into small models by SFT on R1's traces. Whether that holds up outside the training distribution has to be measured, not assumed.
- **Costs and failure modes**: reasoning is paid in output tokens, the slow and expensive ones (06); overthinking, unfaithful chains of thought, fooled or reward-hacked verifiers, and answer-extraction bugs.
- **Bradley-Terry and pairwise judges**: $P(i \succ j) = e^{\beta_i} / (e^{\beta_i} + e^{\beta_j})$, the model behind Chatbot Arena and RLHF reward models (08); an LLM judge has to be checked for position bias by judging every pair in both orders.

**Results**

| Strategy (`qwen2.5:3b` unless noted, 80 GSM8K problems) | Accuracy [95% CI] | Output tokens per problem | Seconds per problem |
|---|---|---|---|
| direct answer | 12.5 % [6.2, 20.0] | 3 | 0.13 |
| chain of thought (greedy) | **85.0 %** [77.5, 92.5] | 246 | 3.63 |
| self-consistency, 5 samples (maj@5) | **90.0 %** [83.8, 96.2] | 1,274 | 19.05 |
| best-of-5 with a 7B judge (first 40) | 85.0 % (majority 90.0 %, pass@5 90.0 %) | 1,274 + judging | 33.40 |
| DeepSeek-R1-Distill 1.5B | 63.7 % [53.8, 73.8]; 16 hit the token limit | 870 | 7.73 |
| Bradley-Terry ranking | self-consistency > chain of thought > R1-Distill > direct, non-overlapping intervals | | |
| pairwise LLM judge, both orders | consistent on only 43 % of 30 pairs | | |

**Honest findings**
- **Reasoning tokens are the biggest lever by far.** The same 3B model goes from 12.5 % to 85.0 % when it is allowed to write its steps, at 246 output tokens instead of 3 (3.63 s instead of 0.13 s per problem). The example in the notebook shows why: each step writes an intermediate result (125, then 175) that the next step reads back.
- **Voting adds a few points at linear cost.** One sample at temperature 0.7 is a little worse than greedy (78.8 % vs 85.0 %); maj@5 reaches 90.0 %, 5.0 points over greedy chain of thought with a paired interval of [+0.0, +11.2], so a real but modest gain, for five times the tokens (about 1,274 per problem). pass@k stops at 91.2 % from $k = 3$, barely above maj@5: on this set voting already gets almost everything a perfect verifier could.
- **The LLM verifier made things worse.** On the first 40 problems, best-of-5 with the 7B judge scored 85.0 %, against 90.0 % for the vote and for pass@5. The judge gives correct solutions an average of 7.30 and wrong ones 6.15, so it barely tells them apart and often picks a fluent wrong solution. All 200 verdicts were valid JSON, so the problem is discrimination, not formatting; and it was the most expensive strategy, with 14.3 s of judging per problem on top of the sampling.
- **The small distilled reasoning model lost to plain prompting.** R1-Distill 1.5B reached 63.7 % (against 85.0 % for chain of thought with the 3B model) while generating 870 tokens per problem, 3.5 times as many, and taking twice as long. 16 of the 80 answers hit the 2,048-token limit without finishing and 15 ended with no number to extract: at 1.5B, the long thinking style often turns into loops and second-guessing (*overthinking*). This is a measurement of this model with this budget, not a verdict on reasoning models.
- **On the cost-accuracy curve the returns diminish fast**: the first ~243 tokens of reasoning bought 72.5 points, the next ~1,000 of self-consistency bought 5 more, the verifier bought nothing, and the reasoning model spent more tokens for less.
- **Bradley-Terry separates the strategies cleanly**: self-consistency +3.09, chain of thought +1.14, R1-Distill −0.58 and direct −3.65, with bootstrap intervals that do not overlap between neighbours. Read as probabilities: on a problem where they differ, chain of thought beats R1-Distill 85 % of the time.
- **The pairwise judge is the cautionary tale.** All 30 pairs were judged in both orders and the judge gave the same verdict only 43 % of the time. It did not lean towards one position overall (52 % of verdicts for A), so this is less position bias than plain noise. When it was consistent and exactly one solution was right, it picked the right one 89 % of the time, but that was only 9 cases.
- The notebook ends with "Results" and an "In production" section: how to choose a strategy (start with chain of thought, or a reasoning model with a capped thinking budget; add self-consistency, with $k$ between 3 and 8, only where the gain is worth $k$× the cost; use best-of-N only with a verifier shown to discriminate: unit tests, a calculator or solver, a schema validator, a trained reward model); how to serve reasoning models (explicit `max_tokens` or a thinking budget, watching the share of answers that hit the limit, robust answer parsing with extraction failures as their own metric, pricing by output tokens); how to evaluate with judges (both orders, consistency reported, the judge validated against ground truth as in [10](#10) and [15](#15), bounded judge reasoning, Bradley-Terry with intervals); what to monitor (tokens per request and their tail, truncated or unparseable answers, accuracy against the target and latency), and the failure modes.

**Next:** back to the main path with [09](#09) (quantization), or [14](#14) to see reasoning inside agents.

---

<a id="09"></a>
## 09 · Quantization in practice
`notebooks/02_pretraining_and_inference/09_quantization.ipynb` · `optimum-quanto`

**What you build.** First under the hood, in a few lines of torch: absmax per tensor, per row and per block, and NF4, on a real weight matrix of SmolLM2-135M (it has an outlier at 15 standard deviations). Then a real model, SmolLM2-360M-Instruct, quantized to int8, float8, int4 and int2.

**Tools.** `optimum.quanto.quantize(model, weights=qint8 | qfloat8 | qint4 | qint2)` and `freeze(model)` (int4/int2 in groups of 128), with `lm_head` kept in bf16. Measurement against the fp32 model on ~500 held-out tokens. Production table: bitsandbytes (QLoRA), GPTQ, AWQ, GGUF (the course's `qwen2.5:3b` is a `Q4_K_M` GGUF), FP8, torchao, KV-cache quantization.

**Key concepts.** Why finer scales reduce error (and cost ~0.25 extra bits per weight with blocks of 64); NF4 puts its levels at the quantiles of a normal distribution; **the metrics that really detect damage**: KL against the original model and top-1 agreement, which is what llama.cpp reports. Perplexity alone hides differences. Memory savings are guaranteed; speed needs a fused kernel.

**Results.** On the real matrix, the relative weight error at 4 bits goes from 58.7% (per tensor) to 16.7% (per row) and 11.4% (blocks of 64); NF4 lowers it to 9.3%. At 8 bits: 3.4% / 0.9% / 0.6%.

| SmolLM2-360M | Weights | Perplexity | KL vs fp32 | Top-1 agreement | Tokens/s |
|---|---|---|---|---|---|
| fp32 (reference, CPU) | – | 17.16 | – | – | – |
| bf16 | 0.72 GB | 17.13 | 0.0010 | 97.2% | 24.0 |
| int8 | 0.41 GB | 17.32 | 0.0048 | 96.6% | 16.8 |
| float8 | 0.41 GB | 17.13 | 0.0094 | 95.1% | 15.2 |
| int4 (g128) | 0.26 GB | 22.16 | 0.27 | 73.3% | 10.5 |
| int2 (g128) | 0.19 GB | ~228,000 | 9.40 | 1.4% | 8.3 |

- bf16 is not exact either: its 97.2% agreement is the noise floor.
- int8 and float8 are almost free. float8 has **the same perplexity as bf16 but twice int8's KL**: perplexity hides what KL shows.
- int4 cuts memory to a third of bf16, but on a model this small it costs a quarter of the top-1 agreement and +29% perplexity. Plain round-to-nearest int2 destroys the model (word salad).
- Honest: on this GPU **the quantized models are slower** than bf16. At batch 1 a 360M model is limited by kernel launches, and quanto adds unpacking work; real speed-ups need a fused kernel for the format.

---

<a id="10"></a>
## 10 · RAG with LangChain, evaluated with RAGAS
`notebooks/04_applications/10_rag.ipynb` · LangChain, FAISS, RAGAS · also as a script: `python scripts/rag_demo.py`

**What you build.** A full RAG system over the course's own documentation (`data/knowledge_base/*.md`, 5 files, ~190 KB): loading, chunking at three sizes, embeddings, a vector index, BM25, hybrid fusion, an LCEL chain with numbered citations, multi-query (RAG-Fusion), and an evaluation that separates retrieval from generation.

**Tools.** `DirectoryLoader` + `TextLoader`; `RecursiveCharacterTextSplitter` (300 / 800 / 1500 characters → 978 / 358 / 174 chunks); `OllamaEmbeddings(nomic-embed-text)` (768 dimensions) and `ChatOllama(qwen2.5:3b)` through `llm_edu_suite.config`; `FAISS` (exact `IndexFlatL2`, embedding in batches of 64); `BM25Retriever` with a tokenizer that lowercases; `EnsembleRetriever` (Reciprocal Rank Fusion, `c = 60`, from `langchain_classic`), checked against a six-line reimplementation; LCEL with `ChatPromptTemplate`, `RunnableParallel`, `.assign`, `StrOutputParser`; RAGAS 0.4 with `llm_factory` over Ollama's OpenAI-compatible API and the metrics `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`; the judge comes from `config.judge_model()` (`qwen2.5:7b` if it is pulled, or whatever `JUDGE_MODEL` says) and the intervals from `evaluation.bootstrap_ci` and `paired_bootstrap_diff`.

**Key concepts.** Dense versus lexical search and why to combine them (technical docs are full of exact strings); chunk size as a hyperparameter; file-level versus evidence-level metrics (does the retrieved *chunk* contain the evidence?); hit@k and MRR; the four RAGAS metrics and which ones need a reference; bootstrap intervals and the paired difference, so you do not read noise; **validate the judge** before trusting it, because a weak judge does not only add noise, it invents effects.

**Results** (20 labelled questions in `data/eval/rag_eval.json`, 4 per document; chunk 800, k = 4)

| Retriever | File hit@1 | Evidence hit@1 | Evidence hit@4 | Evidence MRR |
|---|---|---|---|---|
| dense | 0.95 | 0.65 | 0.85 | 0.73 |
| BM25 | 0.90 | 0.70 | **0.95** | 0.80 |
| **hybrid (RRF)** | 0.95 | **0.80** | 0.90 | **0.84** |
| hybrid + multi-query | 0.95 | 0.75 | 0.90 | 0.82 |

- File hit@4 is 1.00 for all of them (saturated); the differences only appear at evidence level. Hybrid is best at the top of the ranking, and BM25 is surprisingly strong on this corpus.
- **Multi-query did not help** (MRR 0.82, one extra LLM call per question).
- Chunk size does not push every retriever the same way (evidence MRR at 300 / 800 / 1500): dense 0.77 / 0.73 / 0.67 gets worse, BM25 0.74 / 0.80 / 0.82 and hybrid 0.75 / 0.84 / 0.89 get better.
- Without retrieval the model makes up what MCP stands for ("Management and Configuration Protocol").

RAGAS, with `qwen2.5:7b` judging the answers that `qwen2.5:3b` generates (1038 s for 40 answers × 4 metrics, most of the notebook's runtime). In brackets, the 95% bootstrap interval over the 20 questions:

| Configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| dense | 0.87 [0.74, 0.97] | 0.78 [0.64, 0.89] | 0.79 [0.62, 0.92] | 0.95 [0.85, 1.00] |
| hybrid | 0.87 [0.74, 0.96] | 0.75 [0.60, 0.86] | **0.92** [0.80, 0.99] | 0.95 [0.85, 1.00] |
| hybrid − dense (paired) | −0.00 [−0.16, +0.17] | −0.03 [−0.16, +0.10] | **+0.13 [+0.03, +0.25]** | +0.00 [−0.15, +0.15] |

- There is **one** clear difference: context precision rises from 0.79 to 0.92 and its interval excludes zero. Hybrid puts the useful chunks higher, which is what the label-based MRR said too. The other three metrics show no difference: once the right passage is somewhere in the top 4, the 3B generator answers equally well from either list.
- **The judge mattered more than the system.** When the 3B model graded its own answers (the previous version of this notebook), it reported faithfulness 0.38 → 0.54 and answer relevancy 0.44 → 0.30 from dense to hybrid, and gave relevancy 0 to correct answers. With the 7B judge those "differences" vanish inside the intervals. A weak judge does not only add noise: it can invent effects.
- The low rows of the per-question table separate the three failure modes. "Where does `mini_llm.py` save its checkpoint?" is a **retrieval** failure (no evidence in the top 4; precision and recall 0). The question about notebook 03's copy accuracy is a **generation** failure: the evidence was ranked first, both context metrics are 1.00, and the 3B model still answered "I don't know". The DPO + NLL question shows a **label** problem: our evidence-string check does not find the passage, but the judge finds the fact in the context (recall 1.00). Gold labels need checking too.
- Cost: a stronger judge is slower (about 17 minutes with the 7B against about 7 with the 3B). Evaluation is a budget line: a fixed, small, representative set on every change, not everything.

---

<a id="11"></a>
## 11 · Tools, function calling and structured output with LangChain
`notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb` · `langchain-core`, `langchain`, `langchain-ollama`, `pydantic`

**What you build.** Four tools for a small shop (weather, calculator, order lookup, doc search), the tool-calling loop by hand, the same loop with `create_agent`, error handling, structured output with validation and retries, and a trajectory evaluation on labelled requests.

**Tools.** `@tool` (name, docstring and type hints become the schema) and Pydantic `args_schema` (field descriptions, regex `pattern`); `convert_to_openai_tool`; `llm.bind_tools`, `AIMessage.tool_calls`, `ToolMessage`, `tool.invoke(tool_call)`; `langchain.agents.create_agent`; `GenericFakeChatModel` as a scripted model for deterministic tests; `ToolErrorMiddleware`; `with_structured_output(Schema, method="json_schema" | "function_calling" | "json_mode", include_raw=True)` with a Pydantic `field_validator`.

**Key concepts.** The model never executes anything: your code does. Descriptions and schemas are prompts. Tool schemas are re-sent on every call (~390 prompt tokens here). Validation errors go back to the model automatically; tool exceptions only when you opt in. Constrained decoding guarantees the *shape* of the output, not its *content*. Retry *with the error*, not blindly. Evaluate the trajectory, not just the text.

**Results**
- The hand-written loop solves "order A100 plus 21% tax" in two tool steps (`order_status`, then `calculate`) → $145.20. `create_agent` issues the weather and order calls in parallel in one turn.
- With tools attached, `qwen2.5:3b` declined to answer "who wrote Don Quixote?", a question it could have answered alone.
- An invalid `order_id` (`a-100`) comes back as a `ToolMessage` with `status=error`. Without middleware, an exception inside the tool crashes the run; with `ToolErrorMiddleware` the model reads the error and retries with `2**10` → 1024.
- **Retry with feedback**: the summary went from 22 words to 13 and then 13 again; after 3 attempts it still did not meet the 12-word limit, and the function returned `None`. The feedback helped, but not enough: cap the attempts and log the failures.
- **Structured output, 8 labelled emails**: all three methods produce 8/8 valid objects, but field accuracy is 85% for `json_schema` (it returned `order_id = null` on the emails that mention an order), 98% for `function_calling` and 98% for `json_mode`.
- **Agent evaluation, 14 requests**: tool selection 79%, valid arguments 86%, **task success 64%**, median latency 0.8 s, 864 tokens on average. Failures: arithmetic done without the calculator, `c300` in lower case broke the pattern (the model asked the user instead of fixing it), chit-chat that called `search_docs`, a translation it refused, and a multi-step request that skipped the order lookup.

---

<a id="12"></a>
## 12 · Recommender systems: the industry funnel, with LLMs where they help
`notebooks/04_applications/12_recommender_systems.ipynb` · PyTorch, FAISS, LightGBM, LangChain · modules `recsys`, `model`, `evaluation`

**What you build.** An industrial recommender funnel, stage by stage, on real data: MovieLens-1M (1,000,209 ratings from 6,040 users on 3,706 movies), downloaded into `data/cache/` on the first run (the GroupLens license does not allow redistribution, so it is not in the repository). Two baselines (popularity and ItemKNN), SASRec as a language model over movies, content embeddings with and without keywords written by an LLM, a two-tower model with a FAISS index, the union of five candidate sources, a LambdaMART ranker, an exploration slot for new releases, MMR for diversity, and an LLM layer that turns requests into filters and writes explanations grounded in the user's history. One rule runs through the notebook: **a stage only enters the system if it beats the previous one on validation**.

**Tools.** `llm_edu_suite.recsys` (`download_movielens_1m`, `load_movielens_1m`, `temporal_split`, `target_ranks`, `rank_metrics`); notebook 02's `TransformerLM` as SASRec (2 layers, 64 dimensions, a 200-movie window, full softmax, early stopping on validation nDCG); `OllamaEmbeddings(nomic-embed-text)` for each movie's text; a PyTorch two-tower model with id dropout; a FAISS HNSW index against exact search; `lightgbm` with `objective="lambdarank"`; `evaluation.paired_bootstrap_diff` for every comparison; LangChain with `with_structured_output(method="function_calling", include_raw=True)` and an unfiltered search as the fallback; `scripts/generate_item_keywords.py` for the keywords (already generated in `data/recsys/ml1m_llm_keywords.json`) and `data/recsys/conversational_eval.json` (20 requests with checkable constraints).

**Key concepts**
- The funnel: candidate generation (measured by the recall of the candidate set), ranking (nDCG@10 of the list) and re-ranking (measured by what it trades). The ranker can only order what retrieval brought.
- An honest protocol: *leave-last-two* in time order (the last event is test, the one before validation), **full ranking** over all 3,706 movies instead of 100 sampled negatives (Krichene and Rendle, 2020), 10% of the movies (370) held out as cold new releases (747 of the 6,040 test targets are cold), and a paired bootstrap for every comparison.
- Sequential recommendation as next-token prediction, and why the full softmax beats one sampled negative (at the scale of millions: sampled softmax with the logQ correction).
- Cold start: collaborative models never hit a new movie; content places it and exploration gives it exposure.
- Learning to rank (pointwise, pairwise, listwise) and the recipe that makes a second-stage ranker work: many queries from a later period and features the first stage cannot see.
- Use the LLM for the language around the recommender, not for ranking.

**Results** (test, 6,040 users, full ranking)

| System | hit@10 | nDCG@10 | warm hit@10 | cold hit@10 |
|---|---|---|---|---|
| popularity | 0.036 | 0.018 | 0.041 | 0 |
| ItemKNN | 0.125 | 0.067 | 0.143 | 0 |
| content (metadata text, `nomic-embed-text`) | 0.033 | 0.017 | 0.031 | **0.047** |
| content + LLM keywords | 0.023 | 0.012 | 0.023 | 0.020 |
| two-tower (with id dropout) | 0.194 | 0.109 | 0.221 | 0.001 |
| SASRec | 0.293 | 0.172 | 0.334 | 0 |
| **LambdaMART ranker** | **0.307** | **0.181** | **0.349** | 0.009 |
| ranker + 1 exploration slot | 0.300 | 0.179 | 0.336 | 0.044 |

- **SASRec** is the best single model: more than twice ItemKNN and eight times popularity, inside the published range for this dataset with full ranking. Early stopping kept step 5,500. It never hits a new movie, because those output rows were never trained.
- **The LLM keywords made content worse** (hit@10 0.033 → 0.023), and validation rejected them. Many are generic ("nostalgia", "coming-of-age") and dilute the signal of the title and genres; some are wrong (*The Ref* is described as a high-school sports film). In the two-tower model the two texts tie on validation (nDCG@10 0.1122 against 0.1110), and the simpler one stays.
- **Candidates.** The union of five sources contains the target for 68.4% of users (SASRec's top 100 alone, 60.8%), and it is the only way new releases get in (cold recall 0.149). For the other 32% no ranker can succeed. The HNSW index returns 98.7% of the exact top 100 at ~0.01 ms per user, with no speed gain at 3,706 items: you see the price of the approximation without its benefit.
- **The ranker.** A first attempt, trained on a single positive per user, did not beat SASRec and failed the gate. With the RecTools and Kaggle OTTO recipe (retrieval models fit on the history minus the last 8 events: 48,320 events → 35,762 queries and 1.82M rows, 50 sampled negatives, plus features SASRec cannot see: explicit ratings, demographics, co-visitation, days since the previous event, trending) it passes the gate on validation (nDCG +0.0059 [+0.0022, +0.0096]) and on test lifts hit@10 by 1.4 points [+0.7, +2.1] and nDCG by +0.0092 [+0.0055, +0.0128]. The SASRec features carry most of the gain; then come `days_since_prev`, `is_cold`, occupation, co-visitation, year and the item's mean rating.
- The LLM-keyword feature in the ranker gives +0.0012 [−0.0012, +0.0035]: the interval includes zero, and it is not worth an hour of LLM time per catalog refresh.
- **Exploration.** The slot is filled by the two-tower model, the best at ranking new releases on validation. One slot lifts cold hit@10 from 0.009 to 0.044 for 0.7 points overall (0.307 → 0.300); two slots, 0.290 / 0.074; three, 0.275 / 0.094. Where to stop is a product decision, and a bandit would make it adaptively.
- **Diversity (MMR with genre Jaccard).** λ = 0.8 raises intra-list diversity from 0.659 to 0.719 for 0.4 points of hit@10 (0.307 → 0.303); λ = 0.6 gives 0.775 / 0.288 and λ = 0.2 gives 0.852 / 0.222. Offline, diversity can only look neutral or worse; its benefit is measured online.

The LLM layer (`qwen2.5:3b`):

| Measure | Result |
|---|---|
| constraint satisfaction@5, 20 checkable requests | embedding search 0.60 [0.46, 0.74] · **LLM → filters 0.87 [0.80, 0.93]** |
| structured-output failures | 1 of 221 (the fallback kept the system running) |
| hit@10 inside the filtered set (200 users, request built from the genre and decade of their real next movie) | popularity 0.285 · similarity to the request 0.325 · **SASRec 0.555** |
| grounded explanations (anchors = the history movies closest in SASRec's embedding space), 30 users | cite an anchor 97% · mention a movie not in the history 7% (2/30), against 13% (4/30) without anchors |

- The LLM wins exactly where embeddings are weak: date ranges and negations ("a 90s crime movie like Pulp Fiction" goes from 0.0 to 0.8). The remaining misses (asking for Animation without Children's, putting the reference title in the wrong field) are fixed with better field descriptions.
- The division of labour is clear: the LLM understands *what* is asked, and the history decides *which* of the matching movies.
- Ungrounded explanations invent attributes even when they cite no titles (*The Usual Suspects* gets war elements); grounded ones are true but not always convincing, because SASRec's embeddings encode co-watching, not content. With 30 users, 7% against 13% is a hint, not proof.
- **The honest conclusion:** small LLMs do not beat collaborative models at ranking, but they are excellent at the language around the recommender.

---

<a id="13"></a>
## 13 · Modern LLM application patterns with LangChain
`notebooks/04_applications/13_modern_llm_applications.ipynb` · `langchain-core`, `pydantic`

**What you build.** Six production patterns, each with its own metric: structured extraction, zero-/few-shot classification, map-reduce summarization, caching (exact and semantic), prompt-injection defence, and batch processing with token and cost accounting. Everything at temperature 0 on small hand-labelled sets (8 to 20 items).

**Tools.** Pydantic + `with_structured_output` (`json_schema` and `function_calling`); `FewShotChatMessagePromptTemplate` and a `Literal` label; map-reduce in LCEL with `RecursiveCharacterTextSplitter` and `.batch`; `set_llm_cache(InMemoryCache())` and a semantic cache written against `BaseCache` with `nomic-embed-text`; an LLM classifier with a structured verdict; `.batch(max_concurrency=n)` and `get_usage_metadata_callback()`.

**Key concepts.** With `json_schema`, `ChatOllama` passes the schema only to the decoder, so the model never *reads* the field descriptions; few-shot examples are a prompt change that has to be evaluated; long inputs are silently truncated (check `usage_metadata["input_tokens"]`); a cache serves wrong answers just as fast as right ones; injection is managed in layers, not solved; project cost and duration before scaling up.

**Results**
- **Extraction (8 records)**: every variant returns 8/8 valid objects, but all fields are right in 62% with `json_schema`, 88% with `json_schema` + the rules in the system message, and 88% with `function_calling`.
- **Classification (20 tickets)**: zero-shot 100%, few-shot 100% in the committed run (95% in an earlier run, with a shipping ticket sent to technical). The examples bought nothing.
- **Summarization**: the document is ~4,135 tokens, but the single "stuff" call processed only 2,050 (truncated without any error): 9/13 topics (it lost the beginning: tokenization, attention, MoE) and 779 words instead of a 150-word paragraph; 6/13 in another run. Map-reduce (6+1 calls): 11/13 topics, 210 words, 12× compression, 30.4 s versus 15.7 s.
- **Exact cache**: 0.54 s for the first call, 1–2 ms for the next ones. It also caches the 3B model's *wrong* answer about the KV-cache. **Pitfall found by measuring:** `ChatOllama`'s `llm_string` (the model part of the cache key) contains neither the model name nor the temperature, so a call at T=0.5 was served the T=0 answer (3 ms, a hit).
- **Semantic cache**: paraphrases score cosine 0.73–0.95 and near misses 0.56–0.93 (they overlap). To keep false hits at zero the threshold must be 0.94, and then only 20% of paraphrases hit (3 hits in 25 requests).
- **Prompt injection (10 documents)**: attack success 90% with the naive prompt, **100% when delimiting** (it did not help at all with this model) and 40% with delimiting + a reminder after the document ("sandwich"). The LLM guard catches 10/10 attacks with 0/10 false positives on its own set and blocks 8/10 of the injected documents. Combined: ≈ 0.2 × 0.4 = 8% expected success, not zero.
- **Batch (40 records)**: 8.5 s with concurrency 1 and 8.3 s with 4 (~210 ms per record; a single local Ollama barely gains). ~131 input and 9 output tokens per record → ~$25 per million records at $0.15/M in and $0.60/M out; locally, a million records would take 57 hours.

---

<a id="14"></a>
## 14 · Agentic AI with LangGraph
`notebooks/05_agents_and_mcp/14_agentic_ai.ipynb` · `langgraph`, `langchain`

**What you build.** A ticket-triage `StateGraph`, a ReAct agent, guardrails with explicit stop reasons, an approval gate for refunds, short- and long-term memory, plan-and-execute as an explicit graph, reflection guided by unit tests, a router with specialist subgraphs, and an evaluation of ReAct versus plan-and-execute.

**Tools.** `StateGraph`, `START`/`END`, `add_conditional_edges`, reducers (`Annotated[..., operator.add]`, `add_messages`), `stream(stream_mode="updates")`, `draw_mermaid()`; `create_agent`; `recursion_limit`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` and a custom `AgentMiddleware` (`WallClockLimit`); `interrupt` + `Command(resume=...)` and `HumanInTheLoopMiddleware`; `InMemorySaver` + `thread_id`; `InMemoryStore` with semantic search, `ToolRuntime` and `@dynamic_prompt`; `Command(goto=...)` and compiled graphs as subgraphs.

**Key concepts.** Workflow versus agent (use the simplest design that works); every agent needs limits and a **stop reason**; the approval lives in your code, not in the prompt; memory = checkpointer (per thread) + store (across threads), and long-term memory is retrieval, so you measure its recall; reflection needs a reliable verifier; routing is classification; evaluate the *path* (tool precision/recall), not just the answer.

**Results**
- **Guardrails**: the four stops fire with their own reason (`recursion_limit`, `model_call_limit`, `tool_call_limit`, `timeout`) against a fake model that loops forever.
- **Human in the loop**: with the bare primitive, the 20-dollar refund is approved and the 450-dollar one rejected; with `HumanInTheLoopMiddleware` and a "up to 40 dollars" policy, A100 (20) is approved and B200 (45.5) rejected. Only the approved refund was issued.
- **Memory**: the same thread remembers the order (8 messages saved); a new thread does not. Long-term memory has recall@1 **5/6**; a fact saved in one thread is used in another.
- **Reflection with tests (8 functions)**: 4/8 pass on the first draft and **still 4/8** with up to 3 rounds of feedback. When the 3B model does not know how to fix the bug, seeing the failing assert does not help.
- **Router**: 67% on 12 requests (0.23 s per call). Greetings and thank-yous went to `support`, and "Where is my order B200?" too.
- **ReAct versus plan-and-execute (10 multi-step tasks)**:

| Design | Success | Tool precision | Tool recall | Model calls | Time per task |
|---|---|---|---|---|---|
| **ReAct** | **80%** | 90% | 70% | 2.1 | 1.8 s |
| plan-and-execute | 60% | 87% | 90% | 6.6 | 3.0 s |

Honest: with this model and these short tasks **planning did not pay off**. ReAct did better with a third of the calls (qwen2.5:3b issues parallel tool calls). The plan forces every sub-task to happen (higher recall), but it adds redundant steps and every hand-off loses information (an invented status, `"shased"`, and wrong arithmetic in the example). ReAct's failures were arithmetic done in its head and a policy number invented instead of looked up. With 10 tasks, one success is 10 points.

---

<a id="15"></a>
## 15 · LLMOps with LangSmith: tracing, monitoring and evaluation
`notebooks/06_llmops/15_observability_llmops.ipynb` · `langsmith`, LangGraph · **LangSmith is optional**: without a key the app and the local evaluators run, and the cells that need the server are skipped

**What you build.** A small support assistant (a LangGraph RAG graph over `data/sample_docs/` plus an order tool), synthetic traffic, and then the work of an on-call engineer with the LangSmith SDK: trace trees, threads, percentiles, SLOs, online scores, PII redaction, sampling, export, and an offline experiment comparing two versions.

**Tools.** Automatic tracing of LangChain/LangGraph (`LANGSMITH_TRACING=true`); `@traceable` (`run_type="llm"`, `"tool"`, `"retriever"`), `tracing_context`, `get_current_run_tree`, `langsmith_extra={"run_id": ...}`; `client.runs.query(...)` (async, with filters such as `has(tags, "traffic")`); `thread_id` metadata; `client.create_feedback`; `langsmith.anonymizer.create_anonymizer` and `Client(anonymizer=...)`; `tracing_sampling_rate`; `langsmith.evaluate` with datasets and experiments; `InMemoryVectorStore`.

**Key concepts.** Project, trace, run, tags/metadata, thread, feedback, dataset → experiment; why p50/p95/p99 and not the mean; SLOs as a contract; cheap heuristics on every trace to decide which ones a human should open; redact before uploading (the model still sees the raw text); sampling keeps or drops whole traces; check the LLM judge against deterministic evaluators.

**Results**
- One `graph.invoke` produces a tree of 7 runs (graph, nodes, retriever, prompt, `ChatOllama` with 115+12 tokens).
- **Traffic**: 18 requests (2 to unknown orders, on purpose), 18 traces and 112 runs. Per request: **p50 0.22 s, p95 2.80 s, p99 4.24 s**, mean 0.93 s, error rate 11.1% (tool: 33%), 104 tokens per request (1605 input and 270 output in total), ~$0.00002 per request at illustrative prices (LangSmith leaves `total_cost` empty for a local model). The prompts are six times longer than the answers: the context is the cost lever.
- **SLOs**: p95 < 4 s **OK**, error rate < 5% **VIOLATED** (by design), tokens < 600 **OK**. The latency SLO depends on how busy the GPU is (the notebook saw p95 between ~3 and ~10 s across runs).
- **Online groundedness**: mean 0.81 over 16 answers. The three lowest are order-tool answers: A101 and A103 are **real bugs** (the model had the status and said it had no tracking information); A100 (0.43) is a false alarm.
- **PII**: email, phone, card and API key are replaced in all 8 runs of the trace; names like "Ana" get through.
- **Sampling** at 0.25: 13 of 40 traces kept (it varies between runs).
- **Offline experiment (12 examples)**:

| Version | exact_match | contains_fact | correctness (LLM judge) |
|---|---|---|---|
| closed_book | 0.00 | 0.17 | 0.17 |
| **rag** | 0.58 | **1.00** | **0.83** |

The judge agrees with `contains_fact` on 20 of 24 answers: once it catches a false claim the keyword check missed, and twice it is wrong (it rejects a correct answer for adding the time zone and accepts one with no number at all). `exact_match` measures phrasing, not correctness.

---

<a id="16"></a>
## 16 · MCP with the official SDK
`notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb` · `mcp` (1.x SDK), `langchain-mcp-adapters` · script `scripts/mcp_server.py`

**What you build.** An MCP server with FastMCP (tools, resources, templates and prompts), a client session in memory, the raw JSON-RPC messages over stdio "on the wire", the two kinds of errors, the knowledge-base server (`scripts/mcp_server.py`) over stdio, a LangChain agent that uses its tools, security controls, and an evaluation of the server.

**Tools.** `mcp.server.fastmcp.FastMCP` with `@server.tool(annotations=ToolAnnotations(...))`, `@server.resource` (fixed and templated URIs) and `@server.prompt`; `create_connected_server_and_client_session` (in memory, for tests); `ClientSession`, `stdio_client`, `StdioServerParameters`, `McpError`; `MultiServerMCPClient(tool_name_prefix=True)`, `client.session(...)` + `load_mcp_tools(session)`; `create_agent`; `HumanInTheLoopMiddleware` with `InMemorySaver` and `Command`; tool-call interceptors that scan and wrap results.

**Key concepts.** Host, client and server; three primitives with three different controllers (the model picks tools, the application loads resources, the user picks prompts); JSON-RPC 2.0, handshake (`initialize` + `notifications/initialized`), requests versus notifications; tool errors (`isError`, they go to the model) versus protocol errors (JSON-RPC `error`, client bugs); stdio: never write to stdout; annotations are *hints*; allowlist, namespacing, confirmation of anything not read-only, injection through results and *tool poisoning*.

**Results**
- **14/14 conformance checks** pass (handshake, listings, valid and invalid calls, resources, template, prompt, ping).
- Implementation details the notebook surfaces: FastMCP 1.x reports an unknown tool as an `isError` result (the spec describes it as a protocol error), uses `-32602` for an unknown method (JSON-RPC defines `-32601`) and code `0` for a missing resource.
- `search_docs` (BM25) puts the right document first in **6/6** questions (top-3: 6/6); an easy set.
- **Agent**: the key fact is in 6/6 answers and the right file is cited in **5/6** (the express-shipping answer has no citation); median 0.8 s per question.
- **Latency of a `tools/call`** (p50): 1.4 ms in memory, 2.7 ms over stdio, 3.6 ms through the LangChain adapter with one session, and **~2.3 s with the default session-per-call tool**, because it starts a subprocess and repeats the handshake every time. Keep the session open.
- **Security**: the poisoned description (`<IMPORTANT>`) is detected and that tool never reaches the model; the injected result arrives inside an `<untrusted_tool_result>` envelope with a warning; the reviewer rejects `docs_create_ticket` and the destructive tool never runs.

---

## What next

- **Consolidate:** every notebook ends with *Takeaways*. The scripts package the same pipelines: `scripts/finetune.py` (SFT, DPO, GRPO), `scripts/rag_demo.py` (RAG and its evaluation) and `scripts/mcp_server.py`.
- **Change the data:** pass your own documents to `scripts/rag_demo.py --docs` or `scripts/mcp_server.py --docs`, and your own text to notebook 00.
- **Connect the server to a real host:** `claude mcp add llm-edu-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py`, or start it with `--http`.
- See the [glossary](glossary.md) for any term, the [theoretical foundations](foundations.md) for the theory with its equations, and the [module reference](module_reference.md) for the API.
