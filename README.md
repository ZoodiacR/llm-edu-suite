# LLM Edu Suite

**Learn LLM engineering in two halves: build a language model from scratch in PyTorch to understand how it works, then build real applications with the tools the industry uses — Hugging Face, LangChain, LangGraph, LangSmith, RAGAS and the Model Context Protocol.** Every notebook explains the idea, runs the code, and ends with an evaluation using the metrics practitioners actually report.

🇪🇸 **[Documentación completa en español](docs/es/README.md)** · 🇬🇧 **[Full documentation in English](docs/en/README.md)**

![python](https://img.shields.io/badge/python-3.11%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c) ![hf](https://img.shields.io/badge/HF-transformers%20·%20peft%20·%20trl-yellow) ![langchain](https://img.shields.io/badge/LangChain-LangGraph%20·%20LangSmith-1c3c3c) ![mcp](https://img.shields.io/badge/MCP-official%20SDK-6b4fbb) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

- **32 notebooks** grouped by topic, all executed with their outputs, plots and an evaluation section
- **Runs on your own laptop**: local models through [Ollama](https://ollama.com) (`qwen2.5:3b`, `nomic-embed-text`) and Hugging Face SmolLM2; an NVIDIA GPU makes it fast, CPU also works. No API keys required (LangSmith is optional and free; an optional Anthropic key runs notebook 27's agent on Claude Haiku 5.5, about US$ 0.42 per full run, instead of a much slower local model).
- **Honest results**: when a technique does not help in our setup, the notebook shows it and explains why
- **Tested**: fast tests with fake LLMs in CI on every push, integration tests against the local models

> **v1 vs v2.** The first version ([`v1.0-from-scratch`](https://github.com/ZoodiacR/llm-edu-suite/tree/v1.0-from-scratch)) implemented *everything* by hand, including RAG, agents, tracing and MCP. v2 keeps the from-scratch part where seeing the internals is the point (the model itself), and teaches the application layer with the frameworks you will use at work.

---

## Topic map

The number is the suggested study order; the folder is the topic.

| Topic | Notebook | What you learn | Tools |
|---|---|---|---|
| **Start here** | [00 · transformers, the big picture](notebooks/00_start_here/00_transformers_big_picture.ipynb) | read this first: one sentence followed through a real model (SmolLM2-135M): tokens, embeddings, attention maps, the logit lens layer by layer, the next-token distribution and temperature, where the parameters live | `transformers` |
| | [01 · mini LLM end to end](notebooks/00_start_here/01_mini_llm_end_to_end.ipynb) | a whole GPT in one notebook: BPE → RoPE/RMSNorm/SwiGLU → training → sampling → checkpoint | PyTorch |
| **Architecture** | [02 · tokenization](notebooks/01_architecture/02_tokenization.ipynb) | byte-level BPE, vocabulary trade-offs, fertility | PyTorch (from scratch) |
| | [03 · attention, RoPE, norms](notebooks/01_architecture/03_attention_rope_norm.ipynb) | RMSNorm, RoPE, MHA → GQA → MQA → MLA, KV-cache | PyTorch (from scratch) |
| | [04 · efficient attention](notebooks/01_architecture/04_efficient_attention.ipynb) | sliding window, Longformer, block-sparse, linear attention, induction heads | PyTorch, FlexAttention |
| | [05 · FFN & MoE](notebooks/01_architecture/05_ffn_moe.ipynb) | SwiGLU, Mixture of Experts, aux-loss vs aux-free balancing | PyTorch (from scratch) |
| | [06 · encoders & encoder-decoders](notebooks/01_architecture/06_encoders_and_encoder_decoders.ipynb) | bidirectional attention and masked LM from scratch; classifying 77 intents with a fine-tuned ModernBERT, embeddings + logistic regression, or an LLM; Flan-T5 cross-attention | PyTorch, `transformers`, scikit-learn, LangChain |
| **Pre-training & inference** | [07 · data engineering](notebooks/02_pretraining_and_inference/07_data_engineering.ipynb) | exact and MinHash/LSH deduplication from scratch, Gopher/C4 rules, the FineWeb-Edu quality classifier, 13-gram decontamination, raw vs clean training, synthetic data | FineWeb, `transformers`, PyTorch |
| | [08 · pre-training](notebooks/02_pretraining_and_inference/08_pretraining.ipynb) | the training pipeline, LR schedules, irreducible loss, scaling curves | PyTorch (from scratch) |
| | [09 · systems](notebooks/02_pretraining_and_inference/09_systems_memory_and_parallelism.ipynb) | memory anatomy, bf16, activation checkpointing, fused attention kernels, profiling, CUDA graphs, MFU, real DDP, ZeRO/FSDP | PyTorch, `torch.distributed` |
| | [10 · inference & serving](notebooks/02_pretraining_and_inference/10_inference_serving.ipynb) | sampling, KV-cache, speculative decoding, batching, TTFT; loading SmolLM2's real weights into our model | PyTorch (from scratch), `transformers` |
| | [11 · quantization](notebooks/02_pretraining_and_inference/11_quantization.ipynb) | int8 / float8 / int4 / int2 on a real model, measured with KL and top-1 agreement | `optimum-quanto` |
| | [12 · serving under load](notebooks/02_pretraining_and_inference/12_serving_and_load.ipynb) | TTFT / TPOT / throughput, prefill vs decode, prefix caching, request slots and batching, Little's law, GGUF Q4 / Q8 / FP16, a gateway with fallbacks, a cost-aware 3B → 7B cascade | Ollama, `httpx` |
| **Post-training** | [13 · SFT, LoRA, distillation](notebooks/03_post_training/13_finetuning_sft_lora.ipynb) | chat templates and loss masking, full SFT vs LoRA vs QLoRA (4-bit NF4), the new-token trap, rank ablation, teacher → student, model merging (task arithmetic, TIES, DARE) | `transformers`, `peft`, `trl`, `bitsandbytes` |
| | [14 · DPO & GRPO](notebooks/03_post_training/14_preference_rl_dpo_grpo.ipynb) | preference tuning, likelihood displacement, DPO + NLL, RL with verifiable rewards | `trl` |
| | [15 · reasoning & test-time compute](notebooks/03_post_training/15_reasoning_test_time_compute.ipynb) | chain of thought, self-consistency (maj@k / pass@k), best-of-N with an LLM verifier, a distilled reasoning model, Bradley-Terry rankings, judge position bias | LangChain, Ollama |
| **Applications** | [16 · tools & structured output](notebooks/04_applications/16_agents_tools_structured_output.ipynb) | how tool calling works, `bind_tools`, the loop by hand, `create_agent`, Pydantic outputs | LangChain |
| | [17 · embeddings & rerankers](notebooks/04_applications/17_embeddings_and_rerankers.ipynb) | bi-encoders vs cross-encoders, InfoNCE by hand, contrastive fine-tuning with hard negatives, Matryoshka, int8 / binary quantization with rescoring, hybrid retrieval and rerankers, on BEIR SciFact | `sentence-transformers`, Ollama |
| | [18 · RAG](notebooks/04_applications/18_rag.ipynb) | loaders, splitters, FAISS, BM25, hybrid RRF, LCEL chains with citations, RAGAS | LangChain, RAGAS |
| | [19 · LLM application patterns](notebooks/04_applications/19_modern_llm_applications.ipynb) | extraction, classification, map-reduce summarization, caching, prompt-injection defense, batch jobs | LangChain |
| | [20 · DSPy prompt optimization](notebooks/04_applications/20_dspy_prompt_optimization.ipynb) | prompts as programs: signatures, modules and adapters, random vs nearest-neighbour few-shot, the BootstrapFewShot, MIPROv2 and GEPA optimizers, a retrieval-first router, an optimized program saved as JSON; on notebook 06's Banking77 sample | DSPy, Ollama, `sentence-transformers` |
| | [21 · clustering & topic modeling](notebooks/04_applications/21_clustering_topic_modeling.ipynb) | k-means vs UMAP + HDBSCAN, choosing k without labels, c-TF-IDF keywords and LLM-named topics, detecting emerging intents in production logs | `umap-learn`, scikit-learn, LangChain |
| | [22 · multimodal](notebooks/04_applications/22_multimodal.ipynb) | ViT, CLIP zero-shot and linear probes, a vision-language model as classifier and document reader, image token cost, typographic attacks | `transformers`, Ollama (`qwen2.5vl`) |
| | [23 · recommender systems](notebooks/04_applications/23_recommender_systems.ipynb) | the industry funnel on MovieLens-1M: SASRec, two-tower + FAISS, LambdaMART ranker, exploration and diversity, LLM requests and explanations | PyTorch, FAISS, LightGBM, LangChain |
| | [24 · online learning & drift](notebooks/04_applications/24_online_learning_drift.ipynb) | models aging on a stream, prequential evaluation, weekly / sliding / online updates, PSI, drift-triggered retraining, embedding drift tests (MMD, classifier two-sample) for LLM apps | NumPy, `sentence-transformers` |
| **Agents & MCP** | [25 · agentic AI](notebooks/05_agents_and_mcp/25_agentic_ai.ipynb) | StateGraph, guardrails, human-in-the-loop, memory, plan-and-execute, reflection, routing, supervisor vs handoffs, sandboxed code execution | LangGraph |
| | [26 · Model Context Protocol](notebooks/05_agents_and_mcp/26_mcp_model_context_protocol.ipynb) | MCP from the wire up, a FastMCP server, an agent using MCP tools, security, sampling / elicitation / roots, a remote server with OAuth 2.1 bearer tokens | `mcp` SDK, `langchain-mcp-adapters` |
| | [27 · agent harnesses](notebooks/05_agents_and_mcp/27_agent_harness.ipynb) | what turns a model into a coding agent (Claude Code, Codex CLI, opencode): mini-swe-agent read function by function, a sandboxed environment with a permission policy and workspace context, 16 coding tasks scored by hidden tests (3 runs each) on Claude Haiku 5.5 or any litellm model, `AGENTS.md` project memory, a prompt injection planted in the repository, a modern harness profile (native tools, lint and verification gates, loop detection, masking) with an ablation | mini-swe-agent, litellm, Anthropic API or Ollama |
| **LLMOps** | [28 · observability & evaluation](notebooks/06_llmops/28_observability_llmops.ipynb) | tracing, latency percentiles, SLOs, feedback, PII anonymization, datasets + `evaluate()` | LangSmith |
| | [29 · evaluation & tracing frameworks](notebooks/06_llmops/29_eval_and_tracing_frameworks.ipynb) | notebook 15's GSM8K evaluation as an Inspect AI task (solvers, scorers, epochs, logs); OpenTelemetry tracing to a self-hosted Phoenix server: latency and tokens per step, annotations on traces, tracing overhead | Inspect AI, Arize Phoenix, OpenInference |
| | [30 · safety & guardrails](notebooks/06_llmops/30_safety_guardrails.ipynb) | refusals vs over-refusal, published jailbreaks, Llama Guard 3 as an input filter, safety prompts, layered defences, validated judges | Ollama, Llama Guard 3 |
| **Capstone** | [31 · a reasoning model from scratch (mini-R1)](notebooks/07_capstone/31_mini_r1_capstone.ipynb) | the whole R1 recipe on the course's own 1.6M-parameter model: pretraining on a verifiable task, R1-Zero (GRPO written by hand, straight from the base), an SFT cold start, R1 (SFT then GRPO), a same-budget SFT control, accuracy by chain length | PyTorch (from scratch) |

Detailed notebook-by-notebook guide: [English](docs/en/notebooks_guide.md) · [Español](docs/es/guia_notebooks.md).

## Selected results

Measured in the executed notebooks on a laptop with an RTX 3060 (6 GB). Models are small on purpose (a 3B chat model, 135M–360M models for training), so the absolute numbers matter less than the comparisons. LLM-judged numbers vary slightly from run to run.

| Notebook | Result |
|---|---|
| 00 · transformers big picture (SmolLM2-135M) | Spanish needs **1.73×** the tokens of English for the same sentences; one attention head (layer 13, head 8) puts 0.73 on "animal" and 0.02 on "street" when reading "it", while **61% of all attention lands on the first token** (the attention sink); the logit lens shows the right answer reaching top-1 only in layers 28–30 of 30; 59% of the parameters are feed-forward (75% in Qwen2.5-7B) |
| 01 · mini LLM | a 1.9M-parameter GPT trains in ~20 s on the GPU and ends **0.6% above its corpus's theoretical minimum loss**; 82% of sampled sentences are grammatical |
| 04 · efficient attention | long-range copy accuracy: **dense 99.9%, block-sparse 88.7%, hybrid 48.4%, sliding window and linear ~0%**; at 300 steps even dense is at 0.3% (the induction-head phase transition) |
| 06 · encoders | from scratch, bidirectional context recovers the first token of a word **100% vs 2.9%** for a decoder. On Banking77 (77 intents): fine-tuned ModernBERT **93.0%**, nomic embeddings + logistic regression **93.3%**, `qwen2.5:3b` 61.9% few-shot; function calling left 181 of 231 outputs invalid, constrained decoding none |
| 07 · data engineering | MinHash + LSH from scratch follows the theoretical S-curve (100% of templated copies and 1% rewrites caught, 0 false merges); rules remove 12% of FineWeb; decontamination finds 50/50 planted leaks plus 29 natural overlaps; a model trained on the raw crawl memorizes 10×-repeated pages (loss 4.35 vs 5.28 for the deduplicated model) |
| 09 · systems | memory matches the 16-bytes-per-parameter rule to the MB; fused attention needs **0.22 GB at 16k tokens** where naive attention runs out of memory; CUDA graphs **4.1×** faster; MFU 32%; two-process DDP reproduces large-batch training to 7e-9 |
| 10 · inference | speculative decoding accepts **98% of draft tokens, ~4.6 tokens per target forward pass**. SmolLM2-135M's published weights run in our from-scratch `TransformerLM` with **identical greedy output** to Hugging Face (logits within 3e-5) |
| 11 · quantization (SmolLM2-360M) | int8 and float8 keep ~96% top-1 agreement at 0.41 GB; int4 drops to 73% (KL 0.27); int2 breaks the model. Perplexity hides what KL shows |
| 12 · serving (Ollama) | prefill runs **35× faster** than decode; prefix caching cuts TTFT from 654 to 39 ms; one request slot turns concurrency into a queue (TTFT 13.4 s at 8 in flight) while four slots double throughput (171 tok/s); Q4_K_M decodes 2× faster than FP16 in a third of the memory for −1.5 points on ARC-Easy; a 3B → 7B cascade beats both models (85% vs 82.5% / 78.8%) |
| 13 · SFT & LoRA (SmolLM2-135M) | few-shot 17.5% → **full SFT 35%**, LoRA r=16 32.5% training 3.5% of the parameters, r=64 45%. A new special token with frozen embeddings is **never emitted (0%)**; with `modules_to_save` it is emitted **100%** of the time |
| 13 · QLoRA | the same LoRA on a 4-bit NF4 base: **no loss of quality** (37.5% vs 32.5% exact, same 86.5% of fields), weights 269 → 110 MB; for a 7B model the frozen weights drop from 14 GB to 3.6 GB |
| 13 · distillation | a 360M teacher (87.5%) labels 600 extra prompts; the 135M student rises from 35% to **55%** |
| 13 · model merging | two LoRA adapters with nearly orthogonal task vectors (cosine 0.01): task arithmetic keeps the booking task at 32.5% and converts times at **90%** (specialist 97.5%) with no training; `linear`, TIES and DARE do worse; multi-task training beats every merge (55% + 97.5%) |
| 14 · DPO & GRPO | plain DPO wins its training metric but **drops exact match from 30% to 7.5%** (likelihood displacement); DPO + NLL holds 32.5%. **GRPO lifts arithmetic from 5% to 46% in 40 steps** with no labels, only a verifier |
| 15 · reasoning (GSM8K, `qwen2.5:3b`) | chain of thought lifts accuracy **12.5% → 85%**; self-consistency with 5 votes adds **nothing measurable** (85%) at 5× the tokens; best-of-5 with a 7B judge (80%) does **worse than the vote**; DeepSeek-R1-Distill 1.5B 63.7% with 16/80 answers cut off; a pairwise judge agrees with itself only 47% when the order is swapped; on Claude Haiku 5.5, `effort` low→max goes 92.5% → 98.8% for 4× the tokens, mostly a cost dial |
| 16 · tool calling | 79% tool selection and 86% valid arguments on 14 labelled requests; `function_calling` / `json_mode` 98% field accuracy vs 85% for `json_schema` on Ollama |
| 17 · embeddings & rerankers (SciFact) | fine-tuning MiniLM for **22 s** on 919 labelled pairs with BM25 hard negatives lifts nDCG@10 **0.648 → 0.762**, above bge-small (0.706) and nomic (0.708), with no loss on another domain; Matryoshka keeps 0.646 at 64 dims vs 0.589; binary codes + int8 rescoring match float32 (0.760 vs 0.762) in 32× less memory; general rerankers did not beat the fine-tuned bi-encoder |
| 18 · RAG + RAGAS | hybrid BM25 + dense retrieval: **evidence MRR 0.84** vs 0.73 dense-only. RAGAS with a 7B judge and bootstrap intervals: the one clear gain is **context precision 0.79 → 0.92**; the faithfulness "gain" a 3B judge reported (0.38 → 0.54) was the judge's noise |
| 19 · application patterns | map-reduce summary covers **11/13 topics** vs 9/13 for one silently truncated call (6/13 in another run); an LLM injection guard catches **10/10 attacks with 0/10 false positives** |
| 20 · DSPy (Banking77, `qwen2.5:3b`, 231 messages) | DSPy's zero-shot prompt scores 0.558 vs 0.489 for notebook 06's hand-written one; 16 random demos drop it to 0.433 and 16 nearest-neighbour demos lift it to 0.870; **MIPROv2, GEPA and bootstrap random search gain nothing** (0.554–0.558); retrieval alone (8-NN vote, no LLM) 0.913; a router that calls the LLM only when the neighbours disagree reaches **0.922 at 0.03 s per message, with 3% of the LLM calls** (the 7B with kNN demos: 0.926 at 1.6 s) |
| 21 · topic modeling (Banking77) | UMAP + HDBSCAN finds **73 clusters without being told k** (77 true intents), as good as k-means given k (ARI 0.683 vs 0.685); silhouette picks k = 60; an LLM names clusters; all **7 held-out intents surface as clusters** of flagged production messages |
| 22 · multimodal (Oxford-IIIT Pet) | CLIP zero-shot names 37 breeds at **86.3%** in 6 ms/image (a linear probe needs ~100 labels per breed to pass it); `qwen2.5vl:3b` gets 72% at 1.45 s and ~1,071 tokens per image; a wrong breed name written on the image makes the VLM answer it **72%** of the time vs 8% for CLIP |
| 23 · recommender (MovieLens-1M) | full-ranking hit@10: popularity 0.036, ItemKNN 0.125, two-tower 0.194, SASRec 0.293, **LambdaMART ranker 0.307** (+1.4 points, 95% CI [+0.7, +2.1]); a new-movie slot lifts cold-start hits 0.009 → 0.044; an LLM front end satisfies request constraints in **87% vs 60%** for embedding search, while LLM keywords did not help ranking |
| 24 · drift (MovieLens-1M stream, Banking77) | a frozen co-visitation recommender decays 0.184 → 0.160 hit@10 in 22 weeks while weekly retraining holds ~0.20 (+18%); PSI stayed at 0.01–0.03, so **input monitors missed the concept drift**; embedding drift tests detect a 40% shift in user questions 90–100% of the time but a 20% shift only 25–30% |
| 25 · LangGraph agents | every guardrail stops with an explicit reason; on 10 short tasks a single ReAct agent (70%, 2.2 model calls) beats plan-and-execute (60%, 6.6 calls), ties a supervisor (80% at 2.4× the calls) and beats free handoffs (30%); a code sandbox contains 8 of 9 attack probes (the `ctypes` hole shown on purpose) and no secret is reachable; a 7B code-writing agent answers 6/10 data questions with execution feedback |
| 26 · MCP | the FastMCP server passes **14/14 conformance checks**; the agent answers 6/6 questions through MCP tools (5/6 cited); a tool call costs 3.4 ms over stdio and 6.0 ms over authenticated Streamable HTTP; requests without a valid token get 401 with RFC 9728 metadata |
| 27 · agent harness (mini-swe-agent, 16 coding tasks × 3 runs, Claude Haiku 5.5) | scored by hidden tests: **48/48 solved, 0 false "done"**, 3.5 steps, 7.0 s and US$ 0.0010 per run; a modern harness profile solves the same 100% for **28% more tokens** (8,571 vs 6,701), and its verification gate first rejected correct work until it checked for tracebacks instead of exit codes; the model opened `AGENTS.md` by itself in 9 of 9 runs; a planted prompt injection was seen in 12/12 runs and followed in 0; 366 runs cost **US$ 0.42**. In v2.8, three local 3–8B models solved 20–30% of tasks 01–10, and 11 of 30 runs claimed a false "done" |
| 28 · LangSmith | with retrieval, judged correctness goes from **0.17 to 0.83**; online feedback surfaced a real tool-prompt bug |
| 29 · Inspect AI + Phoenix | Inspect reproduces notebook 15's GSM8K result exactly (**85.0%**); its built-in `match` scorer on the same generations gives 80.0% (−5 points, all four disagreements `\boxed{}` answers); three sampled epochs land at 80.0–82.5%; Phoenix traces show the LLM call is the whole latency (p50 3.6 s, every other step < 1 ms), and tracing costs **2.5 ms per request** (0.07% of an LLM call) |
| 30 · safety (JailbreakBench, XSTest) | `qwen2.5:3b` refuses 89% of harmful requests but over-refuses 17% of safe ones (7B: 5%); **prefix injection lifts attack success from 4% to 64%**; Llama Guard 3 1B catches 95–100% of harmful requests, disguised or not, but blocks 8–22% of safe ones; a 7B LLM judge agreed with humans only 70% vs 90% for a keyword judge |
| 31 · mini-R1 capstone (1.6M from-scratch model, mod-10 operation chains) | with only a verifier, GRPO from the base (R1-Zero) lifts accuracy **69.4% → 99.8%** (+30.4 points [+27.7, +33.1]) and the share of samples that reason from 11% to 67%; it learns to skip the steps on one-operation chains (0%) and reason on ~80% from four operations. The base, made to reason, already scored 99.9%: **RL elicited a skill pretraining put there**. A 300-example SFT cold start, R1 and a same-budget SFT control all reach 100.0%; with a weaker base (earlier run) R1 beat the control by +9.9 points. On 8–10 operations, longer than any in training, every model scores 14.8–22.6%, the base highest (no length generalization) |

## Quick start

```bash
git clone https://github.com/ZoodiacR/llm-edu-suite.git
cd llm-edu-suite

# 1) Install Ollama from https://ollama.com (the setup script pulls the two models for you)
# 2) One-time setup: virtual env + PyTorch (CUDA if you have an NVIDIA GPU) + all tools + Jupyter kernel + .env
powershell -ExecutionPolicy Bypass -File setup_env.ps1      # Windows
bash setup_env.sh                                           # Linux / macOS

jupyter lab notebooks/          # pick the kernel "Python (llm-edu-suite)" and start with 00_start_here/ (00, then 01)
```

Optional: put your LangSmith API key in `.env` (smith.langchain.com → Settings → API Keys) to see every LLM call of the course traced in LangSmith.

From the command line (inside the activated `.venv`):

```bash
pytest -q -m "not ollama"                 # fast tests with fake LLMs (~1 min)
pytest -q -m ollama                       # integration tests against the local models
python tools/run_notebooks.py 10 14       # execute notebooks headlessly, by number

python scripts/mini_llm.py                # the one-file mini GPT (notebook 01)
python scripts/finetune.py sft            # TRL fine-tuning CLI: sft | dpo | grpo
python scripts/rag_demo.py --ask "How do I enable LangSmith tracing?"
python scripts/mcp_server.py              # MCP server (stdio) for the sample docs
claude mcp add llm-edu-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py   # use it from Claude Code
```

Switching the whole course to a hosted model is a `.env` change (`LLM_PROVIDER=openai` or `anthropic`), because every notebook gets its models from `llm_edu_suite.config`.

## Repository layout

```
llm-edu-suite/
├── notebooks/                 32 notebooks, one folder per topic
│   ├── 00_start_here/  01_architecture/  02_pretraining_and_inference/
│   └── 03_post_training/  04_applications/  05_agents_and_mcp/  06_llmops/  07_capstone/
├── llm_edu_suite/             the from-scratch model (tokenizer, layers, model, training, inference, recsys,
│                              evaluation) + config.py, the one place where models and services are chosen
├── scripts/                   mini_llm, train, sample, finetune (TRL), rag_demo (LangChain), mcp_server (FastMCP),
│                              harness_run + harness_modern (mini-swe-agent, run from its own .venv-harness),
│                              inspect_tasks/gsm8k_cot (an Inspect AI task, run from its own .venv-eco)
├── tests/                     fast tests with fake LLMs + Ollama-marked integration tests
├── tools/notebooks/           one generator per notebook (the source of truth) + CONVENTIONS.md
├── data/                      knowledge base and eval sets (RAG, LangSmith, MCP, recommender), LLM item keywords;
│                              data/cache/ holds downloaded datasets (MovieLens, git-ignored)
├── docs/es/  docs/en/         full documentation in Spanish and English
└── AGENTS.md                  how the repo works, for coding agents (and humans in a hurry)
```

> **Notebooks are generated.** Each `.ipynb` is built from `tools/notebooks/nbXX_*.py`. Edit the generator, then `python tools/build_notebooks.py XX` and `python tools/run_notebooks.py XX`. See [tools/notebooks/CONVENTIONS.md](tools/notebooks/CONVENTIONS.md).

## From notebook to production

The notebooks are lessons: small data, one machine, measured results. The patterns carry over to real systems; the configuration, the data and the safety margins do not. Per topic, where it pays off at work and what to change first:

| topic (notebooks) | where you use it at work | what to change, harden or tune |
|---|---|---|
| **Foundations** (00–06) | reading model cards and papers, choosing a model (context length, GQA, MoE), debugging tokenization issues | nothing to deploy: it is the mental model. Reuse the tokenizer checks (e.g. count tokens per language before pricing a feature) |
| **Pretraining and inference** (07–12) | data pipelines (dedup, quality filters), sizing GPUs, serving a model, choosing a quantization | swap the toy model for an open checkpoint; serve with vLLM / SGLang / TGI instead of a loop; load-test with your own traffic shape (12) and set SLOs on p95 latency, not the mean |
| **Post-training** (13–15) | adapting a small model to a narrow task (extraction, classification, a house style), distilling a big model into a cheap one, adding reasoning where answers are checkable | real labelled data and a held-out set; LoRA first, full fine-tuning only if LoRA plateaus; a verifier (tests, a schema, a calculator) before any RL; e.g. distil an API model's answers into a 1–3B model for a high-volume endpoint |
| **Applications** (16–24) | RAG over internal docs, structured extraction, search and recommendations, clustering user feedback, prompt optimization | your corpus, chunking and an evaluation set built from real questions; a reranker before a bigger LLM; schema-validated outputs; drift monitors (24) on inputs and outputs; e.g. point 18 at your wiki instead of the course docs in `data/knowledge_base/`, write 20–50 real questions, and keep the same retrieval and RAGAS metrics |
| **Agents and MCP** (25–27) | internal assistants that act (tickets, queries), tools exposed over MCP, coding agents | least-privilege tools, human approval for anything that writes or leaves the machine, step/cost/time limits, a sandbox, and an `AGENTS.md`; evaluate with hidden tests and several runs per task (27) |
| **LLMOps** (28–30) | tracing every request, evaluations in CI, guardrails | send traces to a backend you control (Phoenix, Langfuse, LangSmith); pin the scorer and report confidence intervals (29); measure over-refusal as well as refusals (30); alert on latency, error rate and score drift |
| **Capstone** (31) | deciding whether RL with verifiable rewards is worth it for a task (math, code, structured output), and how to set it up | check pass@k on the base first (no successes, no signal); start with a small SFT cold start; compare RL against a same-budget SFT control; track the share of groups with no signal and filter problems by difficulty; evaluate by length and difficulty, beyond the training range; at scale use TRL's `GRPOTrainer`, verl or OpenRLHF |

**Cross-cutting, for any of them.**
- **Models.** The local defaults (`qwen2.5:3b`, SmolLM2) are chosen for a 6 GB GPU. In production pick by measured quality on *your* evaluation set, then by cost and latency. Switching provider is a `.env` change (`LLM_PROVIDER`, `LLM_MODEL`, `HARNESS_MODEL`).
- **Evaluation first.** Before changing a prompt, a model or a retriever, have a fixed evaluation set and a number to beat. Report the spread across runs, not one run.
- **Secrets and data.** Keys in a secret manager, not in `.env` on a server; prompts and traces treated as user data (retention, PII masking, access control).
- **Cost.** Log tokens per request from day one; cache what repeats; use the smallest model that passes the evaluation.

## How to use this repo

- **To study:** follow the numbers. Each notebook explains the idea, runs it, and ends with what was measured and how it is used in production. `docs/en/foundations.md` has the theory behind each topic.
- **At work:** copy the pattern, not the notebook. Start from the closest notebook, replace the data and the model, keep its evaluation, and harden it with the table above.
- **As practice material for others:** every notebook runs on a laptop GPU, and every result is reproducible from its generator. Change one thing (a model, a chunk size, a prompt), rerun, and compare the numbers.

## Working with a coding agent

Most engineering work today is done together with a coding agent (Claude Code, Codex CLI, opencode). This repo is set up for it: [`AGENTS.md`](AGENTS.md) tells an agent how the repo works (commands, how notebooks are generated, the rules: measured numbers only, bilingual docs, no secrets, one GPU job at a time). Prompts that work well:

- *"Read AGENTS.md and notebook 18. Adapt the RAG pipeline to the PDFs in `my_docs/`, keep the same evaluation, and tell me which metrics changed."*
- *"In notebook 13, replace the booking task with my extraction task in `data/my_task.jsonl`. Train LoRA r=16, evaluate on a held-out 20%, and write the Results table from the executed outputs only."*
- *"Run notebook 27 with `HARNESS_MODEL=openai/...` and compare its solve rate, tokens and cost with the Claude results in the notebook."*
- *"Review `scripts/harness_run.py` for ways a task could escape the sandbox, and propose deny rules with a test for each."*

Review what the agent changes, run the tests, and read the outputs before trusting a number.

## Design principles

- **Understand the inside, then use the real tool.** The model is built by hand because that is how attention, RoPE or a KV-cache stop being magic. Applications are built with the frameworks you will meet at work, using their current APIs.
- **Everything is evaluated.** Each notebook ends with the metric a practitioner would report: perplexity, exact match, hit@k / MRR, RAGAS scores, tool-selection accuracy, KL divergence, latency percentiles, judged correctness.
- **Local first.** Ollama + small Hugging Face models: anyone can reproduce the course on a laptop, offline, for free.
- **Honest.** Negative results stay in (plain DPO that degrades the model, a small judge that invents effects, LLM keywords that do not help a recommender), each with the explanation of why. A component only enters a system if it beats the previous one on held-out data.

## Versions

| Version | What it is |
|---|---|
| [v1.0](https://github.com/ZoodiacR/llm-edu-suite/tree/v1.0-from-scratch) | Everything implemented from scratch, including RAG, agents and observability, to learn how each piece works. |
| [v2.0](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.0) | The model internals stay from scratch; the application layer moves to the tools used in industry (Hugging Face, LangChain, LangGraph, LangSmith, RAGAS, MCP SDK) with local models through Ollama. |
| [v2.1](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.1) | Theory sections in every notebook and a bilingual foundations guide; the recommender rebuilt on MovieLens-1M as an industry funnel; QLoRA; a stronger LLM judge with confidence intervals. |
| [v2.2](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.2) | From scratch to real models: SmolLM2's published weights run in the from-scratch transformer with identical output; a new notebook on encoders and encoder-decoders (masked LM from scratch, ModernBERT vs embeddings vs LLM for classification, Flan-T5); every new notebook closes with *Results* and *In production*. |
| [v2.3](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.3) | Training at scale: a data-engineering notebook (MinHash/LSH deduplication from scratch, quality filters, decontamination, raw vs clean training) and a systems notebook (memory anatomy, mixed precision, checkpointing, fused attention, profiling, CUDA graphs, MFU, real DDP, ZeRO/FSDP). |
| [v2.4](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.4) | Reasoning and adaptation: a notebook on test-time compute (chain of thought, self-consistency, best-of-N with an LLM verifier, a distilled reasoning model, Bradley-Terry rankings and judge position bias) and a model-merging section (task arithmetic, TIES, DARE vs multi-task training); notebook 13 closes with *Results* and *In production*. |
| [v2.5](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.5) | Retrieval and agents in depth: an embeddings-and-rerankers notebook (contrastive fine-tuning with hard negatives, Matryoshka, binary quantization with rescoring, rerankers, on BEIR SciFact); multi-agent supervisor vs handoffs and sandboxed code execution in notebook 25; MCP sampling, elicitation, roots and a remote server with OAuth 2.1 bearer tokens in notebook 26. |
| [v2.6](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.6) | Production: a safety notebook (refusals vs over-refusal, published jailbreaks, Llama Guard 3 as an input filter, layered defences, validated judges) and a serving notebook (latency metrics, prefill vs decode, prefix caching, batching slots, GGUF formats, a gateway and a cost-aware cascade). |
| [v2.7](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.7) | Beyond text and beyond a single snapshot: multimodal models (CLIP zero-shot and probes, a vision-language model, document extraction, typographic attacks), clustering and topic modeling of user messages with emerging-intent detection, and online learning and drift (models aging on a stream, PSI, embedding drift tests). |
| [v2.8](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.8) | Agent harness notebook: mini-swe-agent anatomy, 10 coding tasks × 3 local models, `AGENTS.md` memory, repo prompt injection. |
| [v2.9](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.9) | Ecosystem and foundations: transformers big picture (00), DSPy prompt optimization (20), Inspect AI + Phoenix (29); *Results* / *In production* closings on notebooks 01–05, 08, 10, 11, 13, 14, 16, 18, 19, 23, 25 and 28. |
| [v2.10](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.10) | Agent harness on Claude Haiku 5.5: 16 tasks × 3 runs, workspace context, process-tree kill, verify-gate fix, preserved-thinking note; 366 runs for US$ 0.42. |
| v2.11 | Notebooks renumbered in study order (00–30); a closing *From notebook to production* guide per topic; `AGENTS.md` and prompts for working on the repo with a coding agent. |
| v2.12 | mini-R1 capstone (notebook 31): pretraining, R1-Zero, cold start, R1 and a same-budget SFT control on the from-scratch model; Claude `effort` section in notebook 15, and notebook 15 re-run. |

## Roadmap

Everything the previous roadmap listed has shipped: reasoning at inference time (v2.4, notebook 15), fine-tuned embeddings and rerankers (v2.5, notebook 17) and a serving benchmark under load (v2.6, notebook 12). Since then the course has added safety, multimodal models, clustering, drift, the agent harness and the ecosystem notebooks (DSPy, Inspect AI and Phoenix, the transformer big picture); the [Versions](#versions) table has the details.

Both items it listed next have shipped in v2.12:

- **A reasoning model end to end, from scratch** (notebook 31, done). The mini-R1 capstone runs the whole recipe on the course's from-scratch model: pretraining on a verifiable task, R1-Zero, an SFT cold start, R1 and a same-budget SFT control, with GRPO written by hand, measuring accuracy, reasoning share, answer length and accuracy by chain length.
- **Reasoning effort on an API model** (notebook 15, section 7, done). The same GSM8K problems at several `effort` levels of Claude Haiku 5.5: 92.5% → 98.8% from low to max for 4× the tokens, mostly a cost dial.

## Documentation

| | Spanish | English |
|---|---|---|
| Index | [docs/es/README.md](docs/es/README.md) | [docs/en/README.md](docs/en/README.md) |
| Installation & environment (Ollama, `.env`, LangSmith, GPU) | [instalacion.md](docs/es/instalacion.md) | [installation.md](docs/en/installation.md) |
| Notebook guide | [guia_notebooks.md](docs/es/guia_notebooks.md) | [notebooks_guide.md](docs/en/notebooks_guide.md) |
| Theory foundations: why each technique exists, how it works, what it replaced | [fundamentos.md](docs/es/fundamentos.md) | [foundations.md](docs/en/foundations.md) |
| Module reference (`llm_edu_suite`) and where each topic is taught | [referencia_modulos.md](docs/es/referencia_modulos.md) | [module_reference.md](docs/en/module_reference.md) |
| Command-line scripts | [scripts_cli.md](docs/es/scripts_cli.md) | [cli_scripts.md](docs/en/cli_scripts.md) |
| Case study (portfolio write-up) | [caso_de_estudio.md](docs/es/caso_de_estudio.md) | [case_study.md](docs/en/case_study.md) |
| Glossary | [glosario.md](docs/es/glosario.md) | [glossary.md](docs/en/glossary.md) |

## References

Attention Is All You Need · RoFormer (RoPE) · GQA · DeepSeek-V2/V3 (MLA, DeepSeekMoE) · Switch Transformer · FlashAttention · Longformer · In-context Learning and Induction Heads (Olsson 2022) · Chinchilla · Llama 3 · LoRA · QLoRA · DPO · Iterative RPO · DeepSeekMath (GRPO) · DeepSeek-R1 · Speculative Decoding · RAG (Lewis 2020) · RAGAS · ReAct · Building Effective Agents (Anthropic 2024) · Model Context Protocol specification · SASRec · and the documentation of Hugging Face TRL/PEFT, LangChain, LangGraph, LangSmith and the MCP Python SDK.

## License

The code, notebooks and documentation are released under the [MIT License](LICENSE): you may use, copy, modify and distribute them, including for commercial purposes, as long as the copyright notice is kept.

Third-party data and models keep their own terms. MovieLens is downloaded at run time from GroupLens and is not redistributed here; models are downloaded from Hugging Face and Ollama under their respective licenses.
