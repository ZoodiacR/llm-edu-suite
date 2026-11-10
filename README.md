# LLM Edu Suite

**Learn LLM engineering in two halves: build a language model from scratch in PyTorch to understand how it works, then build real applications with the tools the industry uses — Hugging Face, LangChain, LangGraph, LangSmith, RAGAS and the Model Context Protocol.** Every notebook explains the idea, runs the code, and ends with an evaluation using the metrics practitioners actually report.

🇪🇸 **[Documentación completa en español](docs/es/README.md)** · 🇬🇧 **[Full documentation in English](docs/en/README.md)**

![python](https://img.shields.io/badge/python-3.11%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c) ![hf](https://img.shields.io/badge/HF-transformers%20·%20peft%20·%20trl-yellow) ![langchain](https://img.shields.io/badge/LangChain-LangGraph%20·%20LangSmith-1c3c3c) ![mcp](https://img.shields.io/badge/MCP-official%20SDK-6b4fbb) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

- **22 notebooks** grouped by topic, all executed with their outputs, plots and an evaluation section
- **Runs on your own laptop**: local models through [Ollama](https://ollama.com) (`qwen2.5:3b`, `nomic-embed-text`) and Hugging Face SmolLM2; an NVIDIA GPU makes it fast, CPU also works. No API keys required (LangSmith is optional and free).
- **Honest results**: when a technique does not help in our setup, the notebook shows it and explains why
- **Tested**: fast tests with fake LLMs in CI on every push, integration tests against the local models

> **v1 vs v2.** The first version ([`v1.0-from-scratch`](https://github.com/ZoodiacR/llm-edu-suite/tree/v1.0-from-scratch)) implemented *everything* by hand, including RAG, agents, tracing and MCP. v2 keeps the from-scratch part where seeing the internals is the point (the model itself), and teaches the application layer with the frameworks you will use at work.

---

## Topic map

The number is the suggested study order; the folder is the topic.

| Topic | Notebook | What you learn | Tools |
|---|---|---|---|
| **Start here** | [00 · mini LLM end to end](notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) | a whole GPT in one notebook: BPE → RoPE/RMSNorm/SwiGLU → training → sampling → checkpoint | PyTorch |
| **Architecture** | [01 · tokenization](notebooks/01_architecture/01_tokenization.ipynb) | byte-level BPE, vocabulary trade-offs, fertility | PyTorch (from scratch) |
| | [02 · attention, RoPE, norms](notebooks/01_architecture/02_attention_rope_norm.ipynb) | RMSNorm, RoPE, MHA → GQA → MQA → MLA, KV-cache | PyTorch (from scratch) |
| | [03 · efficient attention](notebooks/01_architecture/03_efficient_attention.ipynb) | sliding window, Longformer, block-sparse, linear attention, induction heads | PyTorch, FlexAttention |
| | [04 · FFN & MoE](notebooks/01_architecture/04_ffn_moe.ipynb) | SwiGLU, Mixture of Experts, aux-loss vs aux-free balancing | PyTorch (from scratch) |
| | [17 · encoders & encoder-decoders](notebooks/01_architecture/17_encoders_and_encoder_decoders.ipynb) | bidirectional attention and masked LM from scratch; classifying 77 intents with a fine-tuned ModernBERT, embeddings + logistic regression, or an LLM; Flan-T5 cross-attention | PyTorch, `transformers`, scikit-learn, LangChain |
| **Pre-training & inference** | [18 · data engineering](notebooks/02_pretraining_and_inference/18_data_engineering.ipynb) | exact and MinHash/LSH deduplication from scratch, Gopher/C4 rules, the FineWeb-Edu quality classifier, 13-gram decontamination, raw vs clean training, synthetic data | FineWeb, `transformers`, PyTorch |
| | [05 · pre-training](notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | the training pipeline, LR schedules, irreducible loss, scaling curves | PyTorch (from scratch) |
| | [19 · systems](notebooks/02_pretraining_and_inference/19_systems_memory_and_parallelism.ipynb) | memory anatomy, bf16, activation checkpointing, fused attention kernels, profiling, CUDA graphs, MFU, real DDP, ZeRO/FSDP | PyTorch, `torch.distributed` |
| | [06 · inference & serving](notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | sampling, KV-cache, speculative decoding, batching, TTFT; loading SmolLM2's real weights into our model | PyTorch (from scratch), `transformers` |
| | [09 · quantization](notebooks/02_pretraining_and_inference/09_quantization.ipynb) | int8 / float8 / int4 / int2 on a real model, measured with KL and top-1 agreement | `optimum-quanto` |
| **Post-training** | [07 · SFT, LoRA, distillation](notebooks/03_post_training/07_finetuning_sft_lora.ipynb) | chat templates and loss masking, full SFT vs LoRA vs QLoRA (4-bit NF4), the new-token trap, rank ablation, teacher → student, model merging (task arithmetic, TIES, DARE) | `transformers`, `peft`, `trl`, `bitsandbytes` |
| | [08 · DPO & GRPO](notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | preference tuning, likelihood displacement, DPO + NLL, RL with verifiable rewards | `trl` |
| | [20 · reasoning & test-time compute](notebooks/03_post_training/20_reasoning_test_time_compute.ipynb) | chain of thought, self-consistency (maj@k / pass@k), best-of-N with an LLM verifier, a distilled reasoning model, Bradley-Terry rankings, judge position bias | LangChain, Ollama |
| **Applications** | [21 · embeddings & rerankers](notebooks/04_applications/21_embeddings_and_rerankers.ipynb) | bi-encoders vs cross-encoders, InfoNCE by hand, contrastive fine-tuning with hard negatives, Matryoshka, int8 / binary quantization with rescoring, hybrid retrieval and rerankers, on BEIR SciFact | `sentence-transformers`, Ollama |
| | [10 · RAG](notebooks/04_applications/10_rag.ipynb) | loaders, splitters, FAISS, BM25, hybrid RRF, LCEL chains with citations, RAGAS | LangChain, RAGAS |
| | [12 · recommender systems](notebooks/04_applications/12_recommender_systems.ipynb) | the industry funnel on MovieLens-1M: SASRec, two-tower + FAISS, LambdaMART ranker, exploration and diversity, LLM requests and explanations | PyTorch, FAISS, LightGBM, LangChain |
| | [13 · LLM application patterns](notebooks/04_applications/13_modern_llm_applications.ipynb) | extraction, classification, map-reduce summarization, caching, prompt-injection defense, batch jobs | LangChain |
| **Agents & MCP** | [11 · tools & structured output](notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | how tool calling works, `bind_tools`, the loop by hand, `create_agent`, Pydantic outputs | LangChain |
| | [14 · agentic AI](notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | StateGraph, guardrails, human-in-the-loop, memory, plan-and-execute, reflection, routing, supervisor vs handoffs, sandboxed code execution | LangGraph |
| | [16 · Model Context Protocol](notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | MCP from the wire up, a FastMCP server, an agent using MCP tools, security, sampling / elicitation / roots, a remote server with OAuth 2.1 bearer tokens | `mcp` SDK, `langchain-mcp-adapters` |
| **LLMOps** | [15 · observability & evaluation](notebooks/06_llmops/15_observability_llmops.ipynb) | tracing, latency percentiles, SLOs, feedback, PII anonymization, datasets + `evaluate()` | LangSmith |

Detailed notebook-by-notebook guide: [English](docs/en/notebooks_guide.md) · [Español](docs/es/guia_notebooks.md).

## Selected results

Measured in the executed notebooks on a laptop with an RTX 3060 (6 GB). Models are small on purpose (a 3B chat model, 135M–360M models for training), so the absolute numbers matter less than the comparisons. LLM-judged numbers vary slightly from run to run.

| Notebook | Result |
|---|---|
| 00 · mini LLM | a 1.9M-parameter GPT trains in ~20 s on the GPU and ends **0.6% above its corpus's theoretical minimum loss**; 82% of sampled sentences are grammatical |
| 03 · efficient attention | long-range copy accuracy: **dense 99.9%, block-sparse 88.7%, hybrid 48.4%, sliding window and linear ~0%**; at 300 steps even dense is at 0.3% (the induction-head phase transition) |
| 17 · encoders | from scratch, bidirectional context recovers the first token of a word **100% vs 2.9%** for a decoder. On Banking77 (77 intents): fine-tuned ModernBERT **93.0%**, nomic embeddings + logistic regression **93.3%**, `qwen2.5:3b` 61.9% few-shot; function calling left 181 of 231 outputs invalid, constrained decoding none |
| 18 · data engineering | MinHash + LSH from scratch follows the theoretical S-curve (100% of templated copies and 1% rewrites caught, 0 false merges); rules remove 12% of FineWeb; decontamination finds 50/50 planted leaks plus 29 natural overlaps; a model trained on the raw crawl memorizes 10×-repeated pages (loss 4.35 vs 5.28 for the deduplicated model) |
| 19 · systems | memory matches the 16-bytes-per-parameter rule to the MB; fused attention needs **0.22 GB at 16k tokens** where naive attention runs out of memory; CUDA graphs **4.1×** faster; MFU 32%; two-process DDP reproduces large-batch training to 7e-9 |
| 06 · inference | speculative decoding accepts **98% of draft tokens, ~4.6 tokens per target forward pass**. SmolLM2-135M's published weights run in our from-scratch `TransformerLM` with **identical greedy output** to Hugging Face (logits within 3e-5) |
| 07 · SFT & LoRA (SmolLM2-135M) | few-shot 17.5% → **full SFT 35%**, LoRA r=16 32.5% training 3.5% of the parameters, r=64 45%. A new special token with frozen embeddings is **never emitted (0%)**; with `modules_to_save` it is emitted **100%** of the time |
| 07 · QLoRA | the same LoRA on a 4-bit NF4 base: **no loss of quality** (37.5% vs 32.5% exact, same 86.5% of fields), weights 269 → 110 MB; for a 7B model the frozen weights drop from 14 GB to 3.6 GB |
| 07 · distillation | a 360M teacher (87.5%) labels 600 extra prompts; the 135M student rises from 35% to **55%** |
| 07 · model merging | two LoRA adapters with nearly orthogonal task vectors (cosine 0.01): task arithmetic keeps the booking task at 32.5% and converts times at **90%** (specialist 97.5%) with no training; `linear`, TIES and DARE do worse; multi-task training beats every merge (55% + 97.5%) |
| 08 · DPO & GRPO | plain DPO wins its training metric but **drops exact match from 30% to 7.5%** (likelihood displacement); DPO + NLL holds 32.5%. **GRPO lifts arithmetic from 5% to 46% in 40 steps** with no labels, only a verifier |
| 20 · reasoning (GSM8K, `qwen2.5:3b`) | chain of thought lifts accuracy **12.5% → 85%**; self-consistency with 5 votes **90%** at 5× the tokens; best-of-5 with a 7B judge (85%) does **worse than the vote**; DeepSeek-R1-Distill 1.5B 63.7% with 16/80 answers cut off; a pairwise judge agrees with itself only 43% when the order is swapped |
| 09 · quantization (SmolLM2-360M) | int8 and float8 keep ~96% top-1 agreement at 0.41 GB; int4 drops to 73% (KL 0.27); int2 breaks the model. Perplexity hides what KL shows |
| 21 · embeddings & rerankers (SciFact) | fine-tuning MiniLM for **22 s** on 919 labelled pairs with BM25 hard negatives lifts nDCG@10 **0.648 → 0.762**, above bge-small (0.706) and nomic (0.708), with no loss on another domain; Matryoshka keeps 0.646 at 64 dims vs 0.589; binary codes + int8 rescoring match float32 (0.760 vs 0.762) in 32× less memory; general rerankers did not beat the fine-tuned bi-encoder |
| 10 · RAG + RAGAS | hybrid BM25 + dense retrieval: **evidence MRR 0.84** vs 0.73 dense-only. RAGAS with a 7B judge and bootstrap intervals: the one clear gain is **context precision 0.79 → 0.92**; the faithfulness "gain" a 3B judge reported (0.38 → 0.54) was the judge's noise |
| 11 · tool calling | 79% tool selection and 86% valid arguments on 14 labelled requests; `function_calling` / `json_mode` 98% field accuracy vs 85% for `json_schema` on Ollama |
| 12 · recommender (MovieLens-1M) | full-ranking hit@10: popularity 0.036, ItemKNN 0.125, two-tower 0.194, SASRec 0.293, **LambdaMART ranker 0.307** (+1.4 points, 95% CI [+0.7, +2.1]); a new-movie slot lifts cold-start hits 0.009 → 0.044; an LLM front end satisfies request constraints in **87% vs 60%** for embedding search, while LLM keywords did not help ranking |
| 13 · application patterns | map-reduce summary covers **11/13 topics** vs 9/13 for one silently truncated call (6/13 in another run); an LLM injection guard catches **10/10 attacks with 0/10 false positives** |
| 14 · LangGraph agents | every guardrail stops with an explicit reason; on 10 short tasks a single ReAct agent (70%, 2.2 model calls) beats plan-and-execute (60%, 6.6 calls), ties a supervisor (80% at 2.4× the calls) and beats free handoffs (30%); a code sandbox contains 8 of 9 attack probes (the `ctypes` hole shown on purpose) and no secret is reachable; a 7B code-writing agent answers 6/10 data questions with execution feedback |
| 15 · LangSmith | with retrieval, judged correctness goes from **0.17 to 0.83**; online feedback surfaced a real tool-prompt bug |
| 16 · MCP | the FastMCP server passes **14/14 conformance checks**; the agent answers 6/6 questions through MCP tools (4/6 cited); a tool call costs 3.0 ms over stdio and 5.2 ms over authenticated Streamable HTTP; requests without a valid token get 401 with RFC 9728 metadata |

## Quick start

```bash
git clone https://github.com/ZoodiacR/llm-edu-suite.git
cd llm-edu-suite

# 1) Install Ollama from https://ollama.com (the setup script pulls the two models for you)
# 2) One-time setup: virtual env + PyTorch (CUDA if you have an NVIDIA GPU) + all tools + Jupyter kernel + .env
powershell -ExecutionPolicy Bypass -File setup_env.ps1      # Windows
bash setup_env.sh                                           # Linux / macOS

jupyter lab notebooks/          # pick the kernel "Python (llm-edu-suite)" and start with 00_start_here/
```

Optional: put your LangSmith API key in `.env` (smith.langchain.com → Settings → API Keys) to see every LLM call of the course traced in LangSmith.

From the command line (inside the activated `.venv`):

```bash
pytest -q -m "not ollama"                 # fast tests with fake LLMs (~1 min)
pytest -q -m ollama                       # integration tests against the local models
python tools/run_notebooks.py 10 14       # execute notebooks headlessly, by number

python scripts/mini_llm.py                # the one-file mini GPT (notebook 00)
python scripts/finetune.py sft            # TRL fine-tuning CLI: sft | dpo | grpo
python scripts/rag_demo.py --ask "How do I enable LangSmith tracing?"
python scripts/mcp_server.py              # MCP server (stdio) for the sample docs
claude mcp add llm-edu-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py   # use it from Claude Code
```

Switching the whole course to a hosted model is a `.env` change (`LLM_PROVIDER=openai` or `anthropic`), because every notebook gets its models from `llm_edu_suite.config`.

## Repository layout

```
llm-edu-suite/
├── notebooks/                 22 notebooks, one folder per topic
│   ├── 00_start_here/  01_architecture/  02_pretraining_and_inference/
│   └── 03_post_training/  04_applications/  05_agents_and_mcp/  06_llmops/
├── llm_edu_suite/             the from-scratch model (tokenizer, layers, model, training, inference, recsys,
│                              evaluation) + config.py, the one place where models and services are chosen
├── scripts/                   mini_llm, train, sample, finetune (TRL), rag_demo (LangChain), mcp_server (FastMCP)
├── tests/                     fast tests with fake LLMs + Ollama-marked integration tests
├── tools/notebooks/           one generator per notebook (the source of truth) + CONVENTIONS.md
├── data/                      knowledge base and eval sets (RAG, LangSmith, MCP, recommender), LLM item keywords;
│                              data/cache/ holds downloaded datasets (MovieLens, git-ignored)
└── docs/es/  docs/en/         full documentation in Spanish and English
```

> **Notebooks are generated.** Each `.ipynb` is built from `tools/notebooks/nbXX_*.py`. Edit the generator, then `python tools/build_notebooks.py XX` and `python tools/run_notebooks.py XX`. See [tools/notebooks/CONVENTIONS.md](tools/notebooks/CONVENTIONS.md).

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
| [v2.4](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.4) | Reasoning and adaptation: a notebook on test-time compute (chain of thought, self-consistency, best-of-N with an LLM verifier, a distilled reasoning model, Bradley-Terry rankings and judge position bias) and a model-merging section (task arithmetic, TIES, DARE vs multi-task training); notebook 07 closes with *Results* and *In production*. |
| [v2.5](https://github.com/ZoodiacR/llm-edu-suite/tree/v2.5) | Retrieval and agents in depth: an embeddings-and-rerankers notebook (contrastive fine-tuning with hard negatives, Matryoshka, binary quantization with rescoring, rerankers, on BEIR SciFact); multi-agent supervisor vs handoffs and sandboxed code execution in notebook 14; MCP sampling, elicitation, roots and a remote server with OAuth 2.1 bearer tokens in notebook 16. |

## Roadmap

The next additions extend the hands-on side of the course:

- **Reasoning at inference time:** chain of thought, self-consistency and best-of-N with a verifier, measured as a cost-accuracy curve.
- **Better retrieval:** fine-tuned embedding models and cross-encoder rerankers for the RAG notebook.
- **Serving:** a benchmark of time to first token, time per output token and throughput under concurrent load.

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
