# LLM Edu Suite

**Build every layer of a modern LLM system from scratch, then measure it.** From a byte-level tokenizer to a transformer, pre-training, fine-tuning (SFT, LoRA, DPO, GRPO), quantization, RAG, agents, observability and the Model Context Protocol, each piece is implemented in readable, commented PyTorch/NumPy, exercised in a notebook, and **evaluated with the metrics practitioners actually use**.

🇪🇸 **[Documentación completa en español](docs/es/README.md)** · 🇬🇧 **[Full documentation in English](docs/en/README.md)**

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c) ![tests](https://img.shields.io/badge/tests-69%20passing-brightgreen) ![notebooks](https://img.shields.io/badge/notebooks-17-orange) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

- **17 notebooks**, grouped by topic, each ending with an evaluation section and plots
- **An 18-module Python package** (`llmsuite`) you can import, read and reuse
- **6 command-line scripts**, including a one-file mini GPT and a real MCP server
- **Runs offline on a laptop**, on CPU or (much faster) an NVIDIA GPU, with no downloads and no API keys. A deterministic `MockLLM` stands in for an API model, and you can swap in a real one with one line.
- **Verified end to end**: 69 tests, every notebook executed with its outputs saved, and CI on every push

---

## Topic map

Use this table to find things. The number is the suggested study order; the folder is the topic.

| Topic | Notebook | You build / learn | Package module | Docs |
|---|---|---|---|---|
| **Start here** | [`00_mini_llm_end_to_end`](notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) | a complete GPT in one notebook, using only torch: BPE → RoPE/RMSNorm/SwiGLU → training → sampling → checkpoint | (self-contained) · `scripts/mini_llm.py` | [ES](docs/es/guia_notebooks.md#00) · [EN](docs/en/notebooks_guide.md#00) |
| **Architecture** | [`01_tokenization`](notebooks/01_architecture/01_tokenization.ipynb) | byte-level BPE, vocabulary trade-offs, fertility | `tokenizer` | [ES](docs/es/guia_notebooks.md#01) · [EN](docs/en/notebooks_guide.md#01) |
| | [`02_attention_rope_norm`](notebooks/01_architecture/02_attention_rope_norm.ipynb) | RMSNorm, RoPE, attention by hand, MHA → GQA → MQA → MLA, KV-cache | `layers`, `model` | [ES](docs/es/guia_notebooks.md#02) · [EN](docs/en/notebooks_guide.md#02) |
| | [`03_efficient_attention`](notebooks/01_architecture/03_efficient_attention.ipynb) | sliding window, Longformer, block-sparse, linear attention, hybrids | `layers` | [ES](docs/es/guia_notebooks.md#03) · [EN](docs/en/notebooks_guide.md#03) |
| | [`04_ffn_moe`](notebooks/01_architecture/04_ffn_moe.ipynb) | SwiGLU, Mixture of Experts, aux-loss vs aux-free balancing | `layers` | [ES](docs/es/guia_notebooks.md#04) · [EN](docs/en/notebooks_guide.md#04) |
| **Pre-training & inference** | [`05_pretraining`](notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | full pipeline, LR schedules, irreducible loss, scaling curves | `training`, `data` | [ES](docs/es/guia_notebooks.md#05) · [EN](docs/en/notebooks_guide.md#05) |
| | [`06_inference_serving`](notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | sampling strategies, KV-cache, **speculative decoding**, batching, TTFT | `inference`, `model` | [ES](docs/es/guia_notebooks.md#06) · [EN](docs/en/notebooks_guide.md#06) |
| | [`09_quantization`](notebooks/02_pretraining_and_inference/09_quantization.ipynb) | absmax INT8/INT4, block-wise scales, NF4, KL vs memory | `quant` | [ES](docs/es/guia_notebooks.md#09) · [EN](docs/en/notebooks_guide.md#09) |
| **Post-training** | [`07_finetuning_sft_lora`](notebooks/03_post_training/07_finetuning_sft_lora.ipynb) | SFT with loss masking, LoRA (and why it can fail), rank ablation, distillation | `finetune` | [ES](docs/es/guia_notebooks.md#07) · [EN](docs/en/notebooks_guide.md#07) |
| | [`08_preference_rl_dpo_grpo`](notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | DPO, likelihood displacement and DPO+NLL, GRPO with verifiable rewards | `finetune` | [ES](docs/es/guia_notebooks.md#08) · [EN](docs/en/notebooks_guide.md#08) |
| **Applications** | [`10_rag`](notebooks/04_applications/10_rag.ipynb) | chunking, dense + BM25 + RRF, citations, retrieval and answer metrics | `rag`, `evaluation` | [ES](docs/es/guia_notebooks.md#10) · [EN](docs/en/notebooks_guide.md#10) |
| | [`12_recommender_systems`](notebooks/04_applications/12_recommender_systems.ipynb) | SASRec-style sequential recommendation, embeddings, LLM re-ranking | `recsys` | [ES](docs/es/guia_notebooks.md#12) · [EN](docs/en/notebooks_guide.md#12) |
| | [`13_modern_llm_applications`](notebooks/04_applications/13_modern_llm_applications.ipynb) | extraction, classification, summarization, semantic cache, injection guard, batch jobs | `applications` | [ES](docs/es/guia_notebooks.md#13) · [EN](docs/en/notebooks_guide.md#13) |
| **Agents & MCP** | [`11_agents_tools_structured_output`](notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | tool schemas, validation, the agent loop, structured output | `agents` | [ES](docs/es/guia_notebooks.md#11) · [EN](docs/en/notebooks_guide.md#11) |
| | [`14_agentic_ai`](notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | budgets and loop detection, plan-and-execute, reflection, memory, multi-agent routing | `agentic` | [ES](docs/es/guia_notebooks.md#14) · [EN](docs/en/notebooks_guide.md#14) |
| | [`16_mcp_model_context_protocol`](notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | MCP from the wire up: JSON-RPC, server, client, stdio, security | `mcp` · `scripts/mcp_server.py` | [ES](docs/es/guia_notebooks.md#16) · [EN](docs/en/notebooks_guide.md#16) |
| **LLMOps** | [`15_observability_llmops`](notebooks/06_llmops/15_observability_llmops.ipynb) | traces and spans, cost, p95/p99, SLOs, online eval scores, PII redaction | `observability` | [ES](docs/es/guia_notebooks.md#15) · [EN](docs/en/notebooks_guide.md#15) |

## Selected results

Every number below comes from the executed notebooks in this repository, run on a laptop with an **RTX 3060 (6 GB)**. The models are toy-sized on purpose, so the absolute values matter less than the method and the comparisons. Everything also runs on CPU; numbers then differ slightly (small models are sensitive to numerics), but the conclusions hold, and the notebooks point out where a result is fragile.

| Notebook | Result |
|---|---|
| 00 · mini LLM | a 1.9M-parameter GPT trains in **~22 s on the GPU** and ends **0.6% above the corpus's theoretical minimum loss**; 82% of its sampled sentences are grammatical |
| 02 · attention | **MLA needs 160 B/token of KV-cache versus 2,048 B for MHA** (12.8× less) at the same validation perplexity |
| 03 · efficient attention | on a long-range copy task, **dense attention reaches 99.9% copy accuracy, block-sparse 88.7%, a 1-in-4 hybrid 48.4%, sliding window and linear attention ~0%**; after only 300 steps even dense is at 0.3%, which is the induction-head phase transition |
| 05 · pre-training | the model reaches its corpus's **irreducible loss** (computed analytically); scaling and schedule ablations become learning curves against that floor |
| 06 · inference | **speculative decoding accepts 98% of draft tokens, about 4.6 tokens per target forward pass**; batching 16 sequences gives ~17× the throughput of one |
| 07 · fine-tuning | full SFT reaches **100% exact match**. Plain LoRA scores 0% (the new chat tokens have frozen embeddings); **LoRA + trainable embeddings reaches 95.3%** while training 16.5% of the parameters |
| 07 · distillation | a 6× smaller student goes from **24.7% (1k human labels) to 98.0%** when the teacher labels 6,000 extra prompts |
| 08 · DPO / GRPO | plain DPO raises preference accuracy but **drops exact match from 70.7% to 46.7%** (likelihood displacement); **DPO + NLL lifts it to 80.0%** |
| 09 · quantization | **NF4 keeps 98.6% top-1 agreement at 0.31× memory**; at 2 bits agreement drops to 88%, which the KL metric shows long before perplexity does |
| 10 · RAG | hybrid BM25 + dense retrieval with RRF: **recall@1 1.00** (dense alone 0.88) |
| 12 · recommendation | SASRec **hit@10 0.210**, against 0.097 for popularity and 0.032 for random |
| 14 · agentic AI | plan-and-execute solves **6/6 multi-step tasks against 2/6** for a single loop (with the offline MockLLM); every agent run ends with an explicit stop reason |
| 16 · MCP | the custom server passes **7/7 protocol-conformance checks** over a real stdio subprocess; its search tool finds the right document for 6/6 labelled questions |

## Quick start

```bash
git clone https://github.com/ZoodiacR/llm-edu-suite.git
cd llm-edu-suite

# one-time setup: virtual env + CPU PyTorch + Jupyter kernel "Python (llmsuite)" + test run
powershell -ExecutionPolicy Bypass -File setup_env.ps1      # Windows (installs CUDA PyTorch if it finds an NVIDIA GPU)
bash setup_env.sh                                           # Linux / macOS

jupyter lab notebooks/                                      # start with 00_start_here/
```

Everything from the command line (inside the activated `.venv`):

```bash
pytest -q                                                        # 69 tests, ~10 s
python tools/run_notebooks.py                                    # execute every notebook (~15 min GPU, ~40 min CPU)
python tools/run_notebooks.py 07 14                              # ... or just some, by number

python scripts/mini_llm.py                                       # the one-file mini GPT (notebook 00)
python scripts/train.py --steps 300 --out runs/smoke             # pre-train a configurable TransformerLM
python scripts/sample.py --run runs/smoke --max-new-tokens 50    # generate from it
python scripts/finetune.py                                       # SFT -> DPO+NLL / GRPO on a toy task
python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json
python scripts/mcp_server.py                                     # MCP server (stdio) for the sample docs
```

Use a real model anywhere the `MockLLM` is used, through any OpenAI-compatible server (vLLM, Ollama, OpenAI, OpenRouter…):

```python
from llmsuite.llm_client import OpenAICompatibleClient
llm = OpenAICompatibleClient("qwen2.5:7b", base_url="http://localhost:11434/v1")   # e.g. Ollama
```

Connect the MCP server to Claude Code: `claude mcp add llmsuite-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py`.

## Repository layout

```
llm-edu-suite/
├── notebooks/                      17 notebooks, one folder per topic (see the topic map)
│   ├── 00_start_here/  01_architecture/  02_pretraining_and_inference/
│   └── 03_post_training/  04_applications/  05_agents_and_mcp/  06_llmops/
├── llmsuite/                       the package: every module is documented and commented
│   ├── tokenizer.py  layers.py  model.py  data.py  training.py        core model & training
│   ├── finetune.py  quant.py  inference.py                            post-training & efficiency
│   ├── rag.py  recsys.py  applications.py  evaluation.py              applications & metrics
│   ├── agents.py  agentic.py  mcp.py  observability.py  llm_client.py agents, protocol, ops
│   └── utils.py
├── scripts/                        CLI: mini_llm, train, sample, finetune, rag_demo, mcp_server
├── tests/                          pytest suite (unit + behavioural)
├── tools/                          notebook generators (the source of truth) + headless runner
├── data/                           sample documents + labelled questions for RAG / MCP
├── docs/es/  docs/en/              full documentation in Spanish and English
└── setup_env.ps1  setup_env.sh  Makefile  requirements*.txt  .github/workflows/ci.yml
```

> **Notebooks are generated.** The `.ipynb` files are built from `tools/build_notebooks_part{1,2,3}.py`. To change a notebook permanently, edit the generator and run `make notebooks` (or the three scripts), then `python tools/run_notebooks.py`.

## Design principles

- **From scratch first, then the real tool.** Each technique fits in a few dozen readable lines, and the notebook names the production equivalent (`trl`, `peft`, vLLM, FAISS, OpenTelemetry, the MCP SDKs…).
- **One model class, many architectures.** `ModelConfig` switches attention type, dense or MoE FFN, and normalization, so comparisons are apples to apples.
- **Everything is evaluated, honestly.** When a technique does not help on a toy setup, the notebook says so and explains why (for example, why an LLM re-ranker can hurt, or why a denser RL reward did not raise exact match).
- **Offline first.** Synthetic corpora with known structure (so the theoretical optimum can be computed) and a deterministic `MockLLM`. Swap in real data or models with one line.

## Roadmap

**Done**
- [x] Core: tokenizer, transformer with 8 attention variants, MoE, pre-training, inference, quantization
- [x] Post-training: SFT, LoRA (+ trainable embeddings), distillation, DPO (+ NLL), GRPO
- [x] Applications: RAG, recommender systems, extraction/classification/caching/guardrails
- [x] Agentic AI, observability and MCP (notebooks 14–16, modules `agentic`, `observability`, `mcp`)
- [x] Self-contained mini GPT (notebook 00 / `scripts/mini_llm.py`)
- [x] Topic-organized notebooks, bilingual docs (ES/EN), CI, 69 tests

**Next**
- [ ] **Real-model track**: the same SFT → DPO → eval pipeline on Qwen2.5-0.5B with `transformers` + `peft` + `trl` (GPU / Colab)
- [ ] KV-cache and GQA in the mini GPT, plus a tiny-Shakespeare run with sample outputs
- [ ] MCP over Streamable HTTP, with an OAuth example and a remote server
- [ ] OpenTelemetry exporter wired to Langfuse / Phoenix, with an example dashboard
- [ ] Evaluation harness: run the notebooks' eval sets against real API models and track the results over time
- [ ] Dev container / Docker image and "Open in Colab" links for every notebook

## Documentation

| | Spanish | English |
|---|---|---|
| Index | [docs/es/README.md](docs/es/README.md) | [docs/en/README.md](docs/en/README.md) |
| Installation & environment | [instalacion.md](docs/es/instalacion.md) | [installation.md](docs/en/installation.md) |
| Notebook guide (what each one teaches and measures) | [guia_notebooks.md](docs/es/guia_notebooks.md) | [notebooks_guide.md](docs/en/notebooks_guide.md) |
| Module reference (`llmsuite` API) | [referencia_modulos.md](docs/es/referencia_modulos.md) | [module_reference.md](docs/en/module_reference.md) |
| Command-line scripts | [scripts_cli.md](docs/es/scripts_cli.md) | [cli_scripts.md](docs/en/cli_scripts.md) |
| Case study (portfolio write-up) | [caso_de_estudio.md](docs/es/caso_de_estudio.md) | [case_study.md](docs/en/case_study.md) |
| Glossary | [glosario.md](docs/es/glosario.md) | [glossary.md](docs/en/glossary.md) |

## References

Attention Is All You Need · RoFormer (RoPE) · GQA (Ainslie 2023) · DeepSeek-V2/V3 (MLA, DeepSeekMoE, aux-free balancing) · Switch Transformer · Mixtral · FlashAttention · Longformer · Linear Transformers · Chinchilla · Llama 3 · LoRA · QLoRA · InstructGPT · DPO (Rafailov 2023) · Iterative RPO (Pang 2024) · DeepSeekMath (GRPO) · DeepSeek-R1 · LLM.int8() · GPTQ · AWQ · Speculative Decoding (Leviathan 2023) · PagedAttention/vLLM · RAG (Lewis 2020) · BM25 · RRF · ReAct · Toolformer · Building Effective Agents (Anthropic 2024) · OpenTelemetry GenAI semantic conventions · Model Context Protocol specification (2025-06-18) · SASRec · HSTU.

## License

[MIT](LICENSE): use it for learning, teaching and your portfolio.
