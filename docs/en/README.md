# LLM Edu Suite — Documentation in English

🇪🇸 [Versión en español](../es/README.md) · [Main README](../../README.md)

A hands-on LLM engineering course in two halves:

1. **Inside the model (notebooks 00–06):** we build a language model from scratch in PyTorch (tokenizer, attention, RoPE, MoE, training, inference) to understand how it works.
2. **With the industry's tools (notebooks 07–16):** fine-tuning with Hugging Face (`transformers`, `peft`, `trl`), quantization with `optimum-quanto`, RAG with LangChain evaluated with RAGAS, agents with LangGraph, observability and evaluation with LangSmith, and the Model Context Protocol with the official SDK.

Everything runs locally: **Ollama** serves the LLMs (`qwen2.5:3b` and `nomic-embed-text`) and the training models are Hugging Face SmolLM2. An NVIDIA GPU makes it fast (the published results were measured on an RTX 3060, 6 GB); CPU works too. LangSmith is optional.

## Pages

| Page | What it is for |
|---|---|
| [Installation and environment](installation.md) | `.venv`, CUDA PyTorch, Ollama and its models, the `.env`, the LangSmith key, troubleshooting |
| [Notebook guide](notebooks_guide.md) | the 17 notebooks one by one: what you build, with which tools, which concepts and which results |
| [Theoretical foundations](foundations.md) | the theory behind each technique and pattern: the problem it solves, the idea, why it is used today, how it fails |
| [Module reference](module_reference.md) | the `llm_edu_suite/` package (the from-scratch model and `config.py`) and where each topic is taught with frameworks |
| [Command-line scripts](cli_scripts.md) | `mini_llm`, `train`, `sample`, `finetune` (TRL), `rag_demo` (LangChain), `mcp_server` (FastMCP) |
| [Case study](case_study.md) | the project as a portfolio case: v1 from scratch, v2 with frameworks, decisions, problems and results |
| [Glossary](glossary.md) | GQA, LoRA, DPO, GRPO, RAGAS, LCEL, StateGraph, MCP, span, SLO… |

## Topic map

| Topic | Folder | Notebooks | Tools |
|---|---|---|---|
| Start here | `notebooks/00_start_here/` | 00 · a complete mini LLM | PyTorch |
| Architecture | `notebooks/01_architecture/` | 01 tokenization · 02 attention, RoPE and normalization · 03 efficient attention · 04 FFN and MoE | PyTorch from scratch |
| Pre-training and inference | `notebooks/02_pretraining_and_inference/` | 05 pre-training · 06 inference · 09 quantization | PyTorch · `optimum-quanto` |
| Post-training | `notebooks/03_post_training/` | 07 SFT, LoRA and distillation · 08 DPO and GRPO | `transformers` · `peft` · `trl` |
| Applications | `notebooks/04_applications/` | 10 RAG · 12 recommendation · 13 application patterns | LangChain · RAGAS · PyTorch |
| Agents and MCP | `notebooks/05_agents_and_mcp/` | 11 tools and structured output · 14 agentic AI · 16 MCP | LangChain · LangGraph · `mcp` SDK |
| LLMOps | `notebooks/06_llmops/` | 15 observability and evaluation | LangSmith |

```mermaid
flowchart LR
    Z[00 · mini LLM] --> B[01 Tokenizer]
    B --> C[02-04 Transformer<br/>RoPE · GQA/MLA · MoE]
    C --> D[05 Pre-training]
    D --> E[06 Inference]
    D --> F["07-08 Post-training<br/>HF · peft · trl"]
    F --> G["09 Quantization<br/>optimum-quanto"]
    E --> H[Applications with LangChain]
    H --> I["10 RAG + RAGAS"]
    H --> K[12 Recommendation]
    H --> L[13 Patterns]
    H --> J[11 Tools]
    J --> M["14 Agents · LangGraph"]
    J --> N["16 MCP · official SDK"]
    M --> O["15 LangSmith"]
```

## Suggested learning path

0. **Overview:** 00, to see how the pieces fit before studying them one by one.
1. **Architecture:** 01 → 02 → 03 → 04. Afterwards you should be able to write a transformer from memory.
2. **Training and inference:** 05 → 06.
3. **Post-training with Hugging Face:** 07 → 08, then 09 (quantization).
4. **Applications with LangChain:** 10 → 13 → 12. Golden rule: **build your evaluation set before the system**.
5. **Agents:** 11 → 14 → 16 and finally 15, to see how everything is observed and evaluated in production.

## Principles

- **Understand the inside, then use the real tool.** The model is built by hand because that is how attention or the KV-cache stop being magic; applications are built with the frameworks you will use at work, with their current APIs.
- **Everything is evaluated, honestly.** Each notebook ends with the metric a practitioner would report. When something does not help (an LLM re-ranker that hurts, plain DPO that degrades the model, a noisy small judge), the notebook shows it and explains why.
- **Local first.** Ollama and small models: anyone can reproduce the course on a laptop at no cost. To use a cloud provider, change `LLM_PROVIDER` in `.env`.
