# LLM Edu Suite — Documentation in English

🇪🇸 [Versión en español](../es/README.md) · [Main README](../../README.md)

A hands-on LLM engineering course in two halves, with a map to start from: notebook 00 (transformers, the big picture) is read before anything else. It follows one sentence through a real small model, SmolLM2-135M, from tokens to embeddings, attention, the prediction layer by layer (the logit lens) and the next-token distribution, and shows where the parameters live, without training anything. Then:

1. **Inside the model (notebooks 01–05, 08 and 10):** we build a language model from scratch in PyTorch (tokenizer, attention, RoPE, MoE, training, inference) to understand how it works.
2. **With the industry's tools (notebooks 11, 13, 14, 16, 18, 19, 23, 25, 26 and 28):** fine-tuning with Hugging Face (`transformers`, `peft`, `trl`), quantization with `optimum-quanto`, RAG with LangChain evaluated with RAGAS, agents with LangGraph, observability and evaluation with LangSmith, and the Model Context Protocol with the official SDK.

Notebook 06 (encoders and encoder-decoders) completes the architecture topic: an encoder from scratch with MLM, ModernBERT fine-tuned for classification, embeddings + logistic regression against an LLM, and Flan-T5 with its cross-attention.

Notebook 07 (data engineering) comes right before pre-training (08), because it is what happens to text before pre-training: exact deduplication and MinHash + LSH written from scratch, quality filtering with rules and with a classifier, benchmark decontamination and synthetic data, on a real FineWeb sample, plus two models trained on raw and clean data to measure what changes.

Notebook 09 (systems) follows pre-training: it measures on the GPU where the memory and the time of a training run go (weights, gradients, Adam and activations; bf16 and activation checkpointing; FlashAttention and the attention kernels; profiling, CUDA graphs and MFU) and how training is split across several GPUs, with real DDP in two processes and a ZeRO / FSDP calculator for 7B and 70B.

Notebook 12 (serving under load) builds on 10: 10 opens up a single generation, this one serves models the way an application does, through an HTTP server and many requests at a time. With a streaming client against Ollama it measures TTFT, TPOT and throughput, prefill against decode as the prompt grows, what prefix caching saves, one request slot against four from 1 to 8 requests in flight (and Little's law), the same 1.5B model in GGUF Q4_K_M, Q8_0 and FP16 (memory, speed and accuracy), and a small gateway with fallbacks, timeouts and a cost-aware cascade from `qwen2.5:3b` to `qwen2.5:7b`.

Notebook 15 (reasoning and test-time compute) follows 14: GRPO there is how reasoning is trained, here is how it is used and measured. On GSM8K with `qwen2.5:3b` it compares a direct answer with chain of thought, self-consistency (maj@k against pass@k), best-of-N with a 7B judge as verifier and the distilled reasoning model DeepSeek-R1-Distill 1.5B, puts them all on one cost-accuracy curve, ranks them with Bradley-Terry and measures how consistent a pairwise LLM judge is when the order of the answers is swapped.

Notebook 17 (embeddings and rerankers) comes right before RAG (18): it opens up the retrieval layer that RAG is built on. On SciFact it compares BM25 with three embedding models (with and without task prefixes), writes InfoNCE by hand and fine-tunes a small bi-encoder with in-batch and hard negatives, shrinks the vectors with Matryoshka truncation and int8 / binary quantization with rescoring, and tries hybrid retrieval with two cross-encoder rerankers, measuring both quality and latency.

Notebook 20 (DSPy) follows the application patterns (19): it stops writing prompts by hand. On notebook 06's Banking77 sample it declares the classifier as a DSPy signature, compares random few-shot examples with nearest-neighbour examples chosen per message and with retrieval alone, runs three prompt optimizers (bootstrap random search, MIPROv2 and GEPA, with `qwen2.5:7b` writing instructions for the 3B model), and measures what a bigger model and a router that calls the LLM only when the neighbours disagree add, in accuracy, latency and LLM calls.

Notebook 21 (clustering and topic modeling) builds on 17: it uses the same embeddings to find out what is inside a pile of unlabeled text. On Banking77's 3,080 customer messages, treated as unlabeled logs, it compares two embedding models and UMAP before k-means, tries to choose the number of clusters with the silhouette score, lets HDBSCAN find the clusters and the noise on its own, describes each cluster with c-TF-IDF keywords and a name from `qwen2.5:3b`, and runs an emerging-intent monitoring loop. The 77 intent labels are used only to measure each step (ARI, NMI, purity).

Notebook 22 (multimodal models) also builds on 17: CLIP is trained with the same contrastive loss as a text embedding model, with images on one side. On 37 cat and dog breeds (Oxford-IIIT Pet) it measures CLIP zero-shot with bare class names, one prompt template and an ensemble of five, a linear probe with 1 to ~100 labelled images per breed, and `qwen2.5vl:3b` through Ollama as a classifier (accuracy, latency, tokens per image and what resolution changes); then structured extraction from synthetic invoice images with a JSON schema, clean and degraded, and a typographic attack, a wrong breed written on the image, against both CLIP and the VLM.

Notebook 24 (online learning and drift) follows the recommender (23): it replays MovieLens-1M as a weekly stream and measures how a frozen recommender ages against weekly retraining, a sliding window and online updates with decay (with prequential, test-then-train evaluation), whether PSI on the input distribution notices the change, and whether drift-triggered retraining beats a schedule. Then, on Banking77, it asks whether a shift in what users ask can be detected from embeddings alone, with centroid shift, MMD with a permutation test and a classifier two-sample test.

Notebook 27 (agent harnesses) follows MCP (26): it opens up the layer that turns a model into a coding agent, the kind of program Claude Code and Codex CLI are. It reads the source of mini-swe-agent, a minimal harness, function by function, extends it with a sandboxed environment and a permission policy, and runs it with Claude Haiku 5.5 (or any litellm model, local ones included) on 16 coding tasks scored by hidden tests, 3 runs each. Then it measures what an `AGENTS.md` project-memory file adds, what a prompt injection planted in the repository does, and what a modern harness profile costs and buys.

Notebook 29 (evaluation and tracing frameworks) follows LangSmith (28): it puts the same ideas on open-source infrastructure that runs on the laptop. It reproduces notebook 15's GSM8K result as an Inspect AI task, measures how much the scorer and sampling epochs change the number, and traces a LangChain pipeline over OpenTelemetry to a self-hosted Phoenix server, reading latency and tokens per step back from the trace store, writing evaluations onto the traces as annotations and measuring what tracing costs.

Notebook 30 (safety and guardrails) closes the course after observability (28 and 29): observability tells you what the system did; this notebook is about what it should and should not do. With `qwen2.5:3b` and `qwen2.5:7b` on public benchmarks (JailbreakBench and XSTest) it measures both sides of safety, harmful requests refused and safe requests wrongly refused, then three published jailbreak templates, Llama Guard 3 (1B) as an input filter, a safety system prompt and the layers combined, after validating its refusal judge against human labels.

Everything runs locally: **Ollama** serves the LLMs (`qwen2.5:3b` and `nomic-embed-text`) and the training models are Hugging Face SmolLM2. An NVIDIA GPU makes it fast (the published results were measured on an RTX 3060, 6 GB); CPU works too, except for notebook 09, which needs a CUDA GPU. LangSmith is optional.

## Pages

| Page | What it is for |
|---|---|
| [Installation and environment](installation.md) | `.venv`, CUDA PyTorch, Ollama and its models, the `.env`, the LangSmith key, troubleshooting |
| [Notebook guide](notebooks_guide.md) | the 31 notebooks one by one: what you build, with which tools, which concepts and which results |
| [Theoretical foundations](foundations.md) | the theory behind each technique and pattern: the problem it solves, the idea, why it is used today, how it fails |
| [Module reference](module_reference.md) | the `llm_edu_suite/` package (the from-scratch model and `config.py`) and where each topic is taught with frameworks |
| [Command-line scripts](cli_scripts.md) | `mini_llm`, `train`, `sample`, `finetune` (TRL), `rag_demo` (LangChain), `mcp_server` (FastMCP), `harness_run` (mini-swe-agent), `inspect_tasks/gsm8k_cot` (Inspect AI) |
| [Case study](case_study.md) | the project as a portfolio case: v1 from scratch, v2 with frameworks, decisions, problems and results |
| [Glossary](glossary.md) | GQA, logit lens, LoRA, DPO, GRPO, RAGAS, LCEL, DSPy signature, StateGraph, MCP, span, OpenTelemetry, SLO… |

## Topic map

| Topic | Folder | Notebooks | Tools |
|---|---|---|---|
| Start here | `notebooks/00_start_here/` | 00 · transformers, the big picture (read first) · 01 · a complete mini LLM | `transformers` (00) · PyTorch |
| Architecture | `notebooks/01_architecture/` | 02 tokenization · 03 attention, RoPE and normalization · 04 efficient attention · 05 FFN and MoE · 06 encoders and encoder-decoders | PyTorch from scratch · `transformers` and `scikit-learn` (06) |
| Pre-training and inference | `notebooks/02_pretraining_and_inference/` | 07 data engineering (deduplication, filtering, decontamination) · 08 pre-training · 09 systems (memory, precision, kernels, parallelism) · 10 inference · 11 quantization · 12 serving under load (TTFT, batching, caching, GGUF, gateways) | PyTorch · `datasets` and `transformers` (07) · `torch.profiler` and `torch.distributed` (09) · `optimum-quanto` (11) · Ollama over HTTP (12) |
| Post-training | `notebooks/03_post_training/` | 13 SFT, LoRA, distillation and model merging · 14 DPO and GRPO · 15 reasoning and test-time compute | `transformers` · `peft` · `trl` · LangChain (Ollama) (15) |
| Applications | `notebooks/04_applications/` | 16 tools and structured output · 17 embeddings and rerankers · 18 RAG · 19 application patterns · 20 prompt optimization with DSPy (signatures, kNN few-shot, MIPROv2, GEPA) · 21 clustering and topic modeling (UMAP, HDBSCAN, c-TF-IDF) · 22 multimodal models (CLIP, VLMs, document extraction) · 23 recommendation · 24 online learning and drift | LangChain · RAGAS · PyTorch · `sentence-transformers` (17, 20, 21) · DSPy (20) · `umap-learn` and `scikit-learn` (21) · CLIP with `transformers` and `qwen2.5vl:3b` through Ollama (22) |
| Agents and MCP | `notebooks/05_agents_and_mcp/` | 25 agentic AI · 26 MCP · 27 agent harnesses (mini-swe-agent, coding tasks, `AGENTS.md`) | LangGraph · `mcp` SDK · mini-swe-agent with Claude Haiku 5.5 or Ollama (27) |
| LLMOps | `notebooks/06_llmops/` | 28 observability and evaluation · 29 evaluation and tracing frameworks (Inspect AI, Phoenix) · 30 safety: refusals, jailbreaks and guard models | LangSmith · Inspect AI and Arize Phoenix on OpenTelemetry (29) · Llama Guard 3 through Ollama (30) |

```mermaid
flowchart LR
    ZZ["00 · Transformers, the big picture<br/>tokens · attention · logit lens"] --> Z
    Z[01 · mini LLM] --> B[02 Tokenizer]
    B --> C[03-05 Transformer<br/>RoPE · GQA/MLA · MoE]
    C --> Q["06 Encoders and encoder-decoders<br/>BERT · ModernBERT · T5"]
    C --> P["07 Data<br/>MinHash · filters · decontamination"]
    P --> D[08 Pre-training]
    D --> S["09 Systems<br/>memory · kernels · parallelism"]
    S --> E[10 Inference]
    E --> G["11 Quantization<br/>optimum-quanto"]
    E --> V["12 Serving under load<br/>TTFT · batching · caching · gateway"]
    D --> F["13-14 Post-training<br/>HF · peft · trl"]
    F --> R["15 Reasoning<br/>CoT · self-consistency · verifiers"]
    E --> H[Applications with LangChain]
    H --> J[16 Tools]
    H --> T["17 Embeddings and rerankers<br/>bi-encoder · InfoNCE · Matryoshka"]
    T --> I["18 RAG + RAGAS"]
    H --> L[19 Patterns]
    L --> DS["20 DSPy<br/>signatures · kNN few-shot · optimizers"]
    T --> X["21 Clustering and topics<br/>UMAP · HDBSCAN · c-TF-IDF"]
    T --> W["22 Multimodal models<br/>CLIP · VLM · typographic attacks"]
    H --> K[23 Recommendation]
    K --> Y["24 Drift and online learning<br/>PSI · MMD · retraining"]
    J --> M["25 Agents · LangGraph"]
    J --> N["26 MCP · official SDK"]
    N --> AA["27 Agent harnesses<br/>mini-swe-agent · AGENTS.md · hidden tests"]
    M --> O["28 LangSmith"]
    O --> PH["29 Inspect AI + Phoenix<br/>tasks · scorers · OpenTelemetry"]
    O --> U["30 Safety<br/>jailbreaks · over-refusal · Llama Guard"]
```

## Suggested learning path

0. **Overview:** 00, to see what a real model does with a sentence, then 01, to build a whole one and see how the pieces fit before studying them one by one. After 00 you should know that a model reads token ids, that its output is a probability distribution turned into text by a decoding rule, and that most of its parameters are in the feed-forward layers.
1. **Architecture:** 02 → 03 → 04 → 05 → 06. Afterwards you should be able to write a transformer from memory and know when an encoder or an encoder-decoder is a better fit than a decoder-only model.
2. **Data, training and inference:** 07 → 08 → 09 → 10 → 11 → 12. First which text goes into the model and why, then how a model is trained on it and what that training costs in memory and time, on one GPU and on many. After 12 you should know what users wait for when a model is served (TTFT, TPOT, percentiles), why batching, prefix caching and quantized weights help, and how to size a server from its request rate.
3. **Post-training with Hugging Face:** 13 → 14 → 15. After 15 you should know what reasoning tokens buy, when voting or a verifier is worth its cost, and how to tell whether a judge can be trusted.
4. **Applications with LangChain:** 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24. After 17 you should know how to choose, fine-tune, compress and evaluate a retriever before putting it into RAG. After 20 you should know how to write an LLM step as a signature instead of a prompt, why the examples you put in a prompt matter more than its wording, what prompt optimizers search and when they cannot help, and why a non-LLM baseline and a cascade come before any of them. After 21 you should know how to find the topics in unlabeled logs, why the number of clusters is a choice rather than a discovery, and how to watch for new intents. After 22 you should know when to classify images with embeddings and when a VLM is worth its cost, what an image costs in tokens, and why text inside an image is untrusted input. After 24 you should know why a model that worked last month may not work now, how to evaluate it on a stream, and why watching input distributions alone can miss the drift that matters. Golden rule: **build your evaluation set before the system**.
5. **Agents and LLMOps:** 25 → 26 → 27, then 28, to see how everything is observed and evaluated in production, then 29 and 30. After 27 you should know what a coding agent's harness does besides calling the model, why an agent's own "done" needs hidden tests behind it, and why the defence against instructions hidden in a repository belongs in the harness. After 29 you should know how to define an evaluation as code (dataset, solver, scorer, epochs), why a framework has to reproduce your trusted number before you adopt it, why the scorer is part of the metric, and how OpenTelemetry traces, spans and annotations give you per-step latency, tokens and searchable failures on your own machine. After 30 you should know how to measure a model's refusals and its over-refusals, why a model's own refusals are not enough of a defence, and what a guard model costs in false positives.

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

**Cross-cutting, for any of them.**
- **Models.** The local defaults (`qwen2.5:3b`, SmolLM2) are chosen for a 6 GB GPU. In production pick by measured quality on *your* evaluation set, then by cost and latency. Switching provider is a `.env` change (`LLM_PROVIDER`, `LLM_MODEL`, `HARNESS_MODEL`).
- **Evaluation first.** Before changing a prompt, a model or a retriever, have a fixed evaluation set and a number to beat. Report the spread across runs, not one run.
- **Secrets and data.** Keys in a secret manager, not in `.env` on a server; prompts and traces treated as user data (retention, PII masking, access control).
- **Cost.** Log tokens per request from day one; cache what repeats; use the smallest model that passes the evaluation.

## How to use this repo

- **To study:** follow the numbers. Each notebook explains the idea, runs it, and ends with what was measured and how it is used in production. `foundations.md` has the theory behind each topic.
- **At work:** copy the pattern, not the notebook. Start from the closest notebook, replace the data and the model, keep its evaluation, and harden it with the table above.
- **As practice material for others:** every notebook runs on a laptop GPU, and every result is reproducible from its generator. Change one thing (a model, a chunk size, a prompt), rerun, and compare the numbers.

## Working with a coding agent

Most engineering work today is done together with a coding agent (Claude Code, Codex CLI, opencode). This repo is set up for it: [`AGENTS.md`](../../AGENTS.md) tells an agent how the repo works (commands, how notebooks are generated, the rules: measured numbers only, bilingual docs, no secrets, one GPU job at a time). Prompts that work well:

- *"Read AGENTS.md and notebook 18. Adapt the RAG pipeline to the PDFs in `my_docs/`, keep the same evaluation, and tell me which metrics changed."*
- *"In notebook 13, replace the booking task with my extraction task in `data/my_task.jsonl`. Train LoRA r=16, evaluate on a held-out 20%, and write the Results table from the executed outputs only."*
- *"Run notebook 27 with `HARNESS_MODEL=openai/...` and compare its solve rate, tokens and cost with the Claude results in the notebook."*
- *"Review `scripts/harness_run.py` for ways a task could escape the sandbox, and propose deny rules with a test for each."*

Review what the agent changes, run the tests, and read the outputs before trusting a number.

## Principles

- **Understand the inside, then use the real tool.** The model is built by hand because that is how attention or the KV-cache stop being magic; applications are built with the frameworks you will use at work, with their current APIs.
- **Everything is evaluated, honestly.** Each notebook ends with the metric a practitioner would report. When something does not help (an LLM re-ranker that hurts, plain DPO that degrades the model, a noisy small judge), the notebook shows it and explains why.
- **Local first.** Ollama and small models: anyone can reproduce the course on a laptop at no cost. To use a cloud provider, change `LLM_PROVIDER` in `.env`.
