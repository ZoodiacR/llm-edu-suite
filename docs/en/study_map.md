# Study map: books and courses ↔ notebooks

🇪🇸 [Versión en español](../es/mapa_de_estudio.md) · [← Back to the index](README.md)

This page maps the course against widely used reference material for LLM engineering: which chapter or lecture is practised in which notebook, and which topics **do not have a hands-on part yet**. Read a chapter, then open the matching notebook, or the other way round.

Legend: ✅ covered with code and measured results · 🟡 partly covered (theory, or one variant only) · ❌ no hands-on part yet.

## Books

### Build a Large Language Model (From Scratch), Sebastian Raschka

| Chapter | Notebook | |
|---|---|---|
| 2 Text, tokenization, BPE, sliding window | [01](../../notebooks/01_architecture/01_tokenization.ipynb), [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | ✅ |
| 3 Attention (self, causal, multi-head) | [02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) | ✅ (plus RoPE, GQA, MLA) |
| 4 A full GPT model, generating text | [00](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb), [02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) | ✅ (Llama-style architecture) |
| 5 Pretraining, decoding, saving weights | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb), [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | ✅ |
| 5.5 Loading pretrained weights into your own model | — | ❌ |
| 6 Fine-tuning for classification (classification head) | — | ❌ |
| 7 Instruction fine-tuning, LLM-judge evaluation | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | ✅ (with `trl`) |
| Appendix LoRA | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) | ✅ (and QLoRA) |

### Hands-On Large Language Models, Jay Alammar and Maarten Grootendorst

| Chapter | Notebook | |
|---|---|---|
| 1–3 Tokens, embeddings, inside an LLM | [01](../../notebooks/01_architecture/01_tokenization.ipynb)–[06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | ✅ |
| 4 Text classification | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | 🟡 LLM only (zero/few-shot); embeddings + classifier and a fine-tuned encoder are missing |
| 5 Clustering and topic modeling | — | ❌ |
| 6 Prompt engineering | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb), [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | ✅ |
| 7 Advanced generation: chains, memory, agents | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 8 Semantic search and RAG | [10](../../notebooks/04_applications/10_rag.ipynb) | 🟡 the **cross-encoder reranker** is not practised |
| 9 Multimodal LLMs | — | ❌ |
| 10 Creating text embedding models (contrastive) | — | ❌ |
| 11 Fine-tuning representation models (BERT) | — | ❌ |
| 12 Fine-tuning generation models (QLoRA, DPO) | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |

### AI Engineering, Chip Huyen

| Chapter | Notebook | |
|---|---|---|
| 2 Foundation models: sampling, structured outputs | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb), [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | ✅ (test-time compute such as best-of-N and self-consistency is missing) |
| 3–4 Evaluation methodology, evaluating systems | [10](../../notebooks/04_applications/10_rag.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb), [12](../../notebooks/04_applications/12_recommender_systems.ipynb) | ✅ (judges, bootstrap, datasets) · 🟡 comparative evaluation (Elo / Bradley-Terry) |
| 5 Prompt engineering and defensive prompting | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | ✅ |
| 6 RAG and agents | [10](../../notebooks/04_applications/10_rag.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 7 Finetuning (memory, PEFT, model merging) | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) | ✅ · ❌ model merging |
| 8 Dataset engineering (curation, synthesis, dedup) | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) (distillation) | 🟡 deduplication, quality filtering and synthetic data at scale are missing |
| 9 Inference optimization | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb), [09](../../notebooks/02_pretraining_and_inference/09_quantization.ipynb) | ✅ from scratch · 🟡 no real serving engine measured |
| 10 Architecture: guardrails, router, gateway, caching, feedback | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | 🟡 a gateway with cost-based routing between models is missing |

### Hands-On LLM Serving and Optimization

| Chapter | Notebook | |
|---|---|---|
| 1–5 Serving: system design, best practices, challenges | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | 🟡 concepts implemented by hand |
| 6–7 Optimization (quantization, KV-cache, batching, speculative) | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb), [09](../../notebooks/02_pretraining_and_inference/09_quantization.ipynb) | ✅ |
| 8–9 Serving frameworks (vLLM, SGLang, TGI) and optimization in practice | — | ❌ a real benchmark of TTFT, TPOT and throughput under load |

### Large Language Models: The Hard Parts

| Chapter | Notebook | |
|---|---|---|
| 2–3 The evals gap, evaluation tools | [10](../../notebooks/04_applications/10_rag.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | ✅ |
| 4 From data to context | [10](../../notebooks/04_applications/10_rag.ipynb), [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | ✅ |
| 5 Structured output | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | ✅ |
| 6–7 Safety and safety evaluation | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | 🟡 prompt injection only; jailbreaks, safety classifiers and over-refusal are missing |
| 8 Alignment: a case study | [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |
| Appendix: local deployment | [installation](installation.md) | ✅ (Ollama) |

### Build an AI Agent (From Scratch)

| Chapter | Notebook | |
|---|---|---|
| 1–4 LLMs, tools, a basic ReAct agent | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 5 Knowledge bases with RAG | [10](../../notebooks/04_applications/10_rag.ipynb) | ✅ |
| 6 Memory | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 7 Planning and reflection | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 8 Code execution | — | ❌ a code-execution sandbox |
| 9 Multi-agent systems | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | 🟡 theory only |
| 10 Evaluating agents | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb), [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | ✅ |

### AI Agents with MCP · Building AI Agent Platforms

| Topic | Notebook | |
|---|---|---|
| Host, client, server; tools, resources, prompts; stdio | [16](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | ✅ |
| Client capabilities: sampling, roots, elicitation; several servers | — | ❌ |
| Streamable HTTP with authentication | [16](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | 🟡 explained, not executed |
| Agent platforms: deploying as a service, identity, governance, cost | [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | 🟡 observability only |

### Hands-On ML (Géron) and Transactional ML with Data Streams

They are the foundation for classic ML and streaming learning. The course touches them sideways: metrics, validation and LightGBM in [12](../../notebooks/04_applications/12_recommender_systems.ipynb). Online learning and concept drift have no hands-on part (❌).

## Stanford courses

**CS336, Language Modeling from Scratch (2025)**, the course that builds an LM from scratch:

| Block | Notebook | |
|---|---|---|
| Tokenization | [01](../../notebooks/01_architecture/01_tokenization.ipynb) | ✅ |
| Architectures and hyperparameters, MoE | [02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb)–[04](../../notebooks/01_architecture/04_ffn_moe.ipynb) | ✅ |
| GPUs, kernels (FlashAttention, Triton), profiling | [03](../../notebooks/01_architecture/03_efficient_attention.ipynb) | 🟡 theory; no profiling or kernels |
| Parallelism (data, tensor, pipeline, FSDP) | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | 🟡 theory (single GPU) |
| Scaling laws | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | ✅ |
| Inference | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | ✅ |
| Evaluation | [10](../../notebooks/04_applications/10_rag.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | ✅ |
| Data: filtering, deduplication, mixtures | — | ❌ |
| Alignment: SFT, RLHF, RL with verifiable rewards | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |

**CME 295, Transformers & Large Language Models (2025)**, the overview course:

| Block | Notebook | |
|---|---|---|
| Transformer, attention, tokenization, embeddings | [01](../../notebooks/01_architecture/01_tokenization.ipynb)–[02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) | ✅ |
| Variants: encoder-only (BERT) and encoder-decoder (T5) | — | ❌ the course only builds decoder-only models |
| Training, tuning, preference tuning | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb), [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |
| Reasoning models and test-time compute | [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | 🟡 GRPO yes; self-consistency, best-of-N and their cost-accuracy curve are missing |
| RAG, agents, evaluation | [10](../../notebooks/04_applications/10_rag.ipynb)–[16](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | ✅ |

## What is missing, by priority

Priority reflects how central the topic is (how many sources cover it) and what it adds in practice.

| # | Topic | Sources | Proposal |
|---|---|---|---|
| 1 | **Load real weights into our own model** (SmolLM2 into `TransformerLM`) and check the logits match Hugging Face | Raschka 5.5, CS336 | new section in 06: joins the two halves of the course |
| 2 | **Embeddings and rerankers**: fine-tune an embedding model (contrastive loss) and add a cross-encoder, measured on notebook 10's RAG | Alammar ch. 8 and 10, Huyen ch. 6 | new notebook |
| 3 | **Reasoning and test-time compute**: CoT, self-consistency, best-of-N with a verifier, the cost-accuracy curve | CME 295, CS336, Huyen ch. 2 | new notebook |
| 4 | **Real serving**: TTFT, TPOT and throughput under concurrency; quantized GGUF; optional vLLM on WSL2 | LLM Serving ch. 8–9, Huyen ch. 9 | new notebook (already on the roadmap) |
| 5 | **Encoders (BERT/ModernBERT)**: fine-tuned classification vs embeddings + classifier vs zero-shot LLM, with cost per request | Raschka ch. 6, Alammar ch. 4 and 11, CME 295 | new notebook |
| 6 | **Data engineering**: MinHash deduplication, quality filtering, synthetic data, decontamination | CS336, Huyen ch. 8 | new notebook |
| 7 | **Safety**: jailbreaks, a local safety classifier, refusal vs over-refusal rates | The Hard Parts ch. 6–7 | extend 13 or new |
| 8 | **Advanced agents**: multi-agent (supervisor, handoffs) and sandboxed code execution | Build an AI Agent ch. 8–9 | extend 14 |
| 9 | **Advanced MCP**: sampling, roots, elicitation, several servers, Streamable HTTP | AI Agents with MCP ch. 2 | extend 16 |
| 10 | Multimodal (CLIP, vision models in Ollama) | Alammar ch. 9 | new notebook (optional) |
| 11 | Systems: profiling, memory, `torch.compile`, parallelism explained | CS336 | new notebook (limited to one GPU) |
| 12 | Model merging (task arithmetic, TIES) | Huyen ch. 7 | section in 07 |
| 13 | Clustering and topic modeling; online learning and drift | Alammar ch. 5, Transactional ML | optional |
