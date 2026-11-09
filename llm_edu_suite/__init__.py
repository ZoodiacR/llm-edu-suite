"""
llm_edu_suite — a compact, readable LLM engineering toolkit for learning and teaching.

Every module is self-contained and generously commented. The goal is clarity
rather than peak performance: each component mirrors what production frameworks
(PyTorch, Hugging Face transformers/peft/trl, vLLM, FAISS) do under the hood.

Modules
-------
utils       seeding, device selection, parameter counting, timing
tokenizer   byte-level BPE tokenizer (train / encode / decode / save / load)
layers      RMSNorm, RoPE, attention variants (MHA, GQA, MLA, sliding window,
            Longformer, block-sparse, linear), SwiGLU, MoE
model       TransformerLM: configurable decoder-only LM with KV-cache + sampling
data        synthetic corpora, text datasets, batching, SFT formatting/collation
training    LR schedules, optimizer groups, training loop, checkpoints
finetune    SFT loss masking, LoRA, DPO, GRPO, knowledge distillation
quant       block-wise INT8/INT4 and NF4 quantization, quantized Linear
rag         chunking, embeddings, BM25, hybrid retrieval (RRF), prompt building
evaluation  perplexity, exact match, retrieval metrics, LLM-as-judge, latency
agents      tool registry, JSON-schema validation, agent loop
recsys      sequential recommendation with a transformer (SASRec-style),
            embedding retrieval, LLM re-ranking, leave-one-out split + baselines
applications extraction, classification, summarization, semantic cache,
            prompt-injection guard, batch jobs with cost accounting
llm_client  OpenAI-compatible client (vLLM / any API) with an offline MockLLM
"""

__version__ = "0.1.0"
