"""
llm_edu_suite: the small amount of code the course needs that is not a framework.

The course has two halves. In the first one (notebooks 01-05, 08 and 10) we build a language model ourselves
in PyTorch, because seeing attention, RoPE, MoE or a KV-cache written out is the best way to
understand them; the modules below are that model. In the second half (notebooks 11, 13, 14, 16, 18, 19, 23, 25, 26 and 28) we use
the tools the industry uses: Hugging Face transformers/peft/trl, LangChain, LangGraph, LangSmith,
RAGAS and the MCP SDK. For those, this package only provides `config`, the one place where the
models and services are chosen.

Modules
-------
config      get_llm(), get_embeddings(), setup(): models and services for the framework notebooks, read from .env
utils       seeding, device selection, parameter counting, timing
tokenizer   byte-level BPE tokenizer (train / encode / decode / save / load)
layers      RMSNorm, RoPE, attention variants (MHA, GQA, MLA, sliding window, Longformer,
            block-sparse, linear), SwiGLU, Mixture of Experts
model       TransformerLM: a configurable decoder-only LM with KV-cache and sampling
data        synthetic corpora, text datasets and batching for pre-training
training    LR schedules, optimizer groups, the training loop, checkpoints
inference   speculative decoding, batched generation, KV-cache sizing
recsys      sequential recommendation with a transformer (SASRec-style), embedding retrieval,
            leave-one-out split and baselines
evaluation  perplexity, exact match, text-overlap, retrieval/ranking and latency metrics
"""

__version__ = "2.0.0"
