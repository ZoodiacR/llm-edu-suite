"""Notebook 10 - Retrieval-augmented generation

Builds 10_rag.ipynb. Run: python tools/build_notebooks.py 10
"""
from _common import *  # noqa: F401,F403

# Notebook 10 - Retrieval-augmented generation
nb = NB()
nb.md(r'''
# 10 · Retrieval-Augmented Generation

Models know what was in their training data up to a cutoff, approximately. RAG separates **knowledge** (a searchable store you control) from **reasoning** (the model):

```
documents → chunk → embed → index                        (offline)
question → dense search ─┐
         → BM25 search ──┴→ RRF fusion → (rerank) → top-k → prompt with citations → LLM
```

**What you will learn:** every component implemented from scratch (NumPy only), where each fails, and — most importantly — how to **evaluate retrieval and generation separately**. The LLM is a `MockLLM` by default so the notebook runs offline; swap in `OpenAICompatibleClient` for a real model (one line).

**Source:** `llm_edu_suite/rag.py`, `llm_edu_suite/llm_client.py`, `llm_edu_suite/evaluation.py`.
''')
nb.code(SETUP_NOTORCH)
nb.code(r'''
from llm_edu_suite.rag import chunk_documents, HashedTfidfEmbedder, BM25, VectorIndex, reciprocal_rank_fusion, Retriever, build_rag_prompt
from llm_edu_suite.llm_client import MockLLM, OpenAICompatibleClient
from llm_edu_suite.evaluation import evaluate_ranking, context_support, token_f1, pairwise_judge

# A small internal "knowledge base". Swap in your own documents as a dict of name to text.
docs = {
    "rope.md": "Rotary Position Embedding (RoPE) rotates query and key vectors according to their absolute position. Because rotations compose by adding angles, the attention score depends only on the relative distance between tokens. The default base is 10000; Llama 3 raised the base to 500000 to support longer contexts. YaRN and position interpolation rescale the frequencies to extend context after training.",
    "moe.md": "A Mixture of Experts layer replaces the dense feed-forward network with several expert networks and a router. The router selects the top-k experts for each token and the outputs are combined with the router weights. DeepSeek-V3 uses 256 routed experts, activates 8 per token, and adds one shared expert. An auxiliary loss or a bias-based mechanism keeps the expert load balanced.",
    "lora.md": "LoRA freezes the pretrained weights and trains two low-rank matrices A and B whose product approximates the weight update. B is initialized to zero so training starts from the original model. Typical ranks are 8 to 64 with alpha set to twice the rank. QLoRA applies LoRA on top of a 4-bit NF4 quantized base model. Adapters can be merged into the weights for inference.",
    "quant.md": "Block-wise quantization assigns one scale to each group of 32 to 128 weights, which limits the effect of outliers. NF4 places quantization levels at the quantiles of a Gaussian distribution. AWQ protects salient channels by scaling them before quantization. FP8 is supported natively on Hopper GPUs and was used end to end for training DeepSeek-V3.",
    "kvcache.md": "The KV cache stores keys and values of past tokens so decoding does not recompute them. Its size grows with layers, heads, head dimension and context length. Grouped-query attention shares key and value heads across query heads to shrink the cache. Multi-head latent attention compresses keys and values into a low-rank latent vector of 512 dimensions in DeepSeek-V3.",
    "serving.md": "vLLM uses PagedAttention to store the KV cache in non-contiguous blocks and continuous batching to add requests mid-flight. Prefix caching reuses the computation of shared system prompts. Speculative decoding uses a small draft model to propose tokens that the large model verifies in one pass. Time to first token measures prefill latency and tokens per second measures decode speed.",
}
chunks = chunk_documents(docs, size=40, overlap=10)
print(f"{len(docs)} documents -> {len(chunks)} chunks")
print("example chunk:", chunks[0])
''')
nb.md(r'''
## 1. Embeddings and dense retrieval
A neural embedder (BGE, E5, Qwen3-Embedding, text-embedding-3) maps text to a vector so that similar meanings are close. Offline we use `HashedTfidfEmbedder` — lexical only, but the mechanics (normalize → dot product = cosine → top-k) are identical. Replace with `SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")` for real semantics.
''')
nb.code(r'''
texts = [c.text for c in chunks]
emb = HashedTfidfEmbedder(dim=512).fit(texts)
E = emb.encode(texts)
index = VectorIndex(E)
q = "how many experts does DeepSeek-V3 activate per token?"
ids, scores = index.search_with_scores(emb.encode([q])[0], k=3)
for i, s in zip(ids, scores):
    print(f"{s:.3f}  {chunks[i].doc_id}#{chunks[i].chunk_id}: {chunks[i].text[:90]}...")
''')
nb.md(r'''
## 2. Lexical retrieval (BM25) and hybrid fusion
Dense retrieval misses exact strings (model names, error codes, numbers); BM25 misses paraphrases. Run both and fuse with **Reciprocal Rank Fusion**: `score(d) = Σ 1/(60 + rank)`. Robust, no tuning.
''')
nb.code(r'''
bm25 = BM25(texts)
dense = index.search(emb.encode([q])[0], k=5)
lexical = bm25.search(q, k=5)
fused = reciprocal_rank_fusion(dense, lexical)
print("dense  :", dense); print("bm25   :", lexical); print("fused  :", fused[:5])
''')
nb.md(r'''
## 3. The full pipeline with citations
`Retriever` wires it together (optionally with a cross-encoder `reranker(query, texts) -> scores` on the top candidates). `build_rag_prompt` numbers the chunks and asks for citations `[n]` — citations make answers checkable, which is the whole point.
''')
nb.code(r'''
retriever = Retriever(chunks)                      # hybrid (dense + BM25) by default
llm = MockLLM()                                    # for a real model: OpenAICompatibleClient("Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")

def rag_answer(question, k=3):
    top = retriever.retrieve(question, k=k)
    prompt = build_rag_prompt(question, top)
    return llm.complete(prompt), top, prompt

answer, top, prompt = rag_answer(q)
print(prompt[:600], "...\n")
print("ANSWER:", answer)
''')
nb.md(r'''
## 4. Evaluation — retrieval and generation, separately

A RAG system fails in two different places and you must know which:

**Retrieval metrics** (needs (question → relevant chunk ids) labels; build 50–200 of them for your corpus):
recall@k (is the answer in the context at all?), MRR, nDCG. If recall@5 is 60 %, no prompt engineering will fix the other 40 %.

**Generation metrics** given good context: token-F1 / ROUGE vs reference answers, **faithfulness** (is every claim supported by the context? here a cheap `context_support` proxy; in production an LLM judge with a rubric or RAGAS), and **citation accuracy**.
''')
nb.code(r'''
# A small labelled eval set: each question comes with its relevant document and a reference answer.
eval_set = [
    ("What base does Llama 3 use for RoPE?", "rope.md", "Llama 3 raised the RoPE base to 500000."),
    ("How many experts does DeepSeek-V3 activate per token?", "moe.md", "DeepSeek-V3 activates 8 of 256 experts per token."),
    ("How is matrix B initialized in LoRA?", "lora.md", "B is initialized to zero so training starts from the original model."),
    ("What does AWQ do to salient channels?", "quant.md", "AWQ scales salient channels before quantization to protect them."),
    ("What does grouped-query attention do to the KV cache?", "kvcache.md", "GQA shares key and value heads across query heads to shrink the cache."),
    ("What is PagedAttention?", "serving.md", "PagedAttention stores the KV cache in non-contiguous blocks."),
    ("Which latent dimension does MLA use in DeepSeek-V3?", "kvcache.md", "MLA compresses keys and values into a 512-dimensional latent vector."),
    ("What does prefix caching reuse?", "serving.md", "Prefix caching reuses the computation of shared system prompts."),
]

def retrieval_eval(retriever_fn, name):
    queries = []
    for question, rel_doc, _ in eval_set:
        ranked = [c.doc_id for c in retriever_fn(question, 10)]
        # Any chunk from the right document counts as relevant, so we rank document ids with duplicates removed.
        dedup = list(dict.fromkeys(ranked))
        queries.append((dedup, {rel_doc}))
    m = evaluate_ranking(queries, ks=(1, 3))
    print(f"{name:12s} " + "  ".join(f"{k} {v:.2f}" for k, v in m.items()))
    return m

dense_only = Retriever(chunks, use_bm25=False)
bm25_only_fn = lambda qq, k: [chunks[i] for i in bm25.search(qq, k)]
retrieval_eval(lambda qq, k: dense_only.retrieve(qq, k), "dense")
retrieval_eval(bm25_only_fn, "bm25")
retrieval_eval(lambda qq, k: retriever.retrieve(qq, k), "hybrid")
''')
nb.code(r'''
# Now score the generated answers, given the context that was retrieved.
rows = []
for question, rel_doc, ref in eval_set:
    ans, top, _ = rag_answer(question, k=3)
    ctx = " ".join(c.text for c in top)
    rows.append((question[:45], token_f1(ans, ref), context_support(ans, ctx), any(c.doc_id == rel_doc for c in top)))
print(f"{'question':47s} {'F1':>5s} {'support':>8s} {'hit@3':>6s}")
for q_, f1, sup, hit in rows:
    print(f"{q_:47s} {f1:5.2f} {sup:8.2f} {str(hit):>6s}")
print(f"\nmean token-F1 {np.mean([r[1] for r in rows]):.2f} | mean context support {np.mean([r[2] for r in rows]):.2f} | retrieval hit@3 {np.mean([r[3] for r in rows]):.2f}")
''')
nb.md(r'''
### LLM-as-judge for open answers
For questions without a single reference string, compare two systems pairwise with an LLM judge, **swapping positions** to cancel position bias (only count agreements). With `MockLLM` this is a dry run; with a real model it is how Arena-style evaluations work.
''')
nb.code(r'''
no_rag_answer = lambda question: llm.complete(question)          # the model on its own, without retrieved context
wins = {"A": 0, "B": 0, "TIE": 0}
for question, _, _ in eval_set:
    a, _, _ = rag_answer(question)
    b = no_rag_answer(question)
    wins[pairwise_judge(llm, question, a, b)] += 1
print("RAG (A) vs no-RAG (B) judged by the LLM:", wins)
''')
nb.md(r'''
## 5. Failure modes and fixes (checklist)

| symptom | likely cause | fix |
|---|---|---|
| answer exists in docs but was not retrieved | chunking too coarse / lexical mismatch | smaller chunks with overlap, hybrid search, query rewriting / multi-query (`expand_query`), HyDE |
| retrieved right chunk, model ignored it or hallucinated | too much context, weak instructions | fewer, better chunks; reranker; explicit "only from context"; require citations |
| needs several documents | single-shot retrieval | iterative / agentic retrieval, GraphRAG |
| slow | brute-force index, big reranker | ANN index (HNSW), rerank only top-50, cache embeddings |
| wrong answers on names / codes / versions | pure dense retrieval | BM25 in the mix |

Production stack: `sentence-transformers` or an embedding API → FAISS / pgvector / Qdrant → `bge-reranker` → vLLM; evaluate with RAGAS or your own labeled set (as above).

**Next:** `11_agents_tools_structured_output.ipynb`.
''')
nb.save(notebook_path("10_rag.ipynb"))
print("10_rag.ipynb written")
