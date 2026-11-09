"""
Retrieval-Augmented Generation (RAG), from scratch with NumPy (no torch needed).

Pipeline
--------
Offline, we chunk the documents, embed the chunks and build an index. At query time we run a
dense search and a BM25 keyword search side by side, fuse the two rankings with RRF, optionally
rerank, and put the top-k chunks into the prompt:

documents -> chunk -> embed -> index          (offline)
question  -> embed -> dense search ─┐
          -> BM25 keyword search  ──┴-> RRF fusion -> (rerank) -> top-k -> prompt -> LLM

Components
----------
chunk_text / chunk_documents     overlapping word-window chunks that remember their source
HashedTfidfEmbedder              a toy offline embedding: hashed bag-of-words with IDF weights
SentenceTransformerEmbedder      the same interface over a real model (optional dependency)
BM25                             the classic lexical ranking function
VectorIndex                      brute-force cosine search (use FAISS or a vector DB at scale)
reciprocal_rank_fusion           merges the rankings of several retrievers
Retriever                        hybrid retrieval with an optional reranker callback
build_rag_prompt                 numbered context blocks plus citation instructions
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


# Chunking

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int
    meta: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Fixed-size word windows with some overlap. In production you would count tokens and respect paragraph and section boundaries."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, step)]


def chunk_documents(docs: dict[str, str], size: int = 120, overlap: int = 30) -> list[Chunk]:
    out = []
    for doc_id, text in docs.items():
        for j, c in enumerate(chunk_text(text, size, overlap)):
            out.append(Chunk(c, doc_id, j))
    return out


# Embedders

class HashedTfidfEmbedder:
    """
    An offline stand-in for a neural embedding model. Each word is hashed into one of `dim` buckets
    and weighted by TF-IDF, and the vectors are L2-normalized so that a dot product is a cosine.
    It only sees lexical overlap (it knows nothing about synonyms), but that is enough to show the
    mechanics. For real use, swap in SentenceTransformerEmbedder.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.idf: dict[str, float] = {}
        self.n_docs = 0

    def fit(self, texts: list[str]) -> "HashedTfidfEmbedder":
        df = Counter(t for text in texts for t in set(tokenize(text)))
        self.n_docs = len(texts)
        self.idf = {t: math.log((1 + self.n_docs) / (1 + n)) + 1.0 for t, n in df.items()}
        return self

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for t, n in Counter(tokenize(text)).items():
                out[i, self._bucket(t)] += n * self.idf.get(t, 1.0)
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    """A thin wrapper over sentence-transformers (pip install sentence-transformers) with the same interface."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


# Lexical search with BM25

class BM25:
    """Okapi BM25. k1 controls how quickly repeated terms stop adding score, and b controls how much long documents are penalized."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.tf = [Counter(d) for d in self.docs]
        self.df = Counter(t for d in self.docs for t in set(d))
        self.N = len(self.docs)
        self.avg_len = sum(len(d) for d in self.docs) / max(1, self.N)

    def idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, i: int) -> float:
        tf, L = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in tokenize(query):
            if t in tf:
                s += self.idf(t) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * L / self.avg_len))
        return s

    def search(self, query: str, k: int = 10) -> list[int]:
        scores = [(self.score(query, i), i) for i in range(self.N)]
        return [i for s, i in sorted(scores, reverse=True)[:k] if s > 0]


# Dense search

class VectorIndex:
    """Exact cosine search over normalized vectors. At scale you would use FAISS (HNSW/IVF) or a vector database instead."""

    def __init__(self, embeddings: np.ndarray):
        self.E = embeddings.astype(np.float32)

    def search(self, q: np.ndarray, k: int = 10) -> list[int]:
        scores = self.E @ q.reshape(-1)
        return np.argsort(-scores)[:k].tolist()

    def search_with_scores(self, q: np.ndarray, k: int = 10):
        scores = self.E @ q.reshape(-1)
        idx = np.argsort(-scores)[:k]
        return idx.tolist(), scores[idx].tolist()


def reciprocal_rank_fusion(*rankings: list[int], k: int = 60) -> list[int]:
    """
    score(d) = sum over rankings of 1 / (k + rank). It only looks at ranks, never raw scores, so it
    combines very different retrievers robustly and has a single parameter to tune.
    """
    scores: Counter = Counter()
    for r in rankings:
        for rank, i in enumerate(r):
            scores[i] += 1.0 / (k + rank + 1)
    return [i for i, _ in scores.most_common()]


# Retriever

class Retriever:
    """
    Hybrid retriever: the union of dense search (embedder + VectorIndex) and BM25, fused with RRF
    and optionally reranked. A reranker is any callable reranker(query, texts) -> list[float], such
    as a cross-encoder; pass None to skip that step.
    """

    def __init__(self, chunks: list[Chunk], embedder=None, use_bm25: bool = True, reranker=None):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedder = embedder or HashedTfidfEmbedder().fit(texts)
        self.index = VectorIndex(self.embedder.encode(texts))
        self.bm25 = BM25(texts) if use_bm25 else None
        self.reranker = reranker

    def retrieve(self, query: str, k: int = 5, k_candidates: int = 20) -> list[Chunk]:
        dense = self.index.search(self.embedder.encode([query])[0], k_candidates)
        if self.bm25 is not None:
            fused = reciprocal_rank_fusion(dense, self.bm25.search(query, k_candidates))
        else:
            fused = dense
        cands = fused[:k_candidates]
        if self.reranker is not None and cands:
            scores = self.reranker(query, [self.chunks[i].text for i in cands])
            cands = [i for _, i in sorted(zip(scores, cands), key=lambda t: -t[0])]
        return [self.chunks[i] for i in cands[:k]]


def build_rag_prompt(question: str, chunks: list[Chunk], instructions: str | None = None) -> str:
    """Numbered context blocks followed by citation instructions. It is boring and explicit, and that is why it works."""
    instructions = instructions or (
        "Answer ONLY from the context below. Cite sources as [n]. "
        "If the answer is not in the context, say you don't know."
    )
    ctx = "\n".join(f"[{i + 1}] ({c.doc_id}#{c.chunk_id}) {c.text}" for i, c in enumerate(chunks))
    return f"{instructions}\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"


# Query transformations
# These are cheap and often give a noticeable boost in recall.

def expand_query(query: str, llm=None, n: int = 3) -> list[str]:
    """
    Multi-query expansion: ask an LLM for a few paraphrases, then retrieve with all of them and
    merge the results with RRF. Without an LLM we simply return the original query.
    """
    if llm is None:
        return [query]
    prompt = f"Write {n} different short search queries that capture the same intent as: '{query}'. One per line."
    lines = [l.strip("-• ").strip() for l in llm.complete(prompt).splitlines() if l.strip()]
    return [query] + lines[:n]
