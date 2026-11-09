"""
Evaluation toolkit. Every notebook in the suite ends with an evaluation section built on these helpers.

Language-model metrics
    perplexity                exp(mean token NLL) on held-out text
    exact_match_rate          share of prompts answered exactly right (toy tasks, math, extraction)
Text-overlap metrics (summarization, QA)
    token_f1, rouge_l
Retrieval / ranking metrics (RAG, recommenders)
    recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate_at_k, evaluate_ranking
RAG-specific
    context_support            fraction of answer tokens found in the retrieved context (a faithfulness proxy)
LLM-as-judge
    done with the frameworks: RAGAS metrics (notebook 10) and LangSmith evaluators (notebook 15)
Systems
    benchmark_generation       tokens/s, time to first token and peak memory of model.generate
Classification
    accuracy, confusion_counts
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

import numpy as np

try:  # the pure-Python metrics work fine without torch, so it is optional
    import torch
except ImportError:  # pragma: no cover
    torch = None


# Language-model metrics

def perplexity(model, data, block_size: int = 64, batch_size: int = 32, n_batches: int = 20, device: str = "cpu") -> float:
    """Perplexity: exp of the mean next-token NLL over random windows of `data` (a 1-D token tensor)."""
    from .data import get_batch

    model.eval()
    nll = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            nll += loss.item()
    return math.exp(nll / n_batches)


def exact_match_rate(predict_fn, cases) -> float:
    """Share of `cases` (an iterable of (input, expected) pairs) where predict_fn(input) == expected."""
    hits = total = 0
    for inp, expected in cases:
        hits += predict_fn(inp) == expected
        total += 1
    return hits / max(1, total)


# Text-overlap metrics

def _toks(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def token_f1(pred: str, ref: str) -> float:
    """SQuAD-style token F1 between a prediction and a reference."""
    p, r = Counter(_toks(pred)), Counter(_toks(ref))
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure based on the longest common subsequence of tokens."""
    a, b = _toks(pred), _toks(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(a), lcs / len(b)
    return 2 * prec * rec / (prec + rec)


# Retrieval and ranking metrics
# All of these take a ranked list of item ids and the set of ids that are actually relevant.

def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / max(1, len(relevant))


def precision_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def hit_rate_at_k(ranked: list, relevant: set, k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr(ranked: list, relevant: set) -> float:
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    """Binary-relevance nDCG: rewards relevant items near the top with a log discount."""
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_ranking(queries: list[tuple[list, set]], ks=(1, 5, 10)) -> dict:
    """Average recall@k, hit@k, nDCG@k and MRR over `queries`, a list of (ranked_items, relevant_set)."""
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(r, rel, k) for r, rel in queries]))
        out[f"hit@{k}"] = float(np.mean([hit_rate_at_k(r, rel, k) for r, rel in queries]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(r, rel, k) for r, rel in queries]))
    out["mrr"] = float(np.mean([mrr(r, rel) for r, rel in queries]))
    return out


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """
    Mean and a (1 - alpha) percentile-bootstrap confidence interval: resample the per-example scores
    with replacement, recompute the mean, and take the alpha/2 and 1 - alpha/2 quantiles. With 20
    examples the interval is wide, which is the honest answer to "is 0.54 really better than 0.38?".
    NaNs (for example, a judge call that failed) are dropped first.
    """
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """
    Mean difference b - a over paired examples (same questions or users, two systems) with a bootstrap CI.
    If the interval excludes 0, the difference is unlikely to be noise. Pairs with a NaN are dropped.
    """
    pairs = [(x, y) for x, y in zip(a, b) if not (np.isnan(x) or np.isnan(y))]
    d = np.array([y - x for x, y in pairs], dtype=float)
    return bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)


# RAG-specific metrics

def context_support(answer: str, context: str, min_len: int = 4) -> float:
    """
    A faithfulness proxy: the fraction of answer tokens that also appear in the retrieved context.
    Tokens shorter than `min_len` are skipped, which roughly filters out stopwords. It is cheap and
    crude; an LLM judge with a rubric is the real thing.
    """
    ctx = set(_toks(context))
    toks = [t for t in _toks(answer) if len(t) >= min_len]
    return sum(t in ctx for t in toks) / max(1, len(toks))


# LLM-as-judge evaluation lives in the frameworks now: RAGAS (notebook 10), LangSmith evaluators (notebook 15).

# Systems benchmarks

def benchmark_generation(model, prompt_ids, max_new_tokens: int = 64, n_runs: int = 3, **gen_kwargs) -> dict:
    """Measure time to first token and decode tokens/s for model.generate (synchronizing CUDA when it is used)."""
    device = prompt_ids.device
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    ttfts, rates = [], []
    for _ in range(n_runs):
        sync(); t0 = time.perf_counter()
        model.generate(prompt_ids, 1, **gen_kwargs)                  # prefill plus one token approximates TTFT
        sync(); ttfts.append(time.perf_counter() - t0)
        sync(); t0 = time.perf_counter()
        out = model.generate(prompt_ids, max_new_tokens, **gen_kwargs)
        sync(); dt = time.perf_counter() - t0
        rates.append((out.shape[1] - prompt_ids.shape[1]) * prompt_ids.shape[0] / dt)
    res = {"ttft_ms": 1000 * float(np.median(ttfts)), "tokens_per_s": float(np.median(rates)), "batch": prompt_ids.shape[0]}
    if device.type == "cuda":
        res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return res


# Classification

def accuracy(preds, labels) -> float:
    return sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))


def confusion_counts(preds, labels) -> dict:
    return dict(Counter(zip(labels, preds)))
