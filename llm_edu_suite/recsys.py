"""
Recommender systems with LLM-era techniques.

A modern recommendation stack has two stages: *retrieval*, which narrows millions of items down
to a few hundred, and *ranking*, which orders those candidates. More and more, both stages use the
same machinery as language models:

1. Sequential recommendation is next-token prediction over item ids (SASRec, 2018). The same idea
   now powers production systems at Meta, Netflix and Pinterest scale (for example HSTU). We reuse
   `TransformerLM` unchanged; its "vocabulary" is simply the item catalog.
2. Embedding retrieval represents items and users as vectors and takes the top-k by cosine. The
   vectors can come from a two-tower model or, as here, from the item embeddings the sequential
   model learns. Content embeddings of item descriptions solve cold start for items that nobody
   has interacted with yet.
3. LLM re-ranking and explanation: an LLM reads the user's history and the candidate list, then
   re-orders or explains them. That is useful at the top of the funnel (tens of candidates) but far
   too slow for retrieval. Notebook 23 builds this stage with LangChain (see its section 4); this
   module only holds the data, the SASRec training and the embedding rankers.

This module also has synthetic interaction data with hidden taste clusters, a popularity baseline
and the leave-one-out split. The hit@k and nDCG@k metrics live in evaluation.py.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# Synthetic data with structure
_GENRES = ["sci-fi", "romance", "thriller", "documentary", "comedy", "fantasy", "drama", "horror"]
_ADJ = ["dark", "funny", "epic", "quiet", "fast-paced", "emotional", "clever", "classic"]
_NOUN = ["journey", "heist", "family", "mystery", "voyage", "rivalry", "secret", "kingdom"]


@dataclass
class Catalog:
    n_items: int
    titles: list[str]
    descriptions: list[str]
    genres: list[str]
    item_cluster: np.ndarray            # the hidden taste cluster of each item; the model never sees it


def make_catalog(n_items: int = 200, n_clusters: int = 8, seed: int = 0) -> Catalog:
    rng = random.Random(seed)
    clusters = np.array([i % n_clusters for i in range(n_items)])
    titles, descs, genres = [], [], []
    for i in range(n_items):
        g = _GENRES[clusters[i] % len(_GENRES)]
        adj, noun = rng.choice(_ADJ), rng.choice(_NOUN)
        titles.append(f"{adj.title()} {noun.title()} {i}")
        descs.append(f"A {adj} {g} {noun} about {rng.choice(_NOUN)} and {rng.choice(_NOUN)}.")
        genres.append(g)
    return Catalog(n_items, titles, descs, genres, clusters)


def make_interactions(catalog: Catalog, n_users: int = 500, min_len: int = 8, max_len: int = 25,
                      stickiness: float = 0.75, seed: int = 0) -> list[list[int]]:
    """
    Every user has one or two favourite clusters. Each next item comes from one of them with
    probability `stickiness` and from a random cluster otherwise. Within a cluster, items are picked
    with a Zipf popularity skew, so the popularity baseline is genuinely hard to beat. Item ids run
    from 1 to n_items because 0 is reserved for padding.
    """
    rng = np.random.default_rng(seed)
    n_clusters = int(catalog.item_cluster.max()) + 1
    by_cluster = [np.where(catalog.item_cluster == c)[0] for c in range(n_clusters)]
    pop = 1.0 / np.arange(1, catalog.n_items + 1) ** 0.8         # Zipf-like popularity: the lower the index, the more popular
    sessions = []
    for _ in range(n_users):
        favs = rng.choice(n_clusters, size=rng.integers(1, 3), replace=False)
        L = int(rng.integers(min_len, max_len + 1))
        seq = []
        for _ in range(L):
            c = rng.choice(favs) if rng.random() < stickiness else rng.integers(n_clusters)
            items = by_cluster[c]
            p = pop[items] / pop[items].sum()
            seq.append(int(rng.choice(items, p=p)) + 1)          # shift by one so id 0 stays free for padding
        sessions.append(seq)
    return sessions


def leave_one_out_split(sessions: list[list[int]]):
    """Train on everything except each user's last item, which becomes the test target (the standard SASRec protocol)."""
    train = [s[:-1] for s in sessions]
    test = [(s[:-1], s[-1]) for s in sessions]
    return train, test


# Baselines

def popularity_ranker(train_sessions: list[list[int]], n_items: int):
    counts = np.zeros(n_items + 1)
    for s in train_sessions:
        for i in s:
            counts[i] += 1
    order = np.argsort(-counts)
    def rank(history: list[int], k: int = 10) -> list[int]:
        seen = set(history)
        return [int(i) for i in order if i != 0 and int(i) not in seen][:k]
    return rank


# Sequential recommender (SASRec-style) on top of TransformerLM

def sessions_to_tensor(sessions: list[list[int]], max_len: int, pad: int = 0):
    """Left-pad or truncate sessions to shape (N, max_len). Padding on the left keeps the most recent items at the end."""
    out = torch.full((len(sessions), max_len), pad, dtype=torch.long)
    for i, s in enumerate(sessions):
        s = s[-max_len:]
        out[i, max_len - len(s):] = torch.tensor(s)
    return out


def train_sasrec(model, train_sessions: list[list[int]], max_len: int = 32, steps: int = 400, batch_size: int = 64,
                 lr: float = 2e-3, device: str = "cpu", log_every: int = 100, seed: int = 0):
    """
    Next-item prediction: the input is items[:-1], the target is items[1:], and padding targets are
    set to -100 so the loss ignores them. This is exactly the language-modelling objective, with a
    TransformerLM whose vocab_size is n_items + 1.
    """
    rng = random.Random(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    for step in range(steps):
        batch = rng.sample(train_sessions, min(batch_size, len(train_sessions)))
        seqs = sessions_to_tensor(batch, max_len + 1).to(device)
        x, y = seqs[:, :-1], seqs[:, 1:].clone()
        y[y == 0] = -100                                             # never learn to predict padding
        y[x == 0] = -100                                             # and no target where the input itself is padding
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"sasrec step {step:4d}  loss {loss.item():.3f}")
    return losses


def sasrec_rank(model, history: list[int], k: int = 10, max_len: int = 32, device: str = "cpu", exclude_seen: bool = True) -> list[int]:
    """Score every item with the logits at the last position and return the top-k items the user has not seen."""
    model.eval()
    x = sessions_to_tensor([history], max_len).to(device)
    with torch.no_grad():
        logits, _ = model(x)
    scores = logits[0, -1].clone()
    scores[0] = -float("inf")                                        # padding is not a real item
    if exclude_seen:
        scores[torch.tensor(history, device=device)] = -float("inf")
    return scores.topk(k).indices.tolist()


# Embedding retrieval: collaborative vectors, plus content vectors for cold start

def item_embeddings_from_model(model) -> np.ndarray:
    """The input embeddings learned by the sequential model double as item vectors that carry the collaborative signal."""
    E = model.tok_emb.weight.detach().cpu().float().numpy()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E


def content_embeddings(catalog: Catalog, embedder=None) -> np.ndarray:
    """
    Item vectors built from the descriptions, so they also work for brand-new items (cold start). Row 0 is all
    zeros for padding. By default this is a plain TF-IDF over words (enough for our synthetic catalogue); pass any
    LangChain embeddings object (e.g. llm_edu_suite.config.get_embeddings()) to use a neural embedding model instead.
    """
    texts = [f"{t}. {d}" for t, d in zip(catalog.titles, catalog.descriptions)]
    if embedder is not None:
        E = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    else:
        docs = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
        vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        df = np.zeros(len(vocab))
        for d in docs:
            for w in set(d):
                df[vocab[w]] += 1
        idf = np.log((1 + len(docs)) / (1 + df)) + 1.0          # smoothed inverse document frequency
        E = np.zeros((len(docs), len(vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            for w in d:
                E[i, vocab[w]] += idf[vocab[w]]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)  # unit vectors, so a dot product is a cosine
    return np.vstack([np.zeros((1, E.shape[1]), dtype=np.float32), E])


def embedding_rank(E: np.ndarray, history: list[int], k: int = 10, recency_decay: float = 0.8) -> list[int]:
    """The user vector is a recency-weighted mean of the history's item vectors. We rank items by cosine to it and skip the ones already seen."""
    w = np.array([recency_decay ** (len(history) - 1 - i) for i in range(len(history))], dtype=np.float32)
    u = (E[history] * w[:, None]).sum(0)
    u = u / (np.linalg.norm(u) + 1e-8)
    scores = E @ u
    scores[0] = -np.inf
    scores[history] = -np.inf
    return np.argsort(-scores)[:k].tolist()


# MovieLens-1M: real data for the v3 recommender (notebook 23)
#
# MovieLens is published by GroupLens (University of Minnesota) for research use; it may not be
# redistributed, so it is downloaded on first use into data/cache/ (git-ignored) instead of being
# committed. Cite: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
# Context. ACM TiiS 5(4).

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(cache_dir: str | None = None) -> str:
    """Download and unzip MovieLens-1M once; return the folder with movies.dat, ratings.dat and users.dat."""
    import os
    import urllib.request
    import zipfile
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
    folder = os.path.join(cache_dir, "ml-1m")
    if not os.path.exists(os.path.join(folder, "ratings.dat")):
        os.makedirs(cache_dir, exist_ok=True)
        zpath = os.path.join(cache_dir, "ml-1m.zip")
        if not os.path.exists(zpath):
            print("downloading MovieLens-1M (6 MB) from grouplens.org ...")
            urllib.request.urlretrieve(ML1M_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(cache_dir)
    return folder


@dataclass
class MovieLens:
    """MovieLens-1M re-indexed so that item ids run 1..n_items (0 is padding, as in a language model's vocabulary)."""
    titles: list[str]                   # index = item id; titles[0] == ""
    genres: list[list[str]]
    years: np.ndarray                   # release year per item id (0 for padding)
    genre_names: list[str]
    events: dict                        # user id -> list of (timestamp, item id, rating), in time order
    demographics: dict                  # user id -> (gender, age code, occupation code)

    @property
    def n_items(self) -> int:
        return len(self.titles) - 1


def load_movielens_1m(folder: str) -> MovieLens:
    """Parse the three .dat files. Ties in timestamp keep the file order, so the sequence is deterministic."""
    import os
    raw = {}
    for line in open(os.path.join(folder, "movies.dat"), encoding="latin-1"):
        mid, title, genres = line.rstrip("\n").split("::")
        raw[int(mid)] = (title, genres.split("|"))
    rows = [tuple(map(int, line.split("::"))) for line in open(os.path.join(folder, "ratings.dat"), encoding="latin-1")]
    rated = sorted({m for _, m, _, _ in rows})
    idx = {m: i + 1 for i, m in enumerate(rated)}         # only movies that have at least one rating
    titles, genres, years = [""], [[]], [0]
    for m in rated:
        t, g = raw[m]
        titles.append(t); genres.append(g)
        y = re.search(r"\((\d{4})\)\s*$", t)
        years.append(int(y.group(1)) if y else 0)
    events: dict = {}
    for k, (u, m, r, t) in enumerate(rows):
        events.setdefault(u, []).append((t, k, idx[m], r))
    events = {u: [(t, i, r) for t, _, i, r in sorted(v)] for u, v in events.items()}
    demo = {}
    for line in open(os.path.join(folder, "users.dat"), encoding="latin-1"):
        u, g, a, o, _ = line.strip().split("::")
        demo[int(u)] = (g, int(a), int(o))
    names = sorted({g for gs in genres for g in gs})
    return MovieLens(titles, genres, np.array(years), names, events, demo)


def temporal_split(data: MovieLens, cold_frac: float = 0.1, min_count: int = 20, seed: int = 0) -> dict:
    """
    Leave-last-two split in time order, the standard protocol for sequential recommenders: for every user the
    last event is the test target, the one before it the validation target, and everything earlier is training
    history. A fraction of items (among those with at least `min_count` ratings, so they do appear as targets)
    is held out as *cold*: removed from every history and every training sequence, as if they were new releases
    the system has never seen anybody watch. They can still be targets, which is what the cold segment measures.
    """
    users = sorted(data.events)
    seqs = {u: [i for _, i, _ in data.events[u]] for u in users}
    counts = np.bincount([i for s in seqs.values() for i in s], minlength=data.n_items + 1)
    eligible = np.where(counts >= min_count)[0]
    eligible = eligible[eligible > 0]
    rng = np.random.default_rng(seed)
    cold = set(rng.choice(eligible, size=int(cold_frac * data.n_items), replace=False).tolist())
    clean = lambda s: [i for i in s if i not in cold]
    return dict(
        users=users, cold=cold,
        train=[clean(seqs[u][:-2]) for u in users],
        valid=[(clean(seqs[u][:-2]), seqs[u][-2]) for u in users],
        test=[(clean(seqs[u][:-1]), seqs[u][-1]) for u in users],
        valid_time=[data.events[u][-2][0] for u in users],
        test_time=[data.events[u][-1][0] for u in users],
    )


def target_ranks(scores, histories: list[list[int]], targets: list[int]):
    """
    Full-ranking evaluation: the rank (0 = first) of each user's target among *all* items, after removing the
    padding id and the items the user has already seen. `scores` is a (users, n_items + 1) tensor. Ranking
    against every item, instead of against 100 sampled negatives, is the honest protocol (Krichene & Rendle, 2020).
    """
    s = scores.clone().float()
    s[:, 0] = -float("inf")
    for j, h in enumerate(histories):
        if h:
            s[j, torch.as_tensor(h, device=s.device)] = -float("inf")
    tgt = torch.as_tensor(targets, device=s.device)
    return (s > s.gather(1, tgt[:, None])).sum(1)


def rank_metrics(ranks, k: int = 10) -> dict:
    """Per-user hit@k, nDCG@k and reciprocal rank from target ranks (arrays, so they can be bootstrapped or segmented)."""
    r = np.asarray(ranks.cpu() if hasattr(ranks, "cpu") else ranks, dtype=float)
    return {f"hit@{k}": (r < k).astype(float),
            f"ndcg@{k}": np.where(r < k, 1.0 / np.log2(r + 2), 0.0),
            "rr": 1.0 / (r + 1)}
