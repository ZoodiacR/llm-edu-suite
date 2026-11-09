"""Fast tests for the parts of the package that need neither PyTorch nor a model server: tokenizer, metrics, recsys data."""
import math

import numpy as np
import pytest

from llm_edu_suite.evaluation import (accuracy, confusion_counts, context_support, evaluate_ranking, mrr, ndcg_at_k,
                                      recall_at_k, rouge_l, token_f1)
from llm_edu_suite.recsys import (content_embeddings, embedding_rank, leave_one_out_split, make_catalog, make_interactions,
                                  popularity_ranker)
from llm_edu_suite.tokenizer import ByteBPETokenizer, compression_ratio


# Tokenizer

def test_bpe_roundtrip_and_specials(tmp_path):
    text = "the model predicts the next token. the tokenizer splits the text. " * 30
    tok = ByteBPETokenizer().train(text, vocab_size=300)
    for s in ["the model", "unseen ✓ Zyx 🦙 def f(x): return x", ""]:
        assert tok.decode(tok.encode(s)) == s
    assert 256 < tok.vocab_size <= 300 and tok.eos_id != tok.bos_id   # a small corpus may run out of pairs early
    assert tok.encode("a", add_bos=True, add_eos=True)[0] == tok.bos_id
    assert compression_ratio(tok, "the model predicts the next token") > 1.5
    p = tmp_path / "t.json"; tok.save(str(p)); tok2 = ByteBPETokenizer.load(str(p))
    assert tok2.encode(text[:50]) == tok.encode(text[:50]) and tok2.special == tok.special


# Evaluation metrics

def test_text_and_ranking_metrics():
    assert token_f1("a b c", "a b c") == 1.0 and token_f1("x", "y") == 0.0
    assert 0 < rouge_l("a b c d", "a b x d") < 1
    assert recall_at_k([1, 2, 3], {2, 9}, 3) == 0.5
    assert mrr([5, 2], {2}) == 0.5 and mrr([1], {2}) == 0.0
    assert ndcg_at_k([2], {2}, 1) == 1.0 and ndcg_at_k([1, 2], {2}, 2) == pytest.approx(1 / math.log2(3))
    m = evaluate_ranking([([1, 2], {1}), ([3, 4], {9})], ks=(1,))
    assert m["hit@1"] == 0.5 and m["mrr"] == 0.5


def test_context_support_and_classification_metrics():
    assert context_support("express shipping takes two days", "Express shipping takes two business days.") == 1.0
    assert context_support("bananas are purple", "Express shipping takes two days.") == 0.0
    assert accuracy(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(2 / 3)
    assert confusion_counts(["a", "b"], ["a", "a"])[("a", "b")] == 1


# Recommender systems (the parts that don't need torch)

def test_recsys_data_and_baselines():
    cat = make_catalog(40, n_clusters=4)
    sessions = make_interactions(cat, n_users=50, min_len=5, max_len=8)
    assert all(1 <= i <= 40 for s in sessions for i in s) and all(5 <= len(s) <= 8 for s in sessions)
    train, test = leave_one_out_split(sessions)
    assert all(len(t) == len(s) - 1 for t, s in zip(train, sessions)) and test[0][1] == sessions[0][-1]
    pop = popularity_ranker(train, cat.n_items)
    top = pop(train[0], 5)
    assert len(top) == 5 and not set(top) & set(train[0]) and 0 not in top
    E = content_embeddings(cat)
    assert E.shape[0] == cat.n_items + 1 and np.allclose(E[0], 0)
    assert np.allclose(np.linalg.norm(E[1:], axis=1), 1.0, atol=1e-5)      # unit vectors: dot product is cosine
    rec = embedding_rank(E, train[0], 5)
    assert len(rec) == 5 and not set(rec) & set(train[0]) and 0 not in rec
    m = evaluate_ranking([(pop(h, 10), {t}) for h, t in test], ks=(10,))
    assert 0.0 <= m["hit@10"] <= 1.0


def test_content_embeddings_accept_a_langchain_embedder():
    class FakeEmbeddings:                     # the only method content_embeddings needs from a LangChain embeddings object
        def embed_documents(self, texts):
            return [[float(len(t)), 1.0] for t in texts]
    cat = make_catalog(10, n_clusters=2)
    E = content_embeddings(cat, embedder=FakeEmbeddings())
    assert E.shape == (11, 2) and np.allclose(E[0], 0)


# Uncertainty and configuration

def test_bootstrap_intervals():
    from llm_edu_suite.evaluation import bootstrap_ci, paired_bootstrap_diff
    m, lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert m == pytest.approx(0.5) and lo < 0.5 < hi and hi - lo < 0.25          # 100 coin flips: about +-0.1
    m, lo, hi = bootstrap_ci([1.0, float("nan"), 1.0])                          # NaNs (failed judge calls) are dropped
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    d, lo, hi = paired_bootstrap_diff([0.0] * 30, [1.0] * 30)
    assert d == 1.0 and lo == 1.0                                               # every pair improved: the interval excludes 0
    d, lo, hi = paired_bootstrap_diff([0, 1] * 20, [1, 0] * 20)
    assert lo < 0 < hi                                                          # same average, swapped wins: no clear difference


def test_judge_model_falls_back_to_the_chat_model(monkeypatch):
    from llm_edu_suite import config
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(config, "ollama_models", lambda timeout=2.0: ["qwen2.5:3b"])
    assert config.judge_model() == "qwen2.5:3b"                                  # the 7B is not pulled
    monkeypatch.setattr(config, "ollama_models", lambda timeout=2.0: ["qwen2.5:3b", "qwen2.5:7b"])
    assert config.judge_model() == "qwen2.5:7b"
    monkeypatch.setenv("JUDGE_MODEL", "my-judge")
    assert config.judge_model() == "my-judge"


def test_movielens_loader_and_temporal_split(tmp_path):
    """A tiny file in MovieLens-1M format: loading, re-indexing, time order, leave-last-two and the cold holdout."""
    from llm_edu_suite.recsys import load_movielens_1m, temporal_split
    (tmp_path / "movies.dat").write_text("10::Toy Story (1995)::Animation|Children's\n20::Heat (1995)::Action|Crime\n"
                                         "30::Alien (1979)::Horror|Sci-Fi\n40::Unrated (2000)::Drama\n", encoding="latin-1")
    rows = []
    for u in (1, 2, 3):
        for t, m in enumerate((10, 20, 30, 10, 20) if u != 3 else (30, 20, 10, 30, 20)):
            rows.append(f"{u}::{m}::{4}::{100 * t + u}")
    (tmp_path / "ratings.dat").write_text("\n".join(rows) + "\n", encoding="latin-1")
    (tmp_path / "users.dat").write_text("1::F::1::10::0\n2::M::25::3::0\n3::F::35::7::0\n", encoding="latin-1")
    ml = load_movielens_1m(str(tmp_path))
    assert ml.n_items == 3 and ml.titles[1] == "Toy Story (1995)" and ml.years[3] == 1979   # the unrated movie is dropped
    assert [i for _, i, _ in ml.events[1]] == [1, 2, 3, 1, 2]
    sp = temporal_split(ml, cold_frac=0.34, min_count=1, seed=0)
    (cold,) = sp["cold"]
    assert all(cold not in h for h in sp["train"]) and all(cold not in h for h, _ in sp["test"])
    assert [t for _, t in sp["test"]] == [2, 2, 2] and [t for _, t in sp["valid"]] == [1, 1, 3]
