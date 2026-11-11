"""
Tests for the LangChain RAG helpers in scripts/rag_demo.py and for the labelled eval set used by notebook 18.

The fast tests use LangChain's fake models (no Ollama). The tests marked `ollama` run a tiny end-to-end query
against the local models and are skipped when no Ollama server answers.
"""
import json
import os
import sys

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import rag_demo  # noqa: E402
from llm_edu_suite.config import ollama_available  # noqa: E402

DOCS = [
    Document(page_content="Refunds are accepted within 30 days of purchase with a receipt.", metadata={"source": "refunds.md"}),
    Document(page_content="Express shipping delivers in 1 to 2 business days; standard takes 5.", metadata={"source": "shipping.md"}),
    Document(page_content="Set the flag --chunk-size to change the window; the overlap is chunk_size // 4.", metadata={"source": "cli.md"}),
    Document(page_content="Customer data is encrypted at rest with AES-256 and in transit with TLS 1.3.", metadata={"source": "security.md"}),
]


@pytest.fixture(scope="module")
def retrievers():
    return rag_demo.build_retrievers(DOCS, DeterministicFakeEmbedding(size=32), k=2)


def test_tokenize_lowercases_and_strips_punctuation():
    assert rag_demo.tokenize("Llama-3-8B? KV-cache!") == ["llama", "3", "8b", "kv", "cache"]


def test_split_documents_keeps_source_and_respects_size():
    long_doc = Document(page_content=("word " * 400).strip(), metadata={"source": "long.md"})
    chunks = rag_demo.split_documents([long_doc], chunk_size=200)
    assert len(chunks) > 5
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "long.md" and "start_index" in c.metadata for c in chunks)


def test_load_documents_reads_the_knowledge_base():
    docs = rag_demo.load_documents(rag_demo.DEFAULT_DOCS)
    assert {d.metadata["source"] for d in docs} >= {"glossary.md", "installation.md", "cli_scripts.md"}


def test_format_docs_numbers_passages_with_sources():
    text = rag_demo.format_docs(DOCS[:2])
    assert text.startswith("[1] (source: refunds.md)")
    assert "[2] (source: shipping.md)" in text


def test_citation_helpers():
    answer = "Refunds take 30 days [1][3]. Shipping is fast [2]."
    assert rag_demo.cited_numbers(answer) == [1, 2, 3]
    assert rag_demo.strip_citations(answer) == "Refunds take 30 days. Shipping is fast."


def test_bm25_finds_exact_strings(retrievers):
    top = retrievers["bm25"].invoke("what is chunk_size // 4?")
    assert top[0].metadata["source"] == "cli.md"


def test_hybrid_returns_at_most_k_unique_chunks(retrievers):
    out = retrievers["hybrid"].invoke("encrypted with AES-256")
    assert 1 <= len(out) <= 2
    assert len({d.page_content for d in out}) == len(out)
    # BM25 ranks it first, so RRF keeps it in the top k whatever the (random) fake embedding ranks first
    assert "security.md" in {d.metadata["source"] for d in out}


def test_rag_chain_with_fake_llm_returns_answer_and_docs(retrievers):
    llm = FakeListChatModel(responses=["Within 30 days [1]."])
    res = rag_demo.make_rag_chain(retrievers["bm25"], llm).invoke("How many days for a refund?")
    assert res["answer"] == "Within 30 days [1]."
    assert res["question"] == "How many days for a refund?"
    assert res["docs"][0].metadata["source"] == "refunds.md"


def test_retrieval_metrics_file_and_evidence_levels():
    def fake_retrieve(q):
        return [DOCS[1], DOCS[0]] if "refund" in q else [DOCS[3]]
    items = [
        {"question": "refund window?", "sources": ["refunds.md"], "evidence": ["within 30 days"]},
        {"question": "encryption?", "source": "security.md", "evidence": ["AES-256"]},
        {"question": "flags?", "doc": "cli.md", "evidence": ["chunk_size // 4"]},
    ]
    m = rag_demo.retrieval_metrics(fake_retrieve, items, k=2)
    assert m["file_hit@1"] == pytest.approx(1 / 3)
    assert m["file_hit@2"] == pytest.approx(2 / 3)
    assert m["file_mrr"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert m["evidence_mrr"] == pytest.approx((0.5 + 1 + 0) / 3)


def test_eval_set_is_well_formed_and_evidence_is_in_the_sources():
    items = json.load(open(rag_demo.DEFAULT_EVAL, encoding="utf-8"))
    assert 15 <= len(items) <= 30
    kb = {d.metadata["source"]: rag_demo._norm(d.page_content) for d in rag_demo.load_documents(rag_demo.DEFAULT_DOCS)}
    for it in items:
        assert it["question"] and it["answer"] and it["source"] in it["sources"]
        assert set(it["sources"]) <= set(kb), it["question"]
        # every evidence string must appear verbatim in at least one listed source, and the primary source must contain one
        assert any(rag_demo._norm(e) in kb[it["source"]] for e in it["evidence"]), it["question"]
        for e in it["evidence"]:
            assert any(rag_demo._norm(e) in kb[s] for s in it["sources"]), (it["question"], e)


ollama = pytest.mark.skipif(not ollama_available(), reason="needs a running Ollama server")


@pytest.mark.ollama
@ollama
def test_end_to_end_rag_query_with_ollama():
    from llm_edu_suite.config import get_embeddings, get_llm
    r = rag_demo.build_retrievers(DOCS, get_embeddings(), k=2)
    res = rag_demo.make_rag_chain(r["hybrid"], get_llm()).invoke("How long does express shipping take?")
    assert any(d.metadata["source"] == "shipping.md" for d in res["docs"])
    assert "1" in res["answer"] and "2" in res["answer"]
