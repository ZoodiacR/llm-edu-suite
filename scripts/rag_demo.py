#!/usr/bin/env python
"""
RAG over a folder of .md / .txt files with LangChain: hybrid retrieval (FAISS + BM25 fused with RRF) and an
answer with numbered citations. The same pipeline as notebook 10, packaged as a command-line tool.

    python scripts/rag_demo.py --ask "Why must an MCP stdio server never print to stdout?"
    python scripts/rag_demo.py --eval data/eval/rag_eval.json            # retrieval hit rate and MRR (fast)
    python scripts/rag_demo.py --eval data/eval/rag_eval.json --ragas    # + RAGAS generation metrics (slow, uses the LLM as judge)
    python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?" --retriever dense

Models come from llm_edu_suite.config (Ollama qwen2.5:3b and nomic-embed-text by default; edit .env to switch).
The eval file is a JSON list of {"question", "answer", "sources" (or "source" / "doc"), optional "evidence"}.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import time
import types
import warnings

warnings.filterwarnings("ignore", message=".*langchain-community.*sunset")   # FAISS and BM25Retriever still live there
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from langchain_classic.retrievers import EnsembleRetriever  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

DEFAULT_DOCS = os.path.join(ROOT, "data", "knowledge_base")
DEFAULT_EVAL = os.path.join(ROOT, "data", "eval", "rag_eval.json")

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about a software course using ONLY the numbered context passages. "
               "Cite the passages you used with their numbers in square brackets, like [1] or [2][3]. "
               "Be concise: one to three sentences. If the context does not contain the answer, say \"I don't know\"."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# Indexing

def load_documents(folder: str) -> list[Document]:
    """Every top-level .md / .txt file in `folder`, with the file name in metadata['source']."""
    paths = sorted(glob.glob(os.path.join(folder, "*.md")) + glob.glob(os.path.join(folder, "*.txt")))
    return [Document(page_content=open(p, encoding="utf-8").read(), metadata={"source": os.path.basename(p)}) for p in paths]


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int | None = None) -> list[Document]:
    overlap = chunk_size // 8 if chunk_overlap is None else chunk_overlap
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True).split_documents(docs)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens for BM25 (the default splits on whitespace only, so 'Llama-3?' != 'llama-3')."""
    return re.findall(r"\w+", text.lower())


def build_faiss(chunks: list[Document], embeddings, batch: int = 64) -> FAISS:
    """Embed in batches: one huge request can overload a local embedding server, and hosted APIs cap batch size."""
    store = FAISS.from_documents(chunks[:batch], embeddings)
    for i in range(batch, len(chunks), batch):
        store.add_documents(chunks[i:i + batch])
    return store


def build_retrievers(chunks: list[Document], embeddings, k: int = 4, weights=(0.5, 0.5)) -> dict:
    """Returns {"dense", "bm25", "hybrid"} runnables that each map a question to at most k Documents."""
    dense = build_faiss(chunks, embeddings).as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)
    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=list(weights))     # weighted Reciprocal Rank Fusion
    return {"dense": dense, "bm25": bm25, "hybrid": ensemble | RunnableLambda(lambda ds: ds[:k])}


# The chain

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}" for i, d in enumerate(docs, start=1))


def make_rag_chain(retriever, llm):
    """question -> {"question", "docs", "answer"}; the documents travel with the answer so they can be cited."""
    answer = ({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
              | RAG_PROMPT | llm | StrOutputParser())
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer)


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def strip_citations(answer: str) -> str:
    """Remove [n] markers before LLM-judge scoring, so the judge does not treat them as unsupported claims."""
    return re.sub(r"\s*\[\d+\](\[\d+\])*", "", answer).strip()


# Evaluation

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).lower()


def item_sources(item: dict) -> set[str]:
    if "sources" in item:
        return set(item["sources"])
    return {item.get("source") or item["doc"]}


def retrieval_metrics(retrieve, items: list[dict], k: int = 4) -> dict:
    """
    retrieve(question) -> ranked Documents. File level: is a correct source among the top-k chunks (hit@1, hit@k, MRR)?
    Evidence level (items with an "evidence" list): does a top-k chunk contain one of the evidence strings?
    """
    def first(ds, ok):
        return next((i for i, d in enumerate(ds, start=1) if ok(d)), None)

    file_r, ev_r = [], []
    for it in items:
        ds = list(retrieve(it["question"]))[:k]
        srcs = item_sources(it)
        file_r.append(first(ds, lambda d: d.metadata.get("source") in srcs))
        if it.get("evidence"):
            ev_r.append(first(ds, lambda d: any(_norm(e) in _norm(d.page_content) for e in it["evidence"])))

    def summary(ranks, prefix):
        n = max(1, len(ranks))
        return {f"{prefix}hit@1": sum(r == 1 for r in ranks) / n, f"{prefix}hit@{k}": sum(r is not None for r in ranks) / n,
                f"{prefix}mrr": sum(1 / r for r in ranks if r) / n}
    out = summary(file_r, "file_")
    if ev_r:
        out.update(summary(ev_r, "evidence_"))
    return out


def _ragas_compat_shim() -> None:
    """ragas 0.4.3 imports langchain_community.chat_models.vertexai, which langchain-community 0.4 removed."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[stub.__name__] = stub


async def ragas_evaluate(items: list[dict], results: list[dict], chat_model: str, embedding_model: str,
                         base_url: str | None = None, concurrency: int = 4) -> dict:
    """Mean faithfulness, answer relevancy, context precision and context recall, judged through Ollama's OpenAI-compatible API."""
    _ragas_compat_shim()
    import math

    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
    client = AsyncOpenAI(base_url=base_url, api_key="ollama", timeout=180, max_retries=1)          # Ollama ignores the key, the client requires one
    judge = llm_factory(chat_model, client=client, temperature=0)
    metrics = {"faithfulness": Faithfulness(llm=judge),
               "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=OpenAIEmbeddings(client=client, model=embedding_model), strictness=1),
               "context_precision": ContextPrecision(llm=judge), "context_recall": ContextRecall(llm=judge)}
    sem = asyncio.Semaphore(concurrency)

    async def score(name, it, res):
        ctx, ans = [d.page_content for d in res["docs"]], strip_citations(res["answer"])
        kw = {"faithfulness": dict(user_input=it["question"], response=ans, retrieved_contexts=ctx),
              "answer_relevancy": dict(user_input=it["question"], response=ans),
              "context_precision": dict(user_input=it["question"], reference=it["answer"], retrieved_contexts=ctx),
              "context_recall": dict(user_input=it["question"], reference=it["answer"], retrieved_contexts=ctx)}[name]
        async with sem:
            try:
                return name, (await metrics[name].ascore(**kw)).value
            except Exception:
                return name, float("nan")

    scores = await asyncio.gather(*(score(n, it, r) for it, r in zip(items, results) for n in metrics))
    out = {}
    for n in metrics:
        vals = [v for name, v in scores if name == n and not math.isnan(v)]
        out[n] = sum(vals) / len(vals) if vals else float("nan")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", default=DEFAULT_DOCS, help="folder with .md / .txt files (default: data/knowledge_base)")
    p.add_argument("--ask", default=None, help="a question to answer with citations")
    p.add_argument("--eval", nargs="?", const=DEFAULT_EVAL, default=None, help="eval JSON (default when given without a value: data/eval/rag_eval.json)")
    p.add_argument("--ragas", action="store_true", help="with --eval: also generate answers and score them with RAGAS")
    p.add_argument("--retriever", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--k", type=int, default=4, help="chunks per answer")
    p.add_argument("--chunk-size", type=int, default=800, help="characters per chunk (overlap = chunk_size // 8)")
    args = p.parse_args(argv)

    from llm_edu_suite.config import get_embeddings, get_llm, setup
    info = setup(project="llm-edu-suite-rag-demo", quiet=True)
    if info["ollama_running"] is False:
        sys.exit("Ollama is not running: start it and `ollama pull qwen2.5:3b && ollama pull nomic-embed-text`.")

    t0 = time.time()
    docs = load_documents(args.docs)
    chunks = split_documents(docs, args.chunk_size)
    retrievers = build_retrievers(chunks, get_embeddings(), k=args.k)
    print(f"indexed {len(docs)} files -> {len(chunks)} chunks in {time.time() - t0:.1f}s | retriever {args.retriever} | llm {info['chat_model']}")
    chain = make_rag_chain(retrievers[args.retriever], get_llm())

    if args.ask:
        res = chain.invoke(args.ask)
        print("\nANSWER:", res["answer"], "\n\nSOURCES:")
        cited = cited_numbers(res["answer"])
        for i, d in enumerate(res["docs"], start=1):
            print(f"  [{i}]{'*' if i in cited else ' '} {d.metadata['source']}: {d.page_content[:100]!r}")

    if args.eval:
        items = json.load(open(args.eval, encoding="utf-8"))
        print(f"\nretrieval on {len(items)} questions (k={args.k}):")
        for name, r in retrievers.items():
            m = retrieval_metrics(r.invoke, items, k=args.k)
            print(f"  {name:7s} " + "  ".join(f"{key} {v:.2f}" for key, v in m.items()))
        if args.ragas:
            t0 = time.time()
            results = chain.batch([it["question"] for it in items], config={"max_concurrency": 4})
            scores = asyncio.run(ragas_evaluate(items, results, info["chat_model"], info["embedding_model"]))
            print(f"\nRAGAS ({args.retriever}, judge {info['chat_model']}, {time.time() - t0:.0f}s): "
                  + "  ".join(f"{k} {v:.2f}" for k, v in scores.items()))


if __name__ == "__main__":
    main()
