#!/usr/bin/env python
"""
Index a folder of .txt/.md files and answer questions with hybrid retrieval + an LLM (MockLLM offline, or any OpenAI-compatible server).

python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?"
# needs `pip install requests`
python scripts/rag_demo.py --docs data/sample_docs --ask "..." --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json      # eval.json: [{"question":..., "doc":..., "answer":...}, ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_edu_suite.evaluation import context_support, evaluate_ranking, token_f1
from llm_edu_suite.llm_client import MockLLM, OpenAICompatibleClient
from llm_edu_suite.rag import Retriever, build_rag_prompt, chunk_documents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True, help="folder with .txt / .md files")
    p.add_argument("--ask", default=None)
    p.add_argument("--eval", default=None, help="json list of {question, doc, answer}")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--model", default=None, help="model name on an OpenAI-compatible server (omit for MockLLM)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.docs, "*.txt")) + glob.glob(os.path.join(args.docs, "*.md")))
    docs = {os.path.basename(f): open(f, encoding="utf-8").read() for f in files}
    chunks = chunk_documents(docs, size=args.chunk_size, overlap=args.chunk_size // 4)
    retriever = Retriever(chunks)
    llm = OpenAICompatibleClient(args.model, base_url=args.base_url) if args.model else MockLLM()
    print(f"indexed {len(docs)} docs -> {len(chunks)} chunks | llm = {type(llm).__name__}")

    def answer(q):
        top = retriever.retrieve(q, k=args.k)
        return llm.complete(build_rag_prompt(q, top)), top

    if args.ask:
        ans, top = answer(args.ask)
        print("\nSOURCES:")
        for i, c in enumerate(top):
            print(f"  [{i + 1}] {c.doc_id}#{c.chunk_id}: {c.text[:100]}...")
        print("\nANSWER:", ans)

    if args.eval:
        items = json.load(open(args.eval, encoding="utf-8"))
        ranks, f1s, sups = [], [], []
        for it in items:
            ans, top = answer(it["question"])
            ranks.append((list(dict.fromkeys(c.doc_id for c in top)), {it["doc"]}))
            f1s.append(token_f1(ans, it.get("answer", "")))
            sups.append(context_support(ans, " ".join(c.text for c in top)))
        m = evaluate_ranking(ranks, ks=(1, 3))
        print("\nretrieval:", {k: round(v, 3) for k, v in m.items()})
        print(f"generation: mean token-F1 {sum(f1s) / len(f1s):.3f} | mean context support {sum(sups) / len(sups):.3f}")


if __name__ == "__main__":
    main()
