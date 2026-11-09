#!/usr/bin/env python
"""
A real, runnable MCP server (stdio transport) built with llm_edu_suite.mcp — no SDK needed.

It exposes the sample knowledge base in data/sample_docs/ to any MCP client:
    tools      search_docs(query, k)    hybrid BM25 + dense retrieval over the docs   (read-only)
               calculate(expression)    safe arithmetic                                (read-only)
               create_ticket(title, priority)   simulated write action                 (destructive, so clients should confirm)
    resources  docs://<file>            every .md / .txt file, readable by the host app
    prompts    answer_with_sources(question)   a RAG prompt template with citation rules

Run it (it waits for JSON-RPC messages on stdin; Ctrl+C to stop):
    python scripts/mcp_server.py
    python scripts/mcp_server.py --docs path/to/your/docs

Use it from Claude Code:
    claude mcp add llm_edu_suite-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py
Use it from Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"llm_edu_suite-docs": {"command": "python", "args": ["/ABSOLUTE/PATH/scripts/mcp_server.py"]}}}
Test it with the official inspector:
    npx @modelcontextprotocol/inspector python scripts/mcp_server.py

Keep in mind that stdout carries the protocol itself, so an stdio server must never print() to
stdout. Logs go to stderr instead.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_edu_suite.mcp import MCPServer, serve_stdio
from llm_edu_suite.rag import Retriever, chunk_documents

DEFAULT_DOCS = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")


def build_server(docs_dir: str = DEFAULT_DOCS) -> MCPServer:
    files = sorted(glob.glob(os.path.join(docs_dir, "*.md")) + glob.glob(os.path.join(docs_dir, "*.txt")))
    docs = {os.path.basename(f): open(f, encoding="utf-8").read() for f in files}
    retriever = Retriever(chunk_documents(docs, size=80, overlap=20))
    tickets: list[dict] = []

    server = MCPServer("llm_edu_suite-docs", "0.1.0",
                       instructions="Company knowledge base. Use search_docs before answering policy questions and cite the file names.")

    @server.tool(description="Search the company documentation (refunds, shipping, security, support). Returns the best passages with their source file.",
                 annotations={"readOnlyHint": True})
    def search_docs(query: str, k: int = 3) -> str:
        hits = retriever.retrieve(query, k=k)
        return "\n".join(f"[{c.doc_id}] {c.text}" for c in hits) or "no results"

    @server.tool(description="Evaluate an arithmetic expression, e.g. '(17 * 23) + 4'.", annotations={"readOnlyHint": True})
    def calculate(expression: str) -> str:
        if not all(ch in "0123456789+-*/(). " for ch in expression):
            raise ValueError("only arithmetic is allowed")
        return str(eval(expression, {"__builtins__": {}}, {}))

    @server.tool(description="Create a support ticket (simulated write action).",
                 input_schema={"type": "object", "properties": {"title": {"type": "string"},
                                                                "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
                               "required": ["title"]},
                 annotations={"readOnlyHint": False, "destructiveHint": True})
    def create_ticket(title: str, priority: str = "normal") -> dict:
        ticket = {"id": f"T-{1000 + len(tickets)}", "title": title, "priority": priority, "created": time.strftime("%Y-%m-%d %H:%M")}
        tickets.append(ticket)
        print(f"[mcp_server] created {ticket['id']}", file=sys.stderr)          # stdout belongs to the protocol, so log to stderr
        return ticket

    for name, text in docs.items():
        server.resource(f"docs://{name}", description=text.splitlines()[0].lstrip("# ") if text else name,
                        mime_type="text/markdown")(lambda t=text: t)

    @server.prompt(description="Answer a question using the knowledge base, citing sources.")
    def answer_with_sources(question: str) -> str:
        return (f"Use the search_docs tool to find passages relevant to the question, then answer ONLY from them and cite "
                f"the file names in brackets. If the docs do not contain the answer, say so.\n\nQuestion: {question}")

    return server


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default=DEFAULT_DOCS, help="folder with .md / .txt files to expose")
    args = p.parse_args()
    server = build_server(args.docs)
    print(f"[mcp_server] {len(server.tools)} tools, {len(server.resources)} resources, {len(server.prompts)} prompts — waiting on stdin",
          file=sys.stderr)
    serve_stdio(server)
