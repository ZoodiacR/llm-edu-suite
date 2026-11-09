#!/usr/bin/env python
"""
An MCP server for the sample knowledge base, built with the official MCP Python SDK (FastMCP).

It exposes the documents in data/sample_docs/ to any MCP client:
    tools      search_docs(query, k)            BM25 search over the docs, one passage per line   (read-only)
               calculate(expression)            safe arithmetic (+ - * / ** % and parentheses)    (read-only)
               create_ticket(title, priority)   simulated write action                            (destructive: hosts should confirm)
    resources  docs://index                     JSON list of every document with its title
               docs://{name}                    resource template: the full text of one document, e.g. docs://shipping.md
    prompts    answer_with_sources(question)    a RAG prompt template with citation rules

Run it:
    python scripts/mcp_server.py                         # stdio (the default): waits for JSON-RPC on stdin, Ctrl+C to stop
    python scripts/mcp_server.py --docs path/to/docs     # serve another folder of .md / .txt files
    python scripts/mcp_server.py --http --port 8000      # Streamable HTTP on http://127.0.0.1:8000/mcp

Use it from Claude Code:
    claude mcp add llm-edu-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py
Use it from Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"llm-edu-docs": {"command": "python", "args": ["/ABSOLUTE/PATH/scripts/mcp_server.py"]}}}
Test it with the official inspector:
    npx @modelcontextprotocol/inspector python scripts/mcp_server.py

In stdio mode, stdout carries the protocol itself, so this server never prints to stdout:
logs go to stderr.

SDK note: this uses `mcp.server.fastmcp.FastMCP` from the 1.x SDK, which is what langchain-mcp-adapters
supports today. In the 2.x SDK the same class is called `mcp.server.mcpserver.MCPServer`, and the
decorators work the same way.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import operator
import os
import re
import sys
import time
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from rank_bm25 import BM25Okapi

DEFAULT_DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs"))
SERVER_NAME = "llm-edu-docs"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# Arithmetic is evaluated by walking the syntax tree, so nothing but numbers and these operators can run.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % ( ) are allowed")


def build_server(docs_dir: str = DEFAULT_DOCS, log_level: str = "WARNING") -> FastMCP:
    """Create the FastMCP server for a folder of documents. Tests and notebooks import this function.
    FastMCP logs every request at INFO to stderr; WARNING keeps the output readable."""
    files = sorted(glob.glob(os.path.join(docs_dir, "*.md")) + glob.glob(os.path.join(docs_dir, "*.txt")))
    docs = {os.path.basename(f): open(f, encoding="utf-8").read() for f in files}

    # Every non-empty line of the sample docs is one self-contained sentence, so a line is a good passage.
    passages = [(name, line.strip()) for name, text in docs.items()
                for line in text.splitlines() if line.strip() and not line.startswith("#")]
    bm25 = BM25Okapi([_tokens(f"{name.split('.')[0].replace('_', ' ')} {text}") for name, text in passages])
    tickets: list[dict] = []

    server = FastMCP(SERVER_NAME,
                     instructions="Company knowledge base (refunds, shipping, security, support). "
                                  "Use search_docs before answering policy questions and cite the file names.",
                     log_level=log_level)

    @server.tool(annotations=ToolAnnotations(title="Search docs", readOnlyHint=True, openWorldHint=False))
    def search_docs(query: str, k: int = 3) -> str:
        """Search the company documentation (refunds, shipping, security, support).
        Returns the k best passages, one per line, each prefixed with its source file in brackets."""
        scores = bm25.get_scores(_tokens(query))
        best = sorted(range(len(passages)), key=lambda i: -scores[i])[:max(1, min(k, 10))]
        return "\n".join(f"[{passages[i][0]}] {passages[i][1]}" for i in best if scores[i] > 0) or "no results"

    @server.tool(annotations=ToolAnnotations(title="Calculator", readOnlyHint=True, openWorldHint=False))
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '(17 * 23) + 4'."""
        try:
            value = _safe_eval(ast.parse(expression, mode="eval"))
        except SyntaxError as e:
            raise ValueError(f"not a valid expression: {expression!r}") from e
        return str(int(value)) if float(value).is_integer() else str(value)

    @server.tool(annotations=ToolAnnotations(title="Create support ticket", readOnlyHint=False,
                                             destructiveHint=True, idempotentHint=False, openWorldHint=True))
    def create_ticket(title: str, priority: Literal["low", "normal", "high"] = "normal") -> dict:
        """Create a support ticket in the helpdesk (a simulated write action)."""
        ticket = {"id": f"T-{1000 + len(tickets)}", "title": title, "priority": priority,
                  "created": time.strftime("%Y-%m-%d %H:%M")}
        tickets.append(ticket)
        print(f"[{SERVER_NAME}] created {ticket['id']}", file=sys.stderr)    # stdout belongs to the protocol
        return ticket

    @server.resource("docs://index", name="index", mime_type="application/json",
                     description="Every document in the knowledge base with its title")
    def index() -> str:
        return json.dumps([{"uri": f"docs://{n}", "title": t.splitlines()[0].lstrip("# ") if t else n}
                           for n, t in docs.items()])

    @server.resource("docs://{name}", mime_type="text/markdown", description="The full text of one document, e.g. docs://shipping.md")
    def read_doc(name: str) -> str:
        if name not in docs:
            raise ValueError(f"unknown document {name!r}; see docs://index")
        return docs[name]

    @server.prompt(description="Answer a question from the knowledge base, citing sources.")
    def answer_with_sources(question: str) -> str:
        return ("Use the search_docs tool to find passages relevant to the question, then answer ONLY from them and cite "
                f"the file names in brackets. If the docs do not contain the answer, say so.\n\nQuestion: {question}")

    return server


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--docs", default=DEFAULT_DOCS, help="folder with .md / .txt files to expose")
    p.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="WARNING", help="stderr log level (INFO shows every request)")
    args = p.parse_args()
    server = build_server(args.docs, args.log_level)
    if args.http:
        server.settings.host, server.settings.port = args.host, args.port
        print(f"[{SERVER_NAME}] Streamable HTTP on http://{args.host}:{args.port}{server.settings.streamable_http_path}", file=sys.stderr)
        server.run(transport="streamable-http")
    else:
        server.run()                     # stdio
