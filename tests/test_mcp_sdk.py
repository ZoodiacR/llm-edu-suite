"""
Tests for scripts/mcp_server.py, the knowledge-base server built with the official MCP SDK (FastMCP).

Most tests talk to the server in memory through a real ClientSession, so they need neither a subprocess
nor Ollama. One test runs the server as an stdio subprocess, as Claude Code would, and one test, which
is skipped when Ollama is not running, gives its tools to a LangChain agent through langchain-mcp-adapters.
"""
import asyncio
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER_PY = os.path.join(ROOT, "scripts", "mcp_server.py")

from llm_edu_suite.config import ollama_available  # noqa: E402

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402


def load_server_module():
    spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def in_memory(check):
    """Run `await check(session)` against a fresh in-memory server and return its result."""
    async def main():
        async with create_connected_server_and_client_session(load_server_module().build_server()) as session:
            return await check(session)
    return asyncio.run(main())


def test_lists_expected_tools_with_annotations():
    async def check(s):
        return {t.name: t for t in (await s.list_tools()).tools}
    tools = in_memory(check)
    assert set(tools) == {"search_docs", "calculate", "create_ticket"}
    assert tools["search_docs"].annotations.readOnlyHint is True
    assert tools["calculate"].annotations.readOnlyHint is True
    assert tools["create_ticket"].annotations.destructiveHint is True
    assert tools["create_ticket"].inputSchema["properties"]["priority"]["enum"] == ["low", "normal", "high"]
    # every tool declares all four hints explicitly (directories reject tools with missing hints)
    for t in tools.values():
        for hint in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            assert isinstance(getattr(t.annotations, hint), bool), f"{t.name}: {hint} not set"


def test_calculate_and_tool_errors():
    async def check(s):
        ok = await s.call_tool("calculate", {"expression": "(17 * 23) + 4"})
        unsafe = await s.call_tool("calculate", {"expression": "__import__('os')"})
        div0 = await s.call_tool("calculate", {"expression": "1/0"})
        bad_enum = await s.call_tool("create_ticket", {"title": "x", "priority": "urgent"})
        return ok, unsafe, div0, bad_enum
    ok, unsafe, div0, bad_enum = in_memory(check)
    assert ok.isError is False and ok.content[0].text == "395"
    assert unsafe.isError and div0.isError and bad_enum.isError


def test_create_ticket_returns_structured_ticket():
    async def check(s):
        return await s.call_tool("create_ticket", {"title": "refund missing", "priority": "high"})
    r = in_memory(check)
    ticket = json.loads(r.content[0].text)
    assert not r.isError and ticket["id"] == "T-1000" and ticket["priority"] == "high"


def test_search_docs_finds_the_labelled_document():
    eval_set = json.load(open(os.path.join(ROOT, "data", "sample_eval.json"), encoding="utf-8"))

    async def check(s):
        return [(await s.call_tool("search_docs", {"query": it["question"], "k": 1})).content[0].text for it in eval_set]
    outputs = in_memory(check)
    hits = [out.startswith(f"[{it['doc']}]") for it, out in zip(eval_set, outputs)]
    assert sum(hits) >= 5, list(zip([it["question"] for it in eval_set], outputs))


def test_resources_template_and_prompt():
    async def check(s):
        listed = [str(r.uri) for r in (await s.list_resources()).resources]
        templates = [t.uriTemplate for t in (await s.list_resource_templates()).resourceTemplates]
        index = json.loads((await s.read_resource("docs://index")).contents[0].text)
        shipping = (await s.read_resource("docs://shipping.md")).contents[0].text
        prompt = await s.get_prompt("answer_with_sources", {"question": "Are gift cards refundable?"})
        try:
            await s.read_resource("docs://missing.md")
            missing_raises = False
        except McpError:
            missing_raises = True
        return listed, templates, index, shipping, prompt, missing_raises
    listed, templates, index, shipping, prompt, missing_raises = in_memory(check)
    assert listed == ["docs://index"] and templates == ["docs://{name}"]
    assert {d["uri"] for d in index} == {"docs://refund_policy.md", "docs://security.md", "docs://shipping.md", "docs://support.md"}
    assert "Express shipping" in shipping
    assert "Are gift cards refundable?" in prompt.messages[0].content.text
    assert missing_raises


def test_stdio_subprocess_roundtrip():
    async def main():
        params = StdioServerParameters(command=sys.executable, args=[SERVER_PY])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
            init = await s.initialize()
            r = await s.call_tool("calculate", {"expression": "6*7"})
            return init, r
    init, r = asyncio.run(main())
    assert init.serverInfo.name == "llm-edu-docs"
    assert r.content[0].text == "42"


def test_streamable_http_requires_the_bearer_token():
    """--http --require-token: 401 without the token or with a wrong one, a working session with it."""
    import secrets
    import socket
    import subprocess
    import time

    httpx = pytest.importorskip("httpx")
    from mcp.client.streamable_http import streamablehttp_client

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    token = secrets.token_urlsafe(16)
    url = f"http://127.0.0.1:{port}/mcp"
    proc = subprocess.Popen([sys.executable, SERVER_PY, "--http", "--port", str(port), "--require-token"],
                            env={**os.environ, "MCP_AUTH_TOKEN": token}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        meta_url = f"http://127.0.0.1:{port}/.well-known/oauth-protected-resource/mcp"
        for _ in range(150):
            try:
                meta = httpx.get(meta_url, timeout=0.5).json()
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            pytest.fail("server did not start")
        assert meta["resource"] == url and meta["scopes_supported"] == ["docs:read"]
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        hdrs = {"Accept": "application/json, text/event-stream"}
        assert httpx.post(url, json=init, headers=hdrs, timeout=5).status_code == 401
        assert httpx.post(url, json=init, headers={**hdrs, "Authorization": "Bearer wrong"}, timeout=5).status_code == 401

        async def main():
            async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"}) as (read, write, _):
                async with ClientSession(read, write) as s:
                    await s.initialize()
                    return await s.call_tool("calculate", {"expression": "6*7"})
        assert asyncio.run(main()).content[0].text == "42"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_require_token_refuses_to_start_without_the_env_var():
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "MCP_AUTH_TOKEN"}
    r = subprocess.run([sys.executable, SERVER_PY, "--http", "--require-token"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "MCP_AUTH_TOKEN" in r.stderr


@pytest.mark.skipif(not ollama_available(), reason="needs a running Ollama server with qwen2.5:3b")
def test_agent_answers_with_mcp_tools():
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    from llm_edu_suite.config import get_llm

    async def main():
        client = MultiServerMCPClient({"docs": {"transport": "stdio", "command": sys.executable, "args": [SERVER_PY]}})
        async with client.session("docs") as session:
            tools = [t for t in await load_mcp_tools(session) if t.name == "search_docs"]
            agent = create_agent(get_llm(), tools=tools,
                                 system_prompt="Always call search_docs first and answer only from what it returns.")
            res = await agent.ainvoke({"messages": [{"role": "user", "content": "How many days do I have to request a full refund?"}]})
            return res["messages"]
    messages = asyncio.run(main())
    assert any(m.type == "tool" for m in messages)
    assert "30" in messages[-1].content
