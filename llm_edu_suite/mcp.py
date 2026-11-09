"""
Model Context Protocol (MCP) from scratch: server, client, transports, and the bridge to our agents.

What MCP is
-----------
MCP is an open protocol, introduced by Anthropic in November 2024 and now supported by Claude, ChatGPT,
Gemini, Cursor, VS Code, Zed and others. It standardizes how an LLM application gets **context and
capabilities** from external programs. Before MCP, every app wrote its own integration for every tool,
which is N apps x M tools worth of glue code. With MCP a tool is written once as an *MCP server*, and
any *MCP client* (the host app) can use it, so the work drops to N + M.

    Host app (Claude Desktop, an IDE, your agent)
      └── MCP client  ──JSON-RPC 2.0──►  MCP server (your code: DB, files, APIs, ...)
                       stdio | Streamable HTTP

A server can expose three kinds of things, and the difference is who decides to use them:
    tools      functions the MODEL may call (search, write a file, run SQL)        tools/list, tools/call
    resources  read-only data the APPLICATION can load into context (files, rows)  resources/list, resources/read
    prompts    reusable prompt templates the USER can pick (slash commands)        prompts/list, prompts/get

A session starts with a short handshake. The client sends `initialize` with its protocol version and
capabilities, the server replies with its own capabilities, and the client confirms with a
`notifications/initialized` notification. After that, normal requests can flow.
Every message is JSON-RPC 2.0. Requests carry an `id` and receive either a `result` or an `error`;
notifications have no `id` and never get an answer. On the stdio transport each message is a single
line of JSON.

This module implements the part of the protocol (version 2025-06-18) that you need to really
understand it: the lifecycle, ping, tools, resources, prompts, error codes and stdio framing. The
official SDKs (`pip install mcp`, which ships FastMCP, and `@modelcontextprotocol/sdk`) add Streamable
HTTP, auth (OAuth 2.1), pagination, notifications, sampling and elicitation on top.

Components
----------
MCPServer                register tools, resources and prompts with decorators; `handle(message)` returns the response
serve_stdio              runs a server over stdin/stdout, which is how Claude Desktop and Claude Code launch them
InMemoryTransport        client and server in the same process, handy for tests and notebooks
StdioTransport           the client side: launches a server subprocess and talks over its pipes
MCPClient                initialize, list/call tools, list/read resources, list/get prompts
register_mcp_tools       exposes an MCP server's tools inside an agents.ToolRegistry, with an allowlist and gating
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

PROTOCOL_VERSION = "2025-06-18"

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def jsonrpc_request(method: str, params: dict | None = None, id: int | str | None = None) -> dict:
    """Build a request (with an id, so it expects a response) or a notification (id=None, no response)."""
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if id is not None:
        msg["id"] = id
    return msg


def jsonrpc_result(id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(id, code: int, message: str, data: Any = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


# Server
@dataclass
class _ToolDef:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]
    annotations: dict = field(default_factory=dict)     # optional hints: readOnlyHint, destructiveHint, idempotentHint, openWorldHint


@dataclass
class _ResourceDef:
    uri: str
    name: str
    description: str
    mime_type: str
    read: Callable[[], str]


@dataclass
class _PromptDef:
    name: str
    description: str
    arguments: list[dict]
    render: Callable[..., str]


def _schema_from_signature(fn: Callable) -> dict:
    """Derive a JSON schema from the function's type hints, the same way FastMCP does.

    Only str/int/float/bool/list/dict are mapped, and any parameter without a default is required.
    """
    types = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
        if isinstance(ann, str):                                  # annotations arrive as strings under "from __future__ import annotations"
            ann = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}.get(ann, str)
        props[name] = {"type": types.get(ann, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


class MCPServer:
    """
        server = MCPServer("docs-server", "1.0")

        @server.tool(description="Search the docs", annotations={"readOnlyHint": True})
        def search_docs(query: str, k: int = 3) -> str: ...

        @server.resource("docs://refund_policy.md", description="Refund policy")
        def refund(): return open(...).read()

        @server.prompt(description="Summarize a document")
        def summarize(doc: str) -> str: return f"Summarize: {doc}"

    `handle(message)` is a pure function from a message to its response (None for notifications).
    Because it knows nothing about I/O, the same server runs unchanged over any transport.
    """

    def __init__(self, name: str, version: str = "0.1.0", instructions: str = ""):
        self.name, self.version, self.instructions = name, version, instructions
        self.tools: dict[str, _ToolDef] = {}
        self.resources: dict[str, _ResourceDef] = {}
        self.prompts: dict[str, _PromptDef] = {}
        self.initialized = False
        self.log: list[dict] = []                                  # every message received, kept for teaching and debugging

    # Registration decorators
    def tool(self, name: str | None = None, description: str | None = None, input_schema: dict | None = None,
             annotations: dict | None = None):
        def deco(fn):
            n = name or fn.__name__
            self.tools[n] = _ToolDef(n, description or (fn.__doc__ or "").strip(), input_schema or _schema_from_signature(fn),
                                     fn, annotations or {})
            return fn
        return deco

    def resource(self, uri: str, name: str | None = None, description: str = "", mime_type: str = "text/plain"):
        def deco(fn):
            self.resources[uri] = _ResourceDef(uri, name or uri.rsplit("/", 1)[-1], description, mime_type, fn)
            return fn
        return deco

    def prompt(self, name: str | None = None, description: str = ""):
        def deco(fn):
            n = name or fn.__name__
            args = [{"name": a, "required": p.default is inspect.Parameter.empty}
                    for a, p in inspect.signature(fn).parameters.items()]
            self.prompts[n] = _PromptDef(n, description or (fn.__doc__ or "").strip(), args, fn)
            return fn
        return deco

    # Dispatch
    def handle(self, message: dict) -> dict | None:
        self.log.append(message)
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return jsonrpc_error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Invalid Request")
        method, params, msg_id = message["method"], message.get("params") or {}, message.get("id")
        if msg_id is None:                                         # a notification: act on it, but never answer
            if method == "notifications/initialized":
                self.initialized = True
            return None
        handler = {
            "initialize": self._initialize, "ping": lambda p: {},
            "tools/list": self._tools_list, "tools/call": self._tools_call,
            "resources/list": self._resources_list, "resources/read": self._resources_read,
            "prompts/list": self._prompts_list, "prompts/get": self._prompts_get,
        }.get(method)
        if handler is None:
            return jsonrpc_error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        try:
            return jsonrpc_result(msg_id, handler(params))
        except MCPError as e:
            return jsonrpc_error(msg_id, e.code, e.message, e.data)
        except Exception as e:  # never crash the server on a bad request
            return jsonrpc_error(msg_id, INTERNAL_ERROR, f"{type(e).__name__}: {e}")

    def _initialize(self, params: dict) -> dict:
        # Version negotiation: echo the client's version if we support it, otherwise offer ours.
        version = params.get("protocolVersion", PROTOCOL_VERSION)
        caps = {}
        if self.tools:
            caps["tools"] = {"listChanged": False}
        if self.resources:
            caps["resources"] = {"subscribe": False, "listChanged": False}
        if self.prompts:
            caps["prompts"] = {"listChanged": False}
        out = {"protocolVersion": version if version == PROTOCOL_VERSION else PROTOCOL_VERSION,
               "capabilities": caps, "serverInfo": {"name": self.name, "version": self.version}}
        if self.instructions:
            out["instructions"] = self.instructions
        return out

    def _tools_list(self, params: dict) -> dict:
        return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.input_schema,
                           **({"annotations": t.annotations} if t.annotations else {})} for t in self.tools.values()]}

    def _tools_call(self, params: dict) -> dict:
        from .agents import validate_args
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in self.tools:
            raise MCPError(INVALID_PARAMS, f"Unknown tool: {name}")
        tool = self.tools[name]
        problems = validate_args(args, tool.input_schema)
        if problems:                                               # report bad arguments as a tool-level error (isError) so the model can retry
            return {"content": [{"type": "text", "text": "Invalid arguments: " + "; ".join(problems)}], "isError": True}
        try:
            out = tool.fn(**args)
        except Exception as e:
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
        if isinstance(out, (dict, list)):                          # send the structured result along with a text rendering of it
            return {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out if isinstance(out, dict) else {"items": out},
                    "isError": False}
        return {"content": [{"type": "text", "text": str(out)}], "isError": False}

    def _resources_list(self, params: dict) -> dict:
        return {"resources": [{"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
                              for r in self.resources.values()]}

    def _resources_read(self, params: dict) -> dict:
        uri = params.get("uri")
        if uri not in self.resources:
            raise MCPError(-32002, f"Resource not found: {uri}")      # MCP's own error code for a missing resource
        r = self.resources[uri]
        return {"contents": [{"uri": uri, "mimeType": r.mime_type, "text": r.read()}]}

    def _prompts_list(self, params: dict) -> dict:
        return {"prompts": [{"name": p.name, "description": p.description, "arguments": p.arguments} for p in self.prompts.values()]}

    def _prompts_get(self, params: dict) -> dict:
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in self.prompts:
            raise MCPError(INVALID_PARAMS, f"Unknown prompt: {name}")
        p = self.prompts[name]
        missing = [a["name"] for a in p.arguments if a["required"] and a["name"] not in args]
        if missing:
            raise MCPError(INVALID_PARAMS, f"Missing prompt arguments: {missing}")
        return {"description": p.description, "messages": [{"role": "user", "content": {"type": "text", "text": p.render(**args)}}]}


def serve_stdio(server: MCPServer, stdin=None, stdout=None) -> None:
    """
    The server side of the stdio transport: read one JSON message per line from stdin and write the
    responses to stdout.
    stdout is reserved for protocol messages. A stray print() would corrupt the stream, so any
    logging has to go to stderr.
    """
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as e:
            response = jsonrpc_error(None, PARSE_ERROR, f"Parse error: {e}")
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


# Transports (client side)
class InMemoryTransport:
    """Client and server in one process. Messages still go through JSON serialization, just as they would on a wire."""

    def __init__(self, server: MCPServer):
        self.server = server
        self.sent = 0

    def send(self, message: dict) -> dict | None:
        self.sent += 1
        response = self.server.handle(json.loads(json.dumps(message)))
        return json.loads(json.dumps(response)) if response is not None else None

    def close(self) -> None:
        pass


class StdioTransport:
    """
    Launch an MCP server as a subprocess and exchange newline-delimited JSON over its stdin/stdout.
    This is exactly what Claude Desktop does with the "command" entry of its config file.
    """

    def __init__(self, command: list[str], cwd: str | None = None, env: dict | None = None, timeout: float = 30.0):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1, cwd=cwd, env=env)
        self.timeout = timeout

    def send(self, message: dict) -> dict | None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        if "id" not in message:                                   # a notification, so there is no response to wait for
            return None
        result: list[str] = []
        reader = threading.Thread(target=lambda: result.append(self.proc.stdout.readline()), daemon=True)
        reader.start(); reader.join(self.timeout)
        if not result or not result[0]:
            raise TimeoutError(f"no response from MCP server within {self.timeout}s; stderr: {self._stderr()}")
        return json.loads(result[0])

    def _stderr(self) -> str:
        try:
            return self.proc.stderr.read(2000) if self.proc.poll() is not None else "(server still running)"
        except Exception:
            return ""

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# Client
class MCPClient:
    """
        client = MCPClient(InMemoryTransport(server))            # or StdioTransport(["python", "scripts/mcp_server.py"])
        client.initialize()
        tools = client.list_tools()
        result = client.call_tool("search_docs", {"query": "refund"})
    """

    def __init__(self, transport, client_name: str = "llm_edu_suite-client", client_version: str = "0.1.0"):
        self.transport = transport
        self.client_info = {"name": client_name, "version": client_version}
        self._next_id = 0
        self.server_info: dict = {}
        self.server_capabilities: dict = {}
        self.instructions = ""

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        response = self.transport.send(jsonrpc_request(method, params, id=self._next_id))
        if response is None or response.get("id") != self._next_id:
            raise MCPError(INTERNAL_ERROR, f"unexpected response: {response}")
        if "error" in response:
            e = response["error"]
            raise MCPError(e["code"], e["message"], e.get("data"))
        return response["result"]

    def notify(self, method: str, params: dict | None = None) -> None:
        self.transport.send(jsonrpc_request(method, params))

    def initialize(self) -> dict:
        result = self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": self.client_info})
        self.server_info, self.server_capabilities = result.get("serverInfo", {}), result.get("capabilities", {})
        self.instructions = result.get("instructions", "")
        self.notify("notifications/initialized")
        return result

    def ping(self) -> bool:
        return self.request("ping") == {}

    def list_tools(self) -> list[dict]:
        return self.request("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})

    def list_resources(self) -> list[dict]:
        return self.request("resources/list")["resources"]

    def read_resource(self, uri: str) -> str:
        return "\n".join(c.get("text", "") for c in self.request("resources/read", {"uri": uri})["contents"])

    def list_prompts(self) -> list[dict]:
        return self.request("prompts/list")["prompts"]

    def get_prompt(self, name: str, arguments: dict | None = None) -> list[dict]:
        return self.request("prompts/get", {"name": name, "arguments": arguments or {}})["messages"]

    def close(self) -> None:
        self.transport.close()


def result_text(result: dict) -> str:
    """Flatten a tools/call result into the string the LLM will see as the tool's observation."""
    text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
    return f"ERROR: {text}" if result.get("isError") else text


# Bridge from MCP tools to an agents.ToolRegistry
def register_mcp_tools(client: MCPClient, registry, allow: list[str] | None = None, prefix: str = "",
                       confirm_destructive: Callable[[str, dict], bool] | None = None) -> list[str]:
    """
    Make every tool of an MCP server callable by our Agent. The agent sees an ordinary tool, and
    running it sends `tools/call` to the server. A host should apply a few security controls here:
      allow                 an allowlist of tool names (default: all). Never auto-enable every tool of a
                            server you don't trust.
      prefix                a namespace for the tools ("docs__search"), so two servers cannot shadow each
                            other's names.
      confirm_destructive   a human-in-the-loop callback for tools marked destructiveHint=True (or not
                            readOnly). Keep in mind that annotations are only *hints from the server*,
                            so they are worth as much as your trust in that server.
    Also remember that tool descriptions and results are untrusted text and can carry prompt injection.
    """
    from .agents import Tool
    added = []
    for t in client.list_tools():
        if allow is not None and t["name"] not in allow:
            continue
        ann = t.get("annotations", {})
        risky = ann.get("destructiveHint", False) or not ann.get("readOnlyHint", False)

        def make(name, risky_):
            def call(**kwargs):
                if risky_ and confirm_destructive is not None and not confirm_destructive(name, kwargs):
                    return "ERROR: action not confirmed by the user"
                return result_text(client.call_tool(name, kwargs))
            return call
        registry.add(Tool(prefix + t["name"], t.get("description", ""), t.get("inputSchema", {"type": "object", "properties": {}}),
                          make(t["name"], risky)))
        added.append(prefix + t["name"])
    return added
