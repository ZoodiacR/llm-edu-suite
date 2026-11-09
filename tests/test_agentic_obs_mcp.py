"""Tests for the agentic patterns, the observability layer and the MCP implementation (no PyTorch needed)."""
import json
import os
import sys

import pytest

from llm_edu_suite.agentic import (Budget, BudgetedAgent, ConversationMemory, PlanAndExecute, ScriptedLLM, Supervisor,
                              VectorMemory, Worker, evaluate_trajectory, parse_numbered_list, reflect_and_revise,
                              trajectory_report)
from llm_edu_suite.agents import ToolRegistry
from llm_edu_suite.llm_client import MockLLM, Usage
from llm_edu_suite.mcp import (INVALID_PARAMS, METHOD_NOT_FOUND, InMemoryTransport, MCPClient, MCPError, MCPServer,
                          StdioTransport, register_mcp_tools, result_text, serve_stdio)
from llm_edu_suite.observability import (MetricsAggregator, Tracer, TracedLLM, check_slos, instrument_registry, percentile,
                                    redact_pii)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _registry():
    reg = ToolRegistry()

    @reg.register("calculate", "Evaluate an arithmetic expression and return the numeric result",
                  {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
    def calculate(expression: str) -> str:
        return str(eval(expression, {"__builtins__": {}}, {}))

    @reg.register("get_weather", "Get the current temperature for a city",
                  {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    def get_weather(city: str) -> dict:
        return {"city": city, "temp": 19}
    return reg


class _StuckModel:
    """Always requests the same tool call."""
    def __init__(self):
        self.usage = Usage()

    def chat(self, messages, tools=None, **kw):
        self.usage.add(10, 5)
        return {"content": None, "tool_calls": [{"id": "c", "name": "calculate", "arguments": {"expression": "1+1"}}]}


# Agentic patterns

def test_budgeted_agent_stop_reasons():
    reg = _registry()
    assert BudgetedAgent(_StuckModel(), reg, loop_window=3).run("x").stopped_reason == "loop_detected"
    assert BudgetedAgent(_StuckModel(), reg, budget=Budget(max_steps=50, max_tokens=40), loop_window=99).run("x").stopped_reason == "max_tokens"
    r = BudgetedAgent(_StuckModel(), reg, budget=Budget(max_steps=50, max_tool_calls=3), loop_window=99).run("x")
    assert r.stopped_reason == "max_tool_calls" and r.tool_calls == 3
    ok = BudgetedAgent(MockLLM(), reg).run("Please calculate (17 * 23) + 4 arithmetic")
    assert ok.stopped_reason == "answer" and "395" in ok.answer


def test_plan_and_execute_uses_one_tool_per_step():
    run = PlanAndExecute(MockLLM(), _registry()).run("Calculate 6 * 7 arithmetic, then get the weather temperature in Lima")
    assert len(run.plan) == 2
    called = [s.tool for r in run.step_runs for s in r.steps if s.tool]
    assert called == ["calculate", "get_weather"]
    assert "42" in run.answer and "Lima" in run.answer


def test_parse_numbered_list():
    assert parse_numbered_list("1. a\n2) b\n- c\n\n") == ["a", "b", "c"]


def test_reflection_stops_when_verifier_passes():
    r = reflect_and_revise(ScriptedLLM(["bad", "still bad", "good", "unused"]), "task", lambda d: (d == "good", "fix it"), max_rounds=5)
    assert r.passed and r.rounds == 3 and r.answer == "good"
    r2 = reflect_and_revise(ScriptedLLM(["bad"] * 5), "task", lambda d: (False, "no"), max_rounds=2)
    assert not r2.passed and r2.rounds == 2


def test_conversation_memory_stays_bounded():
    mem = ConversationMemory(llm=None, max_tokens=120, keep_last=2)
    for i in range(40):
        mem.add("user", f"turn {i}: the budget is {i * 100} dollars and nothing else changes today")
    assert mem.tokens() <= 120 and mem.n_summarizations > 0
    assert mem.messages()[1]["role"] == "system" and "Summary" in mem.messages()[1]["content"]


def test_vector_memory_dedup_and_recall():
    vm = VectorMemory()
    assert vm.add("Ana's company sells handmade ceramics online.")
    assert vm.add("The API rate limit is 100 requests per minute.")
    assert not vm.add("Ana's company sells handmade ceramics online.")       # duplicate rejected
    assert "ceramics" in vm.search("what does the company sell?", k=1)[0][1]["text"]


def test_supervisor_routes_by_description():
    sup = Supervisor([Worker("math", "arithmetic calculations sums products numbers", lambda q: "m"),
                      Worker("weather", "weather temperature forecast city", lambda q: "w")])
    assert sup.route("what is the temperature forecast for Lima") == "weather"
    assert sup.run("calculate the sum of these numbers") == {"worker": "math", "answer": "m"}


def test_trajectory_metrics():
    t = evaluate_trajectory(["search", "search", "calc"], ["search", "calc"])
    assert t["recall"] == 1.0 and t["precision"] == 1.0 and t["order_correct"] and not t["exact"] and t["extra_calls"] == 1
    t2 = evaluate_trajectory(["calc"], ["search", "calc"])
    assert t2["recall"] == 0.5 and t2["order_correct"] is False
    rep = trajectory_report([dict(t, success=True), dict(t2, success=False)])
    assert rep["success"] == 0.5


# Observability

def test_tracer_nesting_tokens_cost_and_errors():
    tracer = Tracer()
    llm = TracedLLM(MockLLM(), tracer, model="mock")
    with tracer.span("request", kind="chain", user="ana@example.com"):
        with tracer.span("retrieve", kind="retrieval"):
            pass
        llm.complete("Summarize: The model was trained. It works well.")
    with pytest.raises(ValueError):
        with tracer.span("failing", kind="chain"):
            raise ValueError("boom")
    assert len(tracer.traces()) == 2
    root = next(s for s in tracer.spans if s.name == "request")
    kids = [s for s in tracer.spans if s.parent_id == root.span_id]
    assert {s.name for s in kids} == {"retrieve", "llm.complete"}
    assert root.attributes["user"] == "<EMAIL>"                              # redacted before storage
    llm_span = next(s for s in kids if s.kind == "llm")
    assert llm_span.attributes["gen_ai.usage.input_tokens"] > 0 and llm_span.attributes["llm.cost_usd"] > 0
    assert tracer.trace_cost(root.trace_id) == pytest.approx(llm_span.attributes["llm.cost_usd"])
    failing = next(s for s in tracer.spans if s.name == "failing")
    assert failing.status == "error" and "boom" in failing.error


def test_instrument_registry_creates_tool_spans():
    tracer = Tracer()
    reg = instrument_registry(_registry(), tracer)
    assert reg.execute("calculate", {"expression": "2+3"}) == "5"
    assert [s.name for s in tracer.spans] == ["tool.calculate"]


def test_metrics_percentiles_and_slos():
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([1, 2, 3, 4, 5], 100) == 5
    tracer = Tracer()
    for _ in range(5):
        with tracer.span("req"):
            pass
    req = MetricsAggregator(tracer.spans).requests()
    assert req["count"] == 5 and req["error_rate"] == 0
    assert check_slos({"p95_ms": 500.0, "error_rate": 0.0}, {"p95_ms": 100}) == ["SLO violated: p95_ms = 500 > 100"]


def test_redact_pii_and_sampling():
    out = redact_pii("mail ana@x.com, card 4111 1111 1111 1111, key sk-live_abcdefghijklmnop, phone +51 987 654 321")
    assert "@" not in out and "4111" not in out and "sk-live" not in out and "987" not in out
    t = Tracer(sample_rate=0.0)
    with t.span("dropped"):
        pass
    assert t.spans == []


def test_exporters(tmp_path):
    tracer = Tracer()
    with tracer.span("a"):
        with tracer.span("b"):
            pass
    path = tmp_path / "t.jsonl"
    assert tracer.export_jsonl(str(path)) == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    spans = tracer.to_otlp()["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert {s["name"] for s in spans} == {"a", "b"}


# MCP

def _server():
    s = MCPServer("test", "1.0")

    @s.tool(description="add two integers", annotations={"readOnlyHint": True})
    def add(a: int, b: int) -> int:
        return a + b

    @s.tool(description="delete everything", annotations={"destructiveHint": True})
    def wipe() -> str:
        return "wiped"

    @s.resource("mem://hello", description="greeting")
    def hello():
        return "hello world"

    @s.prompt(description="greet someone")
    def greet(name: str) -> str:
        return f"Say hi to {name}"
    return s


def test_mcp_lifecycle_and_primitives():
    c = MCPClient(InMemoryTransport(_server()))
    info = c.initialize()
    assert info["serverInfo"]["name"] == "test" and set(info["capabilities"]) == {"tools", "resources", "prompts"}
    assert c.ping()
    tools = {t["name"]: t for t in c.list_tools()}
    assert tools["add"]["inputSchema"] == {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                           "required": ["a", "b"]}
    assert result_text(c.call_tool("add", {"a": 2, "b": 3})) == "5"
    assert c.read_resource("mem://hello") == "hello world"
    assert c.get_prompt("greet", {"name": "Ana"})[0]["content"]["text"] == "Say hi to Ana"


def test_mcp_errors():
    s = _server()
    c = MCPClient(InMemoryTransport(s))
    c.initialize()
    bad = c.call_tool("add", {"a": "two", "b": 3})
    assert bad["isError"] and "should be integer" in result_text(bad)
    with pytest.raises(MCPError) as e:
        c.call_tool("nope")
    assert e.value.code == INVALID_PARAMS
    with pytest.raises(MCPError) as e:
        c.request("x/y")
    assert e.value.code == METHOD_NOT_FOUND
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None   # notifications get no response
    assert s.initialized


def test_register_mcp_tools_allowlist_prefix_and_confirmation():
    c = MCPClient(InMemoryTransport(_server()))
    c.initialize()
    reg = ToolRegistry()
    added = register_mcp_tools(c, reg, allow=["add", "wipe"], prefix="t__", confirm_destructive=lambda n, a: False)
    assert added == ["t__add", "t__wipe"]
    assert reg.execute("t__add", {"a": 1, "b": 1}) == "2"
    assert reg.execute("t__wipe", {}) == "ERROR: action not confirmed by the user"


def test_serve_stdio_framing():
    import io
    lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), "not json",
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})]
    out = io.StringIO()
    serve_stdio(_server(), stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)
    responses = [json.loads(l) for l in out.getvalue().splitlines()]
    assert responses[0] == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert responses[1]["error"]["code"] == -32700 and len(responses) == 2


def test_mcp_server_script_over_real_stdio():
    c = MCPClient(StdioTransport([sys.executable, os.path.join(ROOT, "scripts", "mcp_server.py")]))
    try:
        assert c.initialize()["serverInfo"]["name"] == "llm_edu_suite-docs"
        assert "[shipping.md]" in result_text(c.call_tool("search_docs", {"query": "express shipping", "k": 1}))
        assert result_text(c.call_tool("calculate", {"expression": "6*7"})) == "42"
    finally:
        c.close()


def test_tool_spans_nest_under_the_active_span():
    # Tools run in a worker thread (for the timeout); the tracing context must follow them there.
    tracer = Tracer()
    reg = instrument_registry(_registry(), tracer)
    with tracer.span("request"):
        reg.execute("calculate", {"expression": "1+1"})
    assert len(tracer.traces()) == 1
    tool_span = next(s for s in tracer.spans if s.name == "tool.calculate")
    assert tool_span.parent_id is not None
