"""
Tests for the LangChain / LangGraph code in notebooks 11 and 14.

The notebook generators keep their reusable cells (tools, helpers, graph builders) in module-level strings,
so these tests execute exactly the code the notebooks run, with fake chat models from langchain_core
instead of Ollama. The two tests marked `ollama` talk to the real local model and are skipped when no
Ollama server is running.
"""
import importlib
import os
import sys
import time

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "notebooks"))

nb11 = importlib.import_module("nb11_agents_tools_structured_output")
nb14 = importlib.import_module("nb14_agentic_ai")


class ScriptedModel(GenericFakeChatModel):
    """Replays AIMessages (tool calls included). bind_tools is a no-op so create_agent accepts it."""
    def bind_tools(self, tools, **kwargs):
        return self


class ListModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter(list(messages)))


def call(name, args, id_):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": id_}])


def run_cells(*cells):
    """Execute notebook cells in one namespace, as the kernel would. No Ollama: get_embeddings is never called."""
    ns = {"ROOT": ROOT, "get_embeddings": None}
    for cell in cells:
        exec(cell, ns)
    return ns


@pytest.fixture(scope="module")
def ns11():
    return run_cells(nb11.TOOLS_CELL, nb11.BY_HAND_CELL, nb11.EVAL_HELPERS_CELL)


@pytest.fixture()
def ns14():
    return run_cells(nb14.TOOLS_CELL, nb14.HELPERS_CELL)


# Notebook 11: tools, the loop by hand, scoring

def test_tool_schemas_come_from_hints_docstrings_and_pydantic(ns11):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    weather = convert_to_openai_tool(ns11["get_weather"])["function"]
    assert weather["description"].startswith("Get the current temperature")
    assert weather["parameters"]["required"] == ["city"]
    assert weather["parameters"]["properties"]["unit"]["enum"] == ["C", "F"]
    order = convert_to_openai_tool(ns11["order_status"])["function"]["parameters"]["properties"]["order_id"]
    assert order["pattern"] == r"^[A-Z]\d{3}$" and "capital letter" in order["description"]


def test_tools_run_and_validate(ns11):
    assert ns11["calculate"].invoke({"expression": "(17 * 23) + 4"}) == "395"
    assert ns11["get_weather"].invoke({"city": "Madrid", "unit": "F"})["temperature"] == 75
    with pytest.raises(Exception):
        ns11["order_status"].invoke({"order_id": "a-100"})          # pattern violation is caught before the function runs
    with pytest.raises(ValueError):
        ns11["calculate"].invoke({"expression": "__import__('os')"})


def test_run_by_hand_executes_tool_calls_and_stops_on_text(ns11):
    model = scripted(call("calculate", {"expression": "2 + 3"}, "c1"),
                     call("no_such_tool", {}, "c2"),
                     AIMessage("The answer is 5."))
    msgs = ns11["run_by_hand"](model, ns11["TOOLS"], "2 + 3?", verbose=False)
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert msgs[2].content == "5" and msgs[2].tool_call_id == "c1"
    assert msgs[4].status == "error" and "no tool named" in msgs[4].content
    assert msgs[-1].content == "The answer is 5."


def test_run_by_hand_respects_max_steps(ns11):
    model = scripted(*[call("calculate", {"expression": "1 + 1"}, f"c{i}") for i in range(10)])
    msgs = ns11["run_by_hand"](model, ns11["TOOLS"], "loop", max_steps=3, verbose=False)
    assert sum(isinstance(m, AIMessage) for m in msgs) == 3


def test_create_agent_returns_validation_errors_to_the_model(ns11):
    from langchain.agents import create_agent
    model = scripted(call("order_status", {"order_id": "a-100"}, "c1"), call("order_status", {"order_id": "A100"}, "c2"), AIMessage("shipped"))
    out = create_agent(model, ns11["TOOLS"]).invoke({"messages": [("user", "status of a-100")]})
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in tool_msgs] == ["error", "success"]
    assert "pattern" in tool_msgs[0].content


def test_score_case(ns11):
    msgs = [HumanMessage("q"), call("calculate", {"expression": "17*23+4"}, "c1"),
            ToolMessage("395", tool_call_id="c1"), AIMessage("It is 395.")]
    s = ns11["score_case"](msgs, ["calculate"], ["395"])
    assert s["tool_ok"] and s["args_ok"] and s["success"] and s["n_calls"] == 1
    s = ns11["score_case"](msgs, ["get_weather"], ["395"])
    assert not s["tool_ok"] and not s["success"]
    s = ns11["score_case"](msgs, ["calculate"], ["395"], arg_check=lambda calls: calls[0]["args"]["expression"] == "1")
    assert not s["args_ok"]
    bad = msgs[:2] + [ToolMessage("boom", tool_call_id="c1", status="error"), AIMessage("sorry")]
    assert ns11["score_case"](bad, ["calculate"], [])["args_ok"] is False
    assert ns11["score_case"]([HumanMessage("hi"), AIMessage("hello")], [], [])["success"]


# Notebook 14: helpers and guardrails

def test_tool_precision_recall(ns14):
    pr = ns14["tool_precision_recall"]
    assert pr(["a", "b"], ["a", "b"]) == (1.0, 1.0)
    assert pr(["a", "a", "b"], ["a", "b"]) == (2 / 3, 1.0)
    assert pr(["a"], ["a", "a", "c"]) == (1.0, 1 / 3)
    assert pr([], []) == (1.0, 1.0)
    assert pr([], ["a"]) == (0.0, 0.0)


def test_nb14_tools(ns14):
    assert ns14["order_status"].invoke({"order_id": " c300 "})["status"] == "delivered"
    assert ns14["calculate"].invoke({"expression": "120 * 1.21"}) == "145.2"
    with pytest.raises(ValueError):
        ns14["calculate"].invoke({"expression": "15% of 120"})


def _stuck_agent(ns, middleware=(), sleep=0.0):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def lookup(order_id: str) -> str:
        """Look up an order."""
        time.sleep(sleep)
        return "shipped"

    model = scripted(*[call("lookup", {"order_id": "A100"}, f"c{i}") for i in range(200)])
    return create_agent(model, [lookup], middleware=list(middleware))


def test_stop_reasons(ns14):
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
    stop_reason, GraphRecursionError = ns14["stop_reason"], ns14["GraphRecursionError"]
    with pytest.raises(GraphRecursionError) as e:
        _stuck_agent(ns14).invoke({"messages": [("user", "x")]}, {"recursion_limit": 6})
    assert stop_reason(error=e.value) == "recursion_limit"

    out = _stuck_agent(ns14, [ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")]).invoke({"messages": [("user", "x")]})
    assert stop_reason(out) == "model_call_limit" and len(ns14["tools_called"](out["messages"])) == 3

    out = _stuck_agent(ns14, [ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")]).invoke({"messages": [("user", "x")]})
    assert stop_reason(out) == "tool_call_limit"

    out = _stuck_agent(ns14, [ns14["WallClockLimit"](0.15)], sleep=0.1).invoke({"messages": [("user", "x")]})
    assert stop_reason(out) == "timeout" and 1 <= len(ns14["tools_called"](out["messages"])) <= 3

    assert stop_reason({"messages": [HumanMessage("q"), AIMessage("done")]}) == "answered"


def test_tool_errors_to_model_middleware(ns14):
    from langchain.agents import create_agent
    model = scripted(call("calculate", {"expression": "15% of 120"}, "c1"), call("calculate", {"expression": "120 * 0.15"}, "c2"), AIMessage("18"))
    out = create_agent(model, ns14["READ_TOOLS"], middleware=[ns14["tool_errors_to_model"]]).invoke({"messages": [("user", "15% of 120")]})
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error" and "calculate failed" in tool_msgs[0].content
    assert tool_msgs[1].content == "18.0"


# Notebook 14: graphs

def test_triage_graph_routes_by_condition(ns14):
    ns = run_cells(nb14.TOOLS_CELL, nb14.HELPERS_CELL, nb14.TRIAGE_CELL)
    triage = ns["triage"]
    assert triage.invoke({"ticket": "refund my order"})["category"] == "billing"
    assert triage.invoke({"ticket": "account hacked"})["reply"].startswith("Paging")
    out = triage.invoke({"ticket": "how do I reset my password"})
    assert out["category"] == "support" and out["log"] == ["classify -> support", "support"]   # the reducer appended
    assert "classify -.-> billing" in triage.get_graph().draw_mermaid()


def test_approval_gate_interrupt_and_resume(ns14):
    ns = run_cells(nb14.TOOLS_CELL, nb14.HELPERS_CELL, nb14.TRIAGE_CELL, nb14.APPROVAL_CELL)
    flow, Command = ns["refund_flow"], ns["Command"]
    cfg = {"configurable": {"thread_id": "t-approve"}}
    paused = flow.invoke({"order_id": "A100", "amount": 20.0}, cfg, version="v2")
    assert paused.interrupts[0].value == {"action": "issue_refund", "order_id": "A100", "amount": 20.0}
    assert flow.get_state(cfg).next == ("approval_gate",)
    assert ns["ISSUED_REFUNDS"] == []                                         # nothing ran while paused
    done = flow.invoke(Command(resume={"approved": True}), cfg, version="v2")
    assert done.value["result"].startswith("Refunded") and ns["ISSUED_REFUNDS"] == [("A100", 20.0)]

    cfg = {"configurable": {"thread_id": "t-reject"}}
    flow.invoke({"order_id": "B200", "amount": 500.0}, cfg, version="v2")
    done = flow.invoke(Command(resume={"approved": False, "reason": "too large"}), cfg, version="v2")
    assert done.value["result"] == "Refund cancelled by reviewer: too large" and len(ns["ISSUED_REFUNDS"]) == 1


def test_plan_execute_graph_wiring(ns14):
    ns = run_cells(nb14.TOOLS_CELL, nb14.HELPERS_CELL, nb14.TRIAGE_CELL, nb14.PLAN_EXEC_CELL)
    Plan = ns["Plan"]
    planner = RunnableLambda(lambda _: Plan(steps=["Look up order A100", "Add 21% tax"]))
    seen_prompts = []

    def fake_step(inp):
        seen_prompts.append(inp["messages"][0][1])
        i = len(seen_prompts)
        return {"messages": [HumanMessage("step"), call("order_status" if i == 1 else "calculate", {}, f"c{i}"),
                             ToolMessage("x", tool_call_id=f"c{i}"), AIMessage(f"result {i}")]}

    writer = ListModel(responses=[AIMessage("final answer")])
    app = ns["build_plan_execute"](planner, RunnableLambda(fake_step), writer)
    out = app.invoke({"task": "status and tax of A100"})
    assert out["plan"] == ["Look up order A100", "Add 21% tax"]
    assert out["results"] == ["result 1", "result 2"]
    assert out["tools"] == ["order_status", "calculate"]
    assert out["model_calls"] == 1 + 2 + 2 + 1                               # planner + 2 per step + writer
    assert out["answer"] == "final answer"
    assert "1. Look up order A100 -> result 1" in seen_prompts[1]             # later steps see earlier results
    assert "status and tax" not in seen_prompts[0]                            # ...but not the whole request

    empty = ns["build_plan_execute"](RunnableLambda(lambda _: Plan(steps=[])), RunnableLambda(fake_step), ListModel(responses=[AIMessage("nothing to do")]))
    assert empty.invoke({"task": "x"})["answer"] == "nothing to do"           # an empty plan goes straight to the writer


def test_reflection_verifier_and_loop(ns14):
    ns = run_cells(nb14.TOOLS_CELL, nb14.HELPERS_CELL, nb14.TRIAGE_CELL, nb14.REFLECTION_CELL)
    run_tests, extract_code = ns["run_tests"], ns["extract_code"]
    assert extract_code("Here:\n```python\ndef f():\n    return 1\n```\nbye") == "def f():\n    return 1"
    assert run_tests("def f(): return 1", "assert f() == 1") == (True, "all tests passed")
    ok, feedback = run_tests("def f(): return 2", "assert f() == 1")
    assert not ok and "assert f() == 1" in feedback and "AssertionError" in feedback
    ok, feedback = run_tests("while True: pass", "", timeout=1)
    assert not ok and "timed out" in feedback

    bad, good = "```python\ndef add(a, b):\n    return a - b\n```", "```python\ndef add(a, b):\n    return a + b\n```"
    tests = "assert add(2, 3) == 5"
    out = ns["build_reflection"](ListModel(responses=[AIMessage(bad), AIMessage(good)]), max_rounds=3).invoke({"task": "add", "tests": tests})
    assert out["passed"] and out["rounds"] == 2 and [h["passed"] for h in out["history"]] == [False, True]

    out = ns["build_reflection"](ListModel(responses=[AIMessage(bad)]), max_rounds=2).invoke({"task": "add", "tests": tests})
    assert not out["passed"] and out["rounds"] == 2                           # the round budget ends the loop


def test_router_sends_to_specialist_subgraph(ns14):
    from langchain.agents import create_agent
    ns = run_cells(nb14.TOOLS_CELL, nb14.HELPERS_CELL, nb14.TRIAGE_CELL, nb14.APPROVAL_CELL, nb14.ROUTER_CELL)
    specialists = {name: create_agent(ListModel(responses=[AIMessage(f"{name} here")]), [], name=name)
                   for name in ("billing", "support", "general")}
    classify = lambda messages: "support" if "ship" in str(messages[-1].content) else "billing"
    app = ns["build_router"](classify, specialists)
    out = app.invoke({"messages": [("user", "how long does shipping take?")]})
    assert out["route"] == "support" and out["messages"][-1].content == "support here"
    assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage", "AIMessage"]   # the subgraph's messages are merged, not duplicated
    out = app.invoke({"messages": [("user", "refund order A100")]})
    assert out["route"] == "billing" and out["messages"][-1].content == "billing here"


# Real model (skipped without Ollama)

def _ollama():
    from llm_edu_suite.config import ollama_available
    return ollama_available()


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama(), reason="needs a running Ollama server with qwen2.5:3b")
def test_real_tool_call(ns11):
    from llm_edu_suite.config import get_llm
    ai = get_llm().bind_tools(ns11["TOOLS"]).invoke("What is (17 * 23) + 4? Use the calculator.")
    assert ai.tool_calls and ai.tool_calls[0]["name"] == "calculate"
    assert ns11["calculate"].invoke(ai.tool_calls[0]).content == "395"


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama(), reason="needs a running Ollama server with qwen2.5:3b")
def test_real_structured_output():
    from typing import Literal
    from pydantic import BaseModel
    from llm_edu_suite.config import get_llm

    class Contact(BaseModel):
        name: str
        email: str
        topic: Literal["refund", "shipping", "account", "other"]

    c = get_llm().with_structured_output(Contact).invoke("Hi, I'm Tom Becker (tom@becker.dev) and I can't log into my account.")
    assert isinstance(c, Contact) and c.email == "tom@becker.dev" and c.topic == "account"
