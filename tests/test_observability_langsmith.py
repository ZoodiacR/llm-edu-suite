"""
Tests for notebook 28 (LLMOps with LangSmith).

The notebook's pure helpers (PII rules, groundedness, evaluators, latency summary) are defined in the HELPERS string
of tools/notebooks/nb28_observability_llmops.py. We read that string with `ast` (without running the generator, which
would rewrite the notebook) and execute it, so these tests check exactly the code the notebook ships.

Everything here runs offline. The one network test is skipped unless LangSmith and Ollama are both configured.
"""
import ast
import os
import uuid

import pytest

GENERATOR = os.path.join(os.path.dirname(__file__), "..", "tools", "notebooks", "nb28_observability_llmops.py")


def _load_helpers() -> dict:
    tree = ast.parse(open(GENERATOR, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "HELPERS" for t in node.targets):
            ns: dict = {}
            exec(compile(ast.literal_eval(node.value), "nb15_HELPERS", "exec"), ns)
            return ns
    raise AssertionError("HELPERS not found in the notebook 28 generator")


H = _load_helpers()


@pytest.fixture
def tracing_off(monkeypatch):
    """Tracing disabled the way a user would do it: LANGSMITH_TRACING=false and no key."""
    import langsmith.utils as ls_utils
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    ls_utils.get_env_var.cache_clear()           # langsmith caches env lookups
    yield
    ls_utils.get_env_var.cache_clear()


def test_traceable_helpers_work_with_tracing_disabled(tracing_off):
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    from langsmith.utils import tracing_is_enabled

    assert not tracing_is_enabled()
    seen = {}

    @traceable(run_type="tool", name="order_status")
    def order_status(order_id: str) -> dict:
        seen["run_tree"] = get_current_run_tree()
        if order_id != "A100":
            raise LookupError(order_id)
        return {"status": "shipped"}

    @traceable(run_type="chain", metadata={"app_version": "test"})
    def assistant(order_id: str) -> dict:
        return order_status(order_id, langsmith_extra={"run_id": uuid.uuid4()})

    assert assistant("A100") == {"status": "shipped"}
    assert seen["run_tree"] is None                # the notebook guards `get_current_run_tree()` for this case
    with pytest.raises(LookupError):               # errors propagate unchanged when tracing is off
        assistant("Z999")


def test_contains_fact_and_exact_match():
    ref = {"answer": "Express shipping delivers in 1 to 2 business days.", "must_contain": [["1 to 2", "1-2"]]}
    assert H["contains_fact"]({"answer": "It takes 1-2 business days."}, ref) == {"key": "contains_fact", "score": 1.0}
    assert H["contains_fact"]({"answer": "About a week."}, ref)["score"] == 0.0
    two_groups = {"answer": "x", "must_contain": [["aes-256"], ["tls 1.3"]]}
    assert H["contains_fact"]({"answer": "AES-256 at rest"}, two_groups)["score"] == 0.0   # every group must match
    assert H["contains_fact"]({"answer": "AES-256 at rest, TLS 1.3 in transit"}, two_groups)["score"] == 1.0
    assert H["contains_fact"]({}, ref)["score"] == 0.0                                       # a failed run has no answer
    assert H["exact_match"]({"answer": "express shipping delivers in 1 to 2 business days"}, ref)["score"] == 1.0
    assert H["exact_match"]({"answer": "It takes 1 to 2 business days."}, ref)["score"] == 0.0


class _FakeStructured:
    def __init__(self, grade):
        self.grade, self.prompts = grade, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.grade


class _FakeJudgeLLM:
    """Stands in for a chat model: with_structured_output(Grade) returns an object whose invoke() returns a Grade."""
    def __init__(self, correct: bool):
        self.correct, self.structured = correct, None

    def with_structured_output(self, schema):
        self.structured = _FakeStructured(schema(reasoning="same number of days", correct=self.correct))
        return self.structured


@pytest.mark.parametrize("correct,score", [(True, 1.0), (False, 0.0)])
def test_llm_judge_evaluator(correct, score):
    fake = _FakeJudgeLLM(correct)
    judge = H["make_correctness_judge"](fake)
    res = judge({"question": "How long does express shipping take?"}, {"answer": "Two days."},
                {"answer": "1 to 2 business days."})
    assert res == {"key": "correctness", "score": score, "comment": "same number of days"}
    prompt = fake.structured.prompts[0]
    assert "How long does express shipping take?" in prompt and "1 to 2 business days." in prompt and "Two days." in prompt


def test_pii_anonymizer_rules():
    out = H["redact"]({"q": "Ana ana.lopez@example.com, +51 987 654 321, card 4111 1111 1111 1111 was charged, key sk-live_9fA3kLmN0pQ",
                       "nested": [{"email": "b@c.org"}]})
    assert out["q"] == "Ana <email>, <phone>, card <card> was charged, key <api-key>"
    assert out["nested"][0]["email"] == "<email>"
    assert H["redact"]({"q": "How long does express shipping take?"})["q"] == "How long does express shipping take?"


def test_groundedness_and_latency_summary():
    ctx = ["Express shipping delivers in 1 to 2 business days and costs 15 dollars."]
    assert H["groundedness"]("Express shipping costs 15 dollars.", ctx) == 1.0
    assert H["groundedness"]("I have no tracking information.", ctx) == 0.0
    assert H["groundedness"]("", ctx) == 0.0
    s = H["latency_summary"]([float(i) for i in range(1, 101)] + [None])
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05) and s["p99"] == pytest.approx(99.01)
    assert H["latency_summary"]([])["n"] == 0


@pytest.mark.ollama            # the registered marker for tests that need local services; also needs LangSmith
def test_langsmith_trace_roundtrip():
    """Optional: trace one call, flush, and read it back. Needs a LangSmith key and a running Ollama."""
    from llm_edu_suite.config import langsmith_enabled, ollama_available
    if not (langsmith_enabled() and ollama_available()):
        pytest.skip("needs LANGSMITH_TRACING=true, LANGSMITH_API_KEY and a running Ollama")
    import asyncio
    import time
    from langsmith import Client, traceable, tracing_context

    client, project, rid = Client(), "llm-edu-suite-15-observability", uuid.uuid4()

    @traceable(run_type="tool")
    def ping(x: int) -> int:
        return x + 1

    with tracing_context(project_name=project, client=client, enabled=True):
        assert ping(1, langsmith_extra={"run_id": rid}) == 2
    client.flush()
    project_id = str(client.read_project(project_name=project).id)

    async def fetch():
        return [r async for r in client.runs.query(project_ids=[project_id], ids=[str(rid)], selects=["ID", "STATUS"])]

    for _ in range(20):
        runs = asyncio.run(fetch())
        if runs and runs[0].status != "pending":
            break
        time.sleep(2)
    assert runs and runs[0].status == "success"
