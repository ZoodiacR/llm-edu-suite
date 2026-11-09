"""
Agentic AI patterns, built on top of `agents.py` (tools + the basic loop).

`agents.Agent` is the core loop: the model asks for a tool, we run it, the model reads the observation,
and so on. Real agentic systems add structure around that loop, and this module implements the
patterns you keep running into in practice (Anthropic's "Building effective agents", LangGraph, the
OpenAI Agents SDK, CrewAI and friends):

Pattern                 What it adds                                             Class / function
---------------------   ------------------------------------------------------   ----------------------------
Guardrails & budgets    hard limits on steps, tool calls, tokens, cost, time;     Budget, BudgetedAgent
                        loop detection (same call repeated)
Plan-and-execute        decompose a task into steps first, then execute each      PlanAndExecute
                        step with tools, then synthesize
Reflection              draft -> verify (code or LLM critic) -> revise, until     reflect_and_revise
                        the verifier passes or rounds run out
Memory                  short-term (token-budgeted conversation window +          ConversationMemory,
                        running summary) and long-term (vector store of facts)    VectorMemory
Multi-agent routing     a supervisor sends each request to the right specialist   Supervisor
Trajectory evaluation   score the PATH, not only the final answer: right tools,   evaluate_trajectory,
                        right order, no wasted steps                              trajectory_report

Everything works with the offline (and deterministic) MockLLM or with any OpenAI-compatible client.
A rule of thumb from production: start with the simplest pattern that works, which is often a single
LLM call or a fixed workflow. Add autonomy only where it measurably helps, and always pair it with
budgets and evaluation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .agents import Agent, AgentRun, AgentStep, ToolRegistry


# Budgets, guardrails and a budget-aware agent loop

@dataclass
class Budget:
    """Hard limits for one agent run. None means unlimited; hitting any limit stops the run cleanly."""
    max_steps: int | None = 8
    max_tool_calls: int | None = 10
    max_tokens: int | None = None           # prompt + completion tokens, as reported by llm.usage
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    usd_per_m_in: float = 0.5
    usd_per_m_out: float = 1.5


@dataclass
class BudgetedRun(AgentRun):
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    tool_calls: int = 0


class BudgetedAgent(Agent):
    """
    agents.Agent with guardrails. Instead of looping forever it stops with an explicit `stopped_reason`,
    one of: max_steps, max_tool_calls, max_tokens, max_cost, max_seconds, loop_detected or answer.

    Loop detection catches a classic failure: the model keeps asking for the same tool with the same
    arguments because it is ignoring the observation. After `loop_window` identical calls in a row we
    stop and say so.
    """

    def __init__(self, llm, registry: ToolRegistry, budget: Budget | None = None, loop_window: int = 3, **kw):
        super().__init__(llm, registry, **kw)
        self.budget = budget or Budget()
        self.loop_window = loop_window

    def _usage(self) -> tuple[int, int]:
        u = getattr(self.llm, "usage", None)
        return (getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)) if u else (0, 0)

    def run(self, user_message: str, verbose: bool = False) -> BudgetedRun:
        b = self.budget
        t0, (in0, out0) = time.perf_counter(), self._usage()
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}]
        steps: list[AgentStep] = []
        tools = self.registry.to_openai()
        recent_calls: list[str] = []
        n_tool_calls = 0

        def spent():
            i, o = self._usage()
            tok = (i - in0) + (o - out0)
            cost = (i - in0) / 1e6 * b.usd_per_m_in + (o - out0) / 1e6 * b.usd_per_m_out
            return tok, cost, time.perf_counter() - t0

        def finish(answer, reason):
            tok, cost, secs = spent()
            return BudgetedRun(answer, steps, messages, reason, tokens=tok, cost_usd=cost, seconds=secs, tool_calls=n_tool_calls)

        for i in range(b.max_steps if b.max_steps is not None else 10**9):
            tok, cost, secs = spent()                                  # check the budgets before we spend any more
            if b.max_tokens is not None and tok >= b.max_tokens:
                return finish("Stopped: token budget exhausted.", "max_tokens")
            if b.max_cost_usd is not None and cost >= b.max_cost_usd:
                return finish("Stopped: cost budget exhausted.", "max_cost")
            if b.max_seconds is not None and secs >= b.max_seconds:
                return finish("Stopped: time budget exhausted.", "max_seconds")

            reply = self.llm.chat(messages, tools=tools)
            if not reply.get("tool_calls"):
                steps.append(AgentStep(i, None, None, None, reply.get("content")))
                messages.append({"role": "assistant", "content": reply.get("content")})
                return finish(reply.get("content") or "", "answer")
            messages.append({"role": "assistant", "content": reply.get("content"), "tool_calls": reply["tool_calls"]})
            for call in reply["tool_calls"]:
                name, args = call["name"], call.get("arguments") or {}
                if b.max_tool_calls is not None and n_tool_calls >= b.max_tool_calls:
                    return finish("Stopped: tool-call budget exhausted.", "max_tool_calls")
                signature = name + json.dumps(args, sort_keys=True)
                recent_calls.append(signature)
                if len(recent_calls) >= self.loop_window and len(set(recent_calls[-self.loop_window:])) == 1:
                    return finish(f"Stopped: loop detected ({name} called {self.loop_window}x with the same arguments).", "loop_detected")
                if self.confirm is not None and not self.confirm(name, args):
                    result = "ERROR: action not confirmed by the user"
                else:
                    result = self.registry.execute(name, args)
                n_tool_calls += 1
                result = result[: self.max_obs_chars]
                if verbose:
                    print(f"  [step {i}] {name}({json.dumps(args)}) -> {result[:100]}")
                steps.append(AgentStep(i, name, args, result, None))
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": result})
        return finish("Stopped: maximum number of steps reached.", "max_steps")


# Plan-and-execute

PLAN_PROMPT = ("Break the following task into a short numbered list of independent steps, one action per step.\n"
               "Task: {task}\nSteps:")
SYNTH_PROMPT = ("Combine the step results into one final answer for the user.\nTask: {task}\n"
                "Step results:\n{results}\nFinal answer:")


def parse_numbered_list(text: str) -> list[str]:
    """'1. foo\\n2) bar\\n- baz' -> ['foo', 'bar', 'baz']"""
    items = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines()]
    return [i for i in items if i]


@dataclass
class PlanRun:
    task: str
    plan: list[str]
    step_runs: list[AgentRun] = field(default_factory=list)
    answer: str = ""


class PlanAndExecute:
    """
    1. Plan: one LLM call turns the task into steps (a cheap model is often good enough here).
    2. Execute: each step runs through a budgeted, tool-using agent.
    3. Synthesize: one more LLM call writes the final answer from the step results.

    Why bother? Long tasks tend to fail inside a single ReAct loop because the model loses track of
    where it is. An explicit plan can be inspected, approved by a human, and redone when a step fails.
    """

    def __init__(self, llm, registry: ToolRegistry, executor_budget: Budget | None = None, max_plan_steps: int = 6):
        self.llm, self.registry, self.max_plan_steps = llm, registry, max_plan_steps
        self.executor = BudgetedAgent(llm, registry, budget=executor_budget or Budget(max_steps=4, max_tool_calls=3))

    def plan(self, task: str) -> list[str]:
        return parse_numbered_list(self.llm.complete(PLAN_PROMPT.format(task=task)))[: self.max_plan_steps]

    def run(self, task: str, verbose: bool = False) -> PlanRun:
        out = PlanRun(task, self.plan(task))
        if verbose:
            print("PLAN:", *[f"\n  {i + 1}. {s}" for i, s in enumerate(out.plan)])
        for i, step in enumerate(out.plan):
            r = self.executor.run(step, verbose=verbose)
            out.step_runs.append(r)
        results = "\n".join(f"{i + 1}. {s} -> {r.answer}" for i, (s, r) in enumerate(zip(out.plan, out.step_runs)))
        out.answer = self.llm.complete(SYNTH_PROMPT.format(task=task, results=results))
        return out


# Reflection: draft, verify, revise

@dataclass
class ReflectionRun:
    answer: str
    passed: bool
    rounds: int
    history: list[dict] = field(default_factory=list)          # [{"draft", "feedback", "passed"}]


def reflect_and_revise(llm, task: str, verifier: Callable[[str], tuple[bool, str]], max_rounds: int = 3) -> ReflectionRun:
    """
    The single most useful "agentic" pattern whenever you can actually check an answer:

        draft = llm(task)
        repeat: ok, feedback = verifier(draft); if ok: stop; draft = llm(task + draft + feedback)

    The verifier can be code (unit tests, a JSON-schema validator, a SQL dry-run, a calculator), which
    is the best option because it is reliable, or an LLM critic with a rubric. Without a trustworthy
    signal, "self-reflection" often just rewrites the answer without making it better (Huang et al. 2024,
    "LLMs cannot self-correct reasoning yet"), so prefer external verification when you can.
    """
    history = []
    draft = llm.complete(task)
    for r in range(1, max_rounds + 1):
        ok, feedback = verifier(draft)
        history.append({"draft": draft, "feedback": feedback, "passed": ok})
        if ok:
            return ReflectionRun(draft, True, r, history)
        if r == max_rounds:
            break
        draft = llm.complete(f"{task}\n\nYour previous answer:\n{draft}\n\nA reviewer found these problems:\n{feedback}\n\n"
                             "Revise the answer. Return only the corrected answer.")
    return ReflectionRun(draft, False, max_rounds, history)


class ScriptedLLM:
    """
    A test double that replays a fixed list of responses and records the prompts it was given. Handy
    for unit-testing control flow (reflection loops, retries, routing) without depending on how good
    the model is.
    """

    def __init__(self, responses: list[str]):
        from .llm_client import Usage
        self.responses, self.prompts, self.usage = list(responses), [], Usage()

    def complete(self, prompt: str, **kw) -> str:
        self.prompts.append(prompt)
        out = self.responses.pop(0) if self.responses else ""
        self.usage.add(len(prompt.split()), len(out.split()))
        return out

    def chat(self, messages, tools=None, **kw) -> dict:
        return {"content": self.complete(messages[-1].get("content") or ""), "tool_calls": []}


# Memory
# Short-term memory keeps the prompt small; long-term memory lets the agent recall facts from long ago.

def _n_tokens(text: str) -> int:
    """Rough token count (English averages about 0.75 words per token). Use the real tokenizer in production."""
    return int(len(text.split()) / 0.75) + 1


class ConversationMemory:
    """
    Short-term memory with a token budget. Recent turns are kept word for word; when the window
    overflows, the oldest turns are folded into a running summary with one LLM call. That keeps the
    context bounded however long the conversation gets, and it is roughly what chat products do when
    "the model remembers" what you said earlier.
    """

    def __init__(self, llm=None, max_tokens: int = 300, keep_last: int = 4):
        self.llm, self.max_tokens, self.keep_last = llm, max_tokens, keep_last
        self.turns: list[dict] = []
        self.summary = ""
        self.n_summarizations = 0

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self._compact()

    def tokens(self) -> int:
        return _n_tokens(self.summary) + sum(_n_tokens(t["content"]) for t in self.turns)

    def _compact(self) -> None:
        while self.tokens() > self.max_tokens and len(self.turns) > self.keep_last:
            old = self.turns[: len(self.turns) - self.keep_last]
            self.turns = self.turns[len(self.turns) - self.keep_last:]
            text = " ".join([self.summary] + [f"{t['role']} said: {t['content'].rstrip('.')}." for t in old]).strip()
            if self.llm is not None:
                summary = self.llm.complete(f"Summarize the conversation so far in 2 sentences, keeping names, numbers and decisions: {text}")
            else:
                summary = text                                     # without an LLM we keep the raw text and let the cap trim it
            # Whatever the summarizer returns, the summary may use at most half of the token budget.
            max_words = max(8, int(self.max_tokens * 0.75 / 2))
            words = summary.split()
            self.summary = " ".join(words[-max_words:]) if len(words) > max_words else summary
            self.n_summarizations += 1

    def messages(self, system: str = "You are a helpful assistant.") -> list[dict]:
        """What actually goes to the model: the system prompt, a summary of older turns, then the recent turns."""
        msgs = [{"role": "system", "content": system}]
        if self.summary:
            msgs.append({"role": "system", "content": f"Summary of earlier conversation: {self.summary}"})
        return msgs + self.turns


class VectorMemory:
    """
    Long-term memory: store facts and episodes as embeddings and recall the most relevant ones for a
    new query. In other words, RAG over the agent's own history. It uses the offline HashedTfidfEmbedder
    by default, but any embedder with fit/encode will do. In production you would use a vector DB
    (pgvector, Qdrant, ...) plus write policies that decide what is worth remembering, deduplicate and
    expire entries; if you write everything down, recall gets noisy.
    """

    def __init__(self, embedder=None, dedup_threshold: float = 0.95):
        from .rag import HashedTfidfEmbedder
        self.embedder = embedder or HashedTfidfEmbedder(dim=1024)
        self.items: list[dict] = []
        self.dedup_threshold = dedup_threshold
        self._matrix: np.ndarray | None = None

    def _reindex(self) -> None:
        texts = [it["text"] for it in self.items]
        self.embedder.fit(texts)
        self._matrix = self.embedder.encode(texts)

    def add(self, text: str, **metadata) -> bool:
        """Return False if a near-duplicate is already stored (deduplication is our one write policy)."""
        if self.items:
            q = self.embedder.encode([text])[0]
            if float((self._matrix @ q).max()) >= self.dedup_threshold:
                return False
        self.items.append({"text": text, "time": time.time(), **metadata})
        self._reindex()
        return True

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        if not self.items:
            return []
        q = self.embedder.encode([query])[0]
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.items[i]) for i in order]


# Multi-agent routing with a supervisor

@dataclass
class Worker:
    name: str
    description: str
    handle: Callable[[str], str]            # e.g. lambda q: some_agent.run(q).answer


class Supervisor:
    """
    Routes each request to one specialist worker, using one of two strategies:
      "embedding"  cosine similarity between the request and each worker's description. Fast, cheap
                   and deterministic; this is what people call a "semantic router".
      "llm"        ask the LLM to classify the request into one of the worker names. More flexible,
                   but it costs an extra call and can be wrong, so measure it.
    Specialists keep prompts and tool sets small, which makes tool selection more accurate; the price
    is one extra hop. Evaluate routing accuracy on labelled requests, just as you would any classifier.
    """

    def __init__(self, workers: list[Worker], llm=None, strategy: str = "embedding"):
        from .rag import HashedTfidfEmbedder
        self.workers = {w.name: w for w in workers}
        self.llm, self.strategy = llm, strategy
        self.embedder = HashedTfidfEmbedder(dim=1024).fit([w.description for w in workers])
        self._desc = self.embedder.encode([f"{w.name} {w.description}" for w in workers])

    def route(self, request: str) -> str:
        names = list(self.workers)
        if self.strategy == "llm" and self.llm is not None:
            label = self.llm.complete(f"Classify the request into one of: {', '.join(names)}.\nText: {request}\nLabel:").strip()
            if label in self.workers:
                return label
        sims = self._desc @ self.embedder.encode([request])[0]
        return names[int(np.argmax(sims))]

    def run(self, request: str) -> dict:
        name = self.route(request)
        return {"worker": name, "answer": self.workers[name].handle(request)}


# Trajectory evaluation

def evaluate_trajectory(called_tools: list[str], expected_tools: list[str]) -> dict:
    """
    Compare the tools an agent actually called with a reference trajectory.
      precision      fraction of calls that were expected (wasted or wrong calls pull it down)
      recall         fraction of expected tools that were called (missed steps pull it down)
      order_correct  whether the expected tools appear in the expected relative order
      exact          whether the two sequences are identical
      extra_calls    how many calls go beyond the reference length (each one costs money and latency)
    """
    called_set, exp_set = set(called_tools), set(expected_tools)
    precision = sum(t in exp_set for t in called_tools) / len(called_tools) if called_tools else float(not expected_tools)
    recall = len(called_set & exp_set) / len(exp_set) if exp_set else 1.0
    it = iter(called_tools)
    order_correct = all(any(c == e for c in it) for e in expected_tools)   # is `expected` a subsequence of `called`?
    return {"precision": precision, "recall": recall, "order_correct": order_correct,
            "exact": called_tools == expected_tools, "extra_calls": max(0, len(called_tools) - len(expected_tools))}


def trajectory_report(results: list[dict]) -> dict:
    """Average a list of evaluate_trajectory outputs (plus optional 'success' booleans) into one report."""
    if not results:
        return {}
    keys = ["precision", "recall", "order_correct", "exact", "extra_calls"] + (["success"] if "success" in results[0] else [])
    return {k: sum(float(r[k]) for r in results) / len(results) for k in keys}
