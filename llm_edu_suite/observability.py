"""
Observability for LLM applications, built from scratch: traces, metrics, costs and evaluation scores.

Why it matters
--------------
An LLM app is a distributed system whose most important component is non-deterministic, slow, priced
per token, and fails *silently*: a wrong answer still comes back as HTTP 200. You cannot debug,
optimize or trust such a system without seeing every step of every request. Which prompt was sent?
Which documents were retrieved? Which tool was called, with which arguments? How long did each step
take, how many tokens did it cost, and how good was the final answer?

We use the industry-standard model from OpenTelemetry, which LangSmith, Langfuse, Arize Phoenix,
Datadog LLM Observability and friends all follow:

    trace  = one end-to-end request (for example, one user question)
    span   = one unit of work inside it (an LLM call, a retrieval, a tool call), with a start and
             end time, attributes, events, a status and a parent. The parent links turn the spans
             of a trace into a tree:

    agent.run ─┬─ llm.chat          1.2 s   812 in / 40 out tokens   $0.0005
               ├─ tool.search_docs  0.1 s
               `- llm.chat          0.9 s   1030 in / 85 out tokens  $0.0007

Attribute names follow the OpenTelemetry *GenAI semantic conventions* (`gen_ai.request.model`,
`gen_ai.usage.input_tokens`, ...), so the same data could be exported to any OTel backend.

Components
----------
Span, Tracer             create nested spans with a context manager or a decorator (built on contextvars)
TracedLLM                wraps any client with .complete/.chat and records one span per call, with tokens, latency and cost
instrument_registry      wraps every tool of an agents.ToolRegistry so each tool call gets its own span
redact_pii               strips e-mails, phone numbers, card numbers and API keys before anything is logged
MetricsAggregator        latency percentiles (p50/p95/p99), error rate, tokens and cost, per span kind or name
check_slos               compares metrics against service-level objectives and lists the violations
Tracer.score             attaches evaluation scores (user feedback, LLM-judge, heuristics) to a trace
exporters                a console tree, a JSON Lines file, and OTLP-like dicts

In production you would reach for the OpenTelemetry SDK with `opentelemetry-instrumentation-openai`,
OpenLLMetry, or the Langfuse / LangSmith / Phoenix SDKs (`@observe`, `@traceable`), plus Prometheus and
Grafana for metrics.
"""

from __future__ import annotations

import contextvars
import functools
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# Spans and traces
_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("current_span", default=None)


def _new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


@dataclass
class Span:
    """One unit of work. `kind` is a coarse category (llm, tool, retrieval, agent, chain) that metrics group by."""
    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start: float                                      # seconds, as returned by time.time()
    end: float | None = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)        # [{"name", "time", "attributes"}]
    status: str = "ok"                                # ok | error
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return 1000 * ((self.end or time.time()) - self.start)

    def set(self, **attrs) -> "Span":
        """Add attributes. Names with dots work through dict unpacking: span.set(**{"gen_ai.request.model": m})."""
        self.attributes.update(attrs)
        return self

    def add_event(self, name: str, **attrs) -> None:
        """Record a timestamped moment inside the span, such as 'retry', 'cache_hit' or 'guardrail_blocked'."""
        self.events.append({"name": name, "time": time.time(), "attributes": attrs})

    def record_error(self, exc: BaseException) -> None:
        self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        self.add_event("exception", type=type(exc).__name__, message=str(exc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_ms"] = round(self.duration_ms, 3)
        return d


class Tracer:
    """
    Collects spans. Nesting is automatic: a span started while another one is active becomes its
    child. Because this is built on contextvars, it stays correct across threads and async tasks.

        tracer = Tracer()
        with tracer.span("answer_question", kind="chain", question=q):
            with tracer.span("retrieve", kind="retrieval") as s:
                docs = retriever.retrieve(q); s.set(n_docs=len(docs))
            answer = traced_llm.complete(prompt)          # TracedLLM opens its own child span here

    A sample_rate below 1 gives you *head sampling*: the keep-or-drop decision is made once at the
    root, and dropped traces are discarded entirely. High-traffic systems often keep around 10 % of
    traces this way, and use tail sampling to make sure every error is kept.
    """

    def __init__(self, service_name: str = "llm-app", sample_rate: float = 1.0, redact: bool = True, seed: int = 0):
        import random
        self.service_name, self.sample_rate, self.redact = service_name, sample_rate, redact
        self.spans: list[Span] = []
        self.scores: list[dict] = []
        self._rng = random.Random(seed)
        self._dropped_traces: set[str] = set()

    # Creating spans
    def span(self, name: str, kind: str = "chain", **attributes) -> "_SpanContext":
        return _SpanContext(self, name, kind, attributes)

    def trace(self, name: str | None = None, kind: str = "chain"):
        """Decorator that turns every call of the function into a span, named after the function by default."""
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.span(name or fn.__name__, kind=kind):
                    return fn(*args, **kwargs)
            return wrapper
        return deco

    def _start(self, name: str, kind: str, attributes: dict) -> Span:
        parent = _current_span.get()
        if parent is None:                                         # a root span starts a new trace, and that is where we sample
            trace_id = _new_id(32)
            if self._rng.random() >= self.sample_rate:
                self._dropped_traces.add(trace_id)
        else:
            trace_id = parent.trace_id
        return Span(name, kind, trace_id, _new_id(), parent.span_id if parent else None, time.time(), attributes=dict(attributes))

    def _finish(self, span: Span) -> None:
        span.end = time.time()
        if span.trace_id not in self._dropped_traces:
            if self.redact:
                span.attributes = {k: redact_pii(v) if isinstance(v, str) else v for k, v in span.attributes.items()}
            self.spans.append(span)

    # Evaluation scores (online evaluation and user feedback)
    def score(self, trace_id: str, name: str, value: float, comment: str = "", source: str = "heuristic") -> None:
        """Attach a quality score to a whole trace, e.g. a thumbs up/down, an LLM-judge rating or a groundedness check."""
        self.scores.append({"trace_id": trace_id, "name": name, "value": float(value), "comment": comment, "source": source})

    # Queries
    def traces(self) -> dict[str, list[Span]]:
        out: dict[str, list[Span]] = {}
        for s in self.spans:
            out.setdefault(s.trace_id, []).append(s)
        return out

    def last_trace_id(self) -> str | None:
        roots = [s for s in self.spans if s.parent_id is None]
        return roots[-1].trace_id if roots else None

    def trace_cost(self, trace_id: str) -> float:
        return sum(s.attributes.get("llm.cost_usd", 0.0) for s in self.spans if s.trace_id == trace_id)

    def trace_tokens(self, trace_id: str) -> tuple[int, int]:
        sp = [s for s in self.spans if s.trace_id == trace_id]
        return (sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in sp),
                sum(s.attributes.get("gen_ai.usage.output_tokens", 0) for s in sp))

    def clear(self) -> None:
        self.spans.clear(); self.scores.clear(); self._dropped_traces.clear()

    # Exporters
    def print_trace(self, trace_id: str | None = None, show_attributes: tuple[str, ...] = ("gen_ai.request.model", "tool.name")) -> None:
        """Console exporter that prints the span tree with durations, tokens and cost, the same view every LLM-ops UI gives you."""
        trace_id = trace_id or self.last_trace_id()
        spans = [s for s in self.spans if s.trace_id == trace_id]
        children: dict[str | None, list[Span]] = {}
        for s in spans:
            children.setdefault(s.parent_id, []).append(s)

        def walk(parent_id, depth):
            for s in sorted(children.get(parent_id, []), key=lambda x: x.start):
                a = s.attributes
                extra = []
                if "gen_ai.usage.input_tokens" in a:
                    extra.append(f"{a['gen_ai.usage.input_tokens']} in / {a.get('gen_ai.usage.output_tokens', 0)} out tok")
                if a.get("llm.cost_usd"):
                    extra.append(f"${a['llm.cost_usd']:.5f}")
                extra += [f"{k}={a[k]}" for k in show_attributes if k in a]
                flag = "  ERROR " + (s.error or "") if s.status == "error" else ""
                print(f"{'  ' * depth}{'`- ' if depth else ''}{s.name:<{max(1, 28 - 2 * depth)}} {s.kind:<9} {s.duration_ms:8.1f} ms  "
                      + "  ".join(extra) + flag)
                walk(s.span_id, depth + 1)
        print(f"trace {trace_id[:12]}...  ({len(spans)} spans, ${self.trace_cost(trace_id):.5f})")
        walk(None, 0)

    def export_jsonl(self, path: str) -> int:
        """Write one JSON object per span per line, the format that log pipelines such as Loki, ELK or BigQuery ingest."""
        with open(path, "w", encoding="utf-8") as f:
            for s in self.spans:
                f.write(json.dumps({"service": self.service_name, **s.to_dict()}, default=str) + "\n")
        return len(self.spans)

    def to_otlp(self) -> dict:
        """Return the spans shaped like an OTLP/JSON `ExportTraceServiceRequest`, nested as resourceSpans, then scopeSpans, then spans."""
        def attr(k, v):
            key = "intValue" if isinstance(v, int) and not isinstance(v, bool) else "doubleValue" if isinstance(v, float) else \
                "boolValue" if isinstance(v, bool) else "stringValue"
            return {"key": k, "value": {key: v if key != "stringValue" else str(v)}}
        return {"resourceSpans": [{
            "resource": {"attributes": [attr("service.name", self.service_name)]},
            "scopeSpans": [{"scope": {"name": "llm_edu_suite.observability"}, "spans": [{
                "traceId": s.trace_id, "spanId": s.span_id, "parentSpanId": s.parent_id or "", "name": s.name,
                "startTimeUnixNano": int(s.start * 1e9), "endTimeUnixNano": int((s.end or s.start) * 1e9),
                "attributes": [attr(k, v) for k, v in s.attributes.items()] + [attr("span.kind", s.kind)],
                "status": {"code": 2 if s.status == "error" else 1, "message": s.error or ""},
            } for s in self.spans]}],
        }]}


class _SpanContext:
    """The context manager returned by Tracer.span(). It opens the span, makes it current and closes it on exit."""

    def __init__(self, tracer: Tracer, name: str, kind: str, attributes: dict):
        self.tracer, self.name, self.kind, self.attributes = tracer, name, kind, attributes

    def __enter__(self) -> Span:
        self.span = self.tracer._start(self.name, self.kind, self.attributes)
        self._token = _current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.span.record_error(exc)
        _current_span.reset(self._token)
        self.tracer._finish(self.span)
        return False                                                # let the exception propagate; tracing must never hide errors


def current_span() -> Span | None:
    return _current_span.get()


# PII redaction
_PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk|ghp|gho|xox[abp])[-_][A-Za-z0-9_-]{12,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]?\d{3}[ .-]?\d{3,4}(?!\w)")),
]


def redact_pii(text: str) -> str:
    """
    Replace personal data with placeholders before it reaches logs or traces. Prompts and responses
    often contain user data, and storing it raw in an observability backend is both a privacy (GDPR)
    and a security risk. Regexes are only a baseline; production systems add NER-based detectors
    such as Microsoft Presidio.
    """
    for label, pat in _PII_PATTERNS:
        text = pat.sub(f"<{label}>", text)
    return text


# Instrumenting LLM clients and tools

# Prices in USD per 1M tokens, as (input, output). These are illustrative; plug in your provider's current prices.
DEFAULT_PRICES = {"mock": (0.5, 1.5), "small": (0.15, 0.6), "large": (3.0, 15.0)}


def _usage_tuple(llm) -> tuple[int, int]:
    u = getattr(llm, "usage", None)
    return (getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)) if u is not None else (0, 0)


class TracedLLM:
    """
    Wrap any client that exposes .complete(prompt) and/or .chat(messages, tools), such as MockLLM or
    OpenAICompatibleClient. Each call becomes an `llm` span recording the model, token usage (read from
    the client's `usage` counter), latency, cost, a truncated and redacted copy of the prompt and
    response, and the error if the call failed. The wrapper is transparent: it returns exactly what
    the client returns.
    """

    def __init__(self, llm, tracer: Tracer, model: str = "mock", price_per_m: tuple[float, float] | None = None,
                 log_content: bool = True, max_content_chars: int = 500):
        self.llm, self.tracer, self.model = llm, tracer, model
        self.price = price_per_m or DEFAULT_PRICES.get(model, DEFAULT_PRICES["mock"])
        self.log_content, self.max_chars = log_content, max_content_chars
        self.usage = getattr(llm, "usage", None)

    def _record(self, span: Span, before: tuple[int, int], prompt_text: str, output_text: str) -> None:
        p_in, p_out = _usage_tuple(self.llm)
        n_in, n_out = p_in - before[0], p_out - before[1]
        cost = n_in / 1e6 * self.price[0] + n_out / 1e6 * self.price[1]
        span.set(**{"gen_ai.system": type(self.llm).__name__, "gen_ai.request.model": self.model,
                    "gen_ai.usage.input_tokens": n_in, "gen_ai.usage.output_tokens": n_out, "llm.cost_usd": cost})
        if self.log_content:
            span.set(**{"gen_ai.prompt": prompt_text[-self.max_chars:], "gen_ai.completion": (output_text or "")[: self.max_chars]})

    def complete(self, prompt: str, **kw) -> str:
        with self.tracer.span("llm.complete", kind="llm", **{"gen_ai.operation.name": "text_completion"}) as span:
            before = _usage_tuple(self.llm)
            out = self.llm.complete(prompt, **kw)
            self._record(span, before, prompt, out)
            return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> dict:
        with self.tracer.span("llm.chat", kind="llm", **{"gen_ai.operation.name": "chat"}) as span:
            before = _usage_tuple(self.llm)
            reply = self.llm.chat(messages, tools=tools, **kw)
            calls = reply.get("tool_calls") or []
            span.set(**{"llm.tool_calls": ",".join(c["name"] for c in calls), "llm.n_messages": len(messages)})
            self._record(span, before, json.dumps(messages[-1], default=str), reply.get("content") or json.dumps(calls))
            return reply


def instrument_registry(registry, tracer: Tracer):
    """Wrap every tool function of an agents.ToolRegistry so that each execution produces a `tool` span."""
    for tool in registry.tools.values():
        if getattr(tool.fn, "_traced", False):
            continue
        original = tool.fn

        def make(fn, name):
            @functools.wraps(fn)
            def traced(**kwargs):
                with tracer.span(f"tool.{name}", kind="tool", **{"tool.name": name, "tool.arguments": json.dumps(kwargs, default=str)[:300]}) as s:
                    result = fn(**kwargs)
                    s.set(**{"tool.result_chars": len(str(result))})
                    return result
            traced._traced = True
            return traced
        tool.fn = make(original, tool.name)
    return registry


# Metrics and SLOs
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, with q in [0, 100].

    Latency is reported as p50 / p95 / p99 rather than as a mean, because the tail is what users
    actually feel and what ends up breaking timeouts.
    """
    if not values:
        return float("nan")
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


class MetricsAggregator:
    """Turn raw spans into dashboard numbers: for each group, the count, error rate, latency percentiles, tokens and cost."""

    def __init__(self, spans: list[Span]):
        self.spans = spans

    def by(self, key: str = "kind") -> dict[str, dict]:
        groups: dict[str, list[Span]] = {}
        for s in self.spans:
            g = getattr(s, key) if key in ("kind", "name", "status") else s.attributes.get(key, "-")
            groups.setdefault(str(g), []).append(s)
        return {g: self._summary(sp) for g, sp in groups.items()}

    def requests(self) -> dict:
        """Metrics per *request*, i.e. per root span: end-to-end latency, error rate, and cost and tokens per request."""
        roots = [s for s in self.spans if s.parent_id is None]
        cost = {s.trace_id: 0.0 for s in roots}
        toks = {s.trace_id: 0 for s in roots}
        for s in self.spans:
            if s.trace_id in cost:
                cost[s.trace_id] += s.attributes.get("llm.cost_usd", 0.0)
                toks[s.trace_id] += s.attributes.get("gen_ai.usage.input_tokens", 0) + s.attributes.get("gen_ai.usage.output_tokens", 0)
        out = self._summary(roots)
        out["cost_per_request_usd"] = sum(cost.values()) / max(1, len(roots))
        out["tokens_per_request"] = sum(toks.values()) / max(1, len(roots))
        return out

    @staticmethod
    def _summary(spans: list[Span]) -> dict:
        lat = [s.duration_ms for s in spans]
        n = len(spans)
        return {"count": n, "error_rate": sum(s.status == "error" for s in spans) / max(1, n),
                "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95), "p99_ms": percentile(lat, 99),
                "input_tokens": sum(s.attributes.get("gen_ai.usage.input_tokens", 0) for s in spans),
                "output_tokens": sum(s.attributes.get("gen_ai.usage.output_tokens", 0) for s in spans),
                "cost_usd": sum(s.attributes.get("llm.cost_usd", 0.0) for s in spans)}


def check_slos(metrics: dict, slos: dict) -> list[str]:
    """
    Compare metrics against SLOs such as {"p95_ms": 2000, "error_rate": 0.01, "cost_per_request_usd": 0.002}
    and return the violations as human-readable strings.
    In production these would be alert rules (Prometheus/Grafana, Datadog monitors) evaluated over a time window.
    """
    violations = []
    for key, limit in slos.items():
        value = metrics.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)) and value > limit:
            violations.append(f"SLO violated: {key} = {value:.4g} > {limit:.4g}")
    return violations
