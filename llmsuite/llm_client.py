"""
A minimal LLM client abstraction, so that application code (RAG, agents, evaluation) never has to
know which provider is behind it. Every client offers the same two calls:

    llm.complete(prompt) -> str
    llm.chat(messages, tools=None) -> dict  {"content": str | None, "tool_calls": [{"name", "arguments"}]}

Implementations
---------------
OpenAICompatibleClient   talks to any OpenAI-compatible endpoint: vLLM, SGLang, Ollama, OpenAI,
                         OpenRouter and so on. It only needs `requests`, no vendor SDK.
LocalTransformerClient   wraps a llmsuite TransformerLM and its tokenizer for offline text completion.
MockLLM                  a deterministic, rule-based stand-in so the notebooks run without any model
                         or API key. It "understands" a few task shapes (extraction, classification,
                         judging, tool selection) just well enough to exercise the code around it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, p: int, c: int):
        self.prompt_tokens += p
        self.completion_tokens += c
        self.calls += 1

    def cost(self, usd_per_m_in: float = 0.5, usd_per_m_out: float = 1.5) -> float:
        return self.prompt_tokens / 1e6 * usd_per_m_in + self.completion_tokens / 1e6 * usd_per_m_out


class OpenAICompatibleClient:
    """
    Talks to /v1/chat/completions. A couple of servers you can point it at:
        vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
        ollama serve   (base_url="http://localhost:11434/v1")
    """

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512, timeout: float = 120.0):
        import requests  # imported here because it is an optional dependency
        self._requests = requests
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.usage = Usage()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, response_format: dict | None = None, **kw) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": kw.get("temperature", self.temperature),
                   "max_tokens": kw.get("max_tokens", self.max_tokens)}
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        r = self._requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout,
                                headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        u = data.get("usage", {})
        self.usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        calls = [{"id": tc.get("id"), "name": tc["function"]["name"],
                  "arguments": json.loads(tc["function"]["arguments"] or "{}")} for tc in msg.get("tool_calls") or []]
        return {"content": msg.get("content"), "tool_calls": calls}

    def complete(self, prompt: str, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kw)["content"] or ""


class LocalTransformerClient:
    """Offline completion with a llmsuite TransformerLM and ByteBPETokenizer. It does not support tools."""

    def __init__(self, model, tokenizer, device: str = "cpu", max_new_tokens: int = 64, temperature: float = 0.8, top_k: int = 50):
        import torch
        self._torch = torch
        self.model, self.tok, self.device = model.to(device), tokenizer, device
        self.max_new_tokens, self.temperature, self.top_k = max_new_tokens, temperature, top_k
        self.usage = Usage()

    def complete(self, prompt: str, **kw) -> str:
        ids = self._torch.tensor([self.tok.encode(prompt)], device=self.device)
        out = self.model.generate(ids, kw.get("max_new_tokens", self.max_new_tokens), temperature=kw.get("temperature", self.temperature),
                                  top_k=self.top_k, eos_id=self.tok.eos_id)
        new = out[0, ids.shape[1]:].tolist()
        self.usage.add(ids.shape[1], len(new))
        return self.tok.decode([t for t in new if t != self.tok.eos_id])

    def chat(self, messages, tools=None, **kw):
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return {"content": self.complete(prompt, **kw), "tool_calls": []}


class MockLLM:
    """
    A deterministic fake LLM for offline notebooks and unit tests. It recognizes a handful of prompt
    shapes and answers each with a simple heuristic:
      - pairwise judging ("Reply with exactly one word: A, B, or TIE"): keyword overlap with the question
      - rubric scoring ("Reply with only the integer score"): also keyword overlap
      - JSON extraction ("Return JSON" plus a schema): regex-based field extraction
      - classification ("Classify ... into: x, y, z"): keyword matching
      - tool selection (tools passed to chat()): picks a tool by keyword, otherwise answers directly
      - RAG answers ("Context:" ... "Question:"): the best-matching context sentence, cited as [n]
      - planning and synthesis prompts from agentic.py: splits the task on "then"/"and" and joins step results
    Anything else gets a short canned answer. Keep your expectations modest; it is a test double.
    """

    def __init__(self, seed: int = 0):
        self.usage = Usage()
        self.seed = seed

    # Helpers
    @staticmethod
    def _overlap(a: str, b: str) -> int:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        return len(ta & tb)

    def _count(self, prompt: str, out: str):
        self.usage.add(len(prompt.split()), len(out.split()))

    # Public API
    def complete(self, prompt: str, **kw) -> str:
        out = self._complete(prompt)
        self._count(prompt, out)
        return out

    def _complete(self, prompt: str) -> str:
        p = prompt
        if "Reply with exactly one word: A, B, or TIE" in p:
            q = re.search(r"Question: (.*?)\n", p, re.S)
            a = re.search(r"Answer A:\n(.*?)\n\nAnswer B:", p, re.S)
            b = re.search(r"Answer B:\n(.*?)\n\nWhich", p, re.S)
            if q and a and b:
                sa, sb = self._overlap(q.group(1), a.group(1)), self._overlap(q.group(1), b.group(1))
                if sa == sb:
                    return "TIE"
                return "A" if sa > sb else "B"
            return "TIE"
        if "Reply with only the integer score" in p:
            q = re.search(r"Question: (.*?)\n", p, re.S)
            a = re.search(r"Answer: (.*?)\n", p, re.S)
            ov = self._overlap(q.group(1), a.group(1)) if q and a else 0
            return str(max(1, min(5, 1 + ov)))
        if "Context:" in p and "Question:" in p:
            q = p.split("Question:")[-1].split("\n")[0]
            ctx = p.split("Context:")[1].split("Question:")[0]
            best, best_s = "I don't know.", 0
            for line in ctx.strip().splitlines():
                m = re.match(r"\[(\d+)\] (?:\([^)]*\) )?(.*)", line.strip())
                if not m:
                    continue
                for sent in re.split(r"(?<=[.!?])\s+", m.group(2)):
                    s = self._overlap(q, sent)
                    if s > best_s:
                        best, best_s = f"{sent.strip()} [{m.group(1)}]", s
            return best
        if "Break the following task into" in p:                       # the planner prompt of agentic.PlanAndExecute
            task = p.split("Task:", 1)[-1].split("\nSteps:")[0].strip().rstrip(".")
            parts = [s.strip(" .,") for s in re.split(r",? and then |,? then |; |, and (?=[a-z])", task) if s.strip(" .,")]
            return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(parts))
        if "Combine the step results into one final answer" in p:       # ...and its synthesizer prompt
            res = p.split("Step results:", 1)[-1].split("Final answer:")[0].strip().splitlines()
            vals = [r.split("->", 1)[-1].replace("Based on the tool result:", "").strip() for r in res if "->" in r]
            return " | ".join(vals)
        if "Rank the following candidate items" in p:
            hist = p.split("recently watched:")[-1].split("Rank the following")[0]
            cands = re.findall(r"^(\d+)\. (.*)$", p.split("relevant for this user:")[-1], re.M)
            ranked = sorted(cands, key=lambda c: -self._overlap(hist, c[1]))
            return ", ".join(n for n, _ in ranked)
        if "In one sentence, explain why" in p:
            m = re.search(r"enjoy '(.*?)'", p)
            return f"Because it shares the tone and themes of what they recently liked, {m.group(1) if m else 'this item'} is a natural next pick."
        if "Return JSON" in p or "return JSON" in p:
            return json.dumps(self._extract_json(p))
        m = re.search(r"[Cc]lassify.*?into[^:]*:\s*([^\n.]+)", p)
        if m:
            labels = [l.strip(" .") for l in re.split(r",|/| or ", m.group(1)) if l.strip()]
            text = (p.split("Text:")[-1] if "Text:" in p else p).lower()
            # A tiny lexicon so sentiment-style demos behave sensibly; everything else falls back to keyword overlap.
            lex = {"positive": ["great", "love", "excellent", "good", "amazing", "perfect", "happy", "recommend"],
                   "negative": ["terrible", "awful", "bad", "worst", "broken", "died", "waste", "hate", "poor", "refund"],
                   "injection": ["ignore all previous", "ignore previous", "disregard the", "reveal the system prompt",
                                 "you are now", "jailbreak", "do not tell the user", "begin your answer with"]}
            def score(label):
                s = self._overlap(label, text)
                s += sum(w in text for w in lex.get(label.lower(), []))
                return s
            scored = sorted(labels, key=lambda l: -score(l))
            if scored and score(scored[0]) == 0:                     # nothing matched, so prefer a "default" label if there is one
                for fallback in ("neutral", "benign", "other", "none", "unknown"):
                    if fallback in [l.lower() for l in labels]:
                        return next(l for l in labels if l.lower() == fallback)
            return scored[0] if scored else ""
        if "search queries" in p.lower():
            m = re.search(r"intent as: '(.*?)'", p)
            base = m.group(1) if m else "query"
            return "\n".join([base, f"what is {base}", f"{base} explained"])
        if "summarize" in p.lower() or "summary" in p.lower():
            text = p.split(":", 1)[-1].strip()
            sents = re.split(r"(?<=[.!?])\s+", text)
            return " ".join(sents[:2])
        return "This is a mock answer. Connect a real model with OpenAICompatibleClient for real outputs."

    @staticmethod
    def _extract_json(prompt: str) -> dict:
        """A very small heuristic extractor for emails, prices, dates, ids, names and quantities."""
        text = prompt.split("Text:")[-1] if "Text:" in prompt else prompt
        out = {}
        if (m := re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)):
            out["email"] = m.group()
        if (m := re.search(r"\$\s?([\d,]+(?:\.\d+)?)", text)):
            out["amount"] = float(m.group(1).replace(",", ""))
        if (m := re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", text)):
            out["date"] = m.group(1)
        if (m := re.search(r"(?:order|invoice|ticket)\s*#?\s*([A-Z0-9-]{4,})", text, re.I)):
            out["id"] = m.group(1)
        if (m := re.search(r"\b(\d+)\s+(?:units|items|pieces|pcs)\b", text, re.I)):
            out["quantity"] = int(m.group(1))
        if (m := re.search(r"(?:from|by|customer|client)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)):
            out["name"] = m.group(1)
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> dict:
        last = messages[-1]
        if last["role"] == "tool":
            out = f"Based on the tool result: {last['content']}"
            self._count(last["content"], out)
            return {"content": out, "tool_calls": []}
        user = last.get("content", "") or ""
        if tools:
            best, best_s = None, 0
            for t in tools:
                fn = t["function"] if "function" in t else t
                s = self._overlap(fn.get("description", "") + " " + fn["name"], user)
                if s > best_s:
                    best, best_s = fn, s
            if best is not None:
                args = {}
                props = (best.get("parameters") or {}).get("properties", {})
                for name, spec in props.items():
                    if "enum" in spec:
                        args[name] = spec["enum"][0]                                     # any valid option will do
                    elif name in ("city", "location", "place"):
                        caps = re.findall(r"\b[A-Z][a-z]+\b", user)
                        args[name] = caps[-1] if caps else user                          # guess the last capitalized word
                    elif spec.get("type") in ("number", "integer"):
                        nums = re.findall(r"-?\d+(?:\.\d+)?", user)
                        if nums:
                            args[name] = float(nums[0]) if spec["type"] == "number" else int(float(nums[0]))
                    elif name in ("expression", "expr"):
                        cands = [c.strip() for c in re.findall(r"[\d\.\s\+\-\*/\(\)]+", user) if any(ch.isdigit() for ch in c)]
                        if cands:
                            args[name] = max(cands, key=len)
                    else:
                        args[name] = user
                self._count(user, json.dumps(args))
                return {"content": None, "tool_calls": [{"id": "call_0", "name": best["name"], "arguments": args}]}
        out = self._complete(user)
        self._count(user, out)
        return {"content": out, "tool_calls": []}
