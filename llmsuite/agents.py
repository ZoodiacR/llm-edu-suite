"""
Tool use and agents: the loop around the model.

The model is told about each tool through a JSON schema. On every turn it either answers or asks for
one or more tool calls; we validate and run those calls (with a time limit), feed the results back as
messages, and go around again until it answers.

Components
----------
Tool / ToolRegistry      register Python functions with a JSON-schema signature and render them for the API
validate_args            check arguments against the schema (types, required fields) before running anything
Agent                    the loop itself, with a step limit, a per-step trace and error feedback to the model
parse_json_output        forgiving JSON extraction for structured output (structured_output adds
                         schema validation and a retry prompt on top)
"""

from __future__ import annotations

import contextvars
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable


# Tools

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                       # JSON schema: {"type": "object", "properties": {...}, "required": [...]}
    fn: Callable[..., Any]
    timeout_s: float = 10.0

    def to_openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


def validate_args(args: dict, schema: dict) -> list[str]:
    """Return a list of problems; an empty list means the arguments are valid.

    Only a small subset of JSON Schema is supported: required fields, basic types and enums.
    """
    problems = []
    props = schema.get("properties", {})
    for r in schema.get("required", []):
        if r not in args:
            problems.append(f"missing required argument '{r}'")
    for k, v in args.items():
        if k not in props:
            problems.append(f"unexpected argument '{k}'")
            continue
        t = props[k].get("type")
        if t and t in _JSON_TYPES and not isinstance(v, _JSON_TYPES[t]):
            if not (t == "number" and isinstance(v, (int, float))):
                problems.append(f"argument '{k}' should be {t}, got {type(v).__name__}")
        if "enum" in props[k] and v not in props[k]["enum"]:
            problems.append(f"argument '{k}' must be one of {props[k]['enum']}")
    return problems


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict, timeout_s: float = 10.0):
        """Decorator: @registry.register("add", "Add two numbers", {...})"""
        def deco(fn):
            self.tools[name] = Tool(name, description, parameters, fn, timeout_s)
            return fn
        return deco

    def add(self, tool: Tool):
        self.tools[tool.name] = tool

    def to_openai(self) -> list[dict]:
        return [t.to_openai() for t in self.tools.values()]

    def execute(self, name: str, args: dict) -> str:
        """Validate and run a tool. Always returns a string, errors included, so the model can recover."""
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'. Available: {list(self.tools)}"
        tool = self.tools[name]
        problems = validate_args(args, tool.parameters)
        if problems:
            return "ERROR: invalid arguments: " + "; ".join(problems)
        # Run in a worker thread so a hung tool cannot block the agent loop forever. Python cannot kill a
        # thread, so a timed-out call keeps running in the background; for hard isolation (untrusted code,
        # runaway CPU) production systems run tools in a subprocess or container with OS-level limits.
        pool = ThreadPoolExecutor(max_workers=1)
        # copy_context() carries context variables (e.g. the active tracing span) into the worker thread
        future = pool.submit(contextvars.copy_context().run, tool.fn, **args)
        try:
            result = future.result(timeout=tool.timeout_s)
        except FutureTimeout:
            return f"ERROR: tool '{name}' exceeded {tool.timeout_s}s"
        except Exception as e:  # hand the error back to the model instead of crashing the loop
            return f"ERROR: {type(e).__name__}: {e}"
        finally:
            pool.shutdown(wait=False)
        return result if isinstance(result, str) else json.dumps(result, default=str)


# The agent loop

@dataclass
class AgentStep:
    step: int
    tool: str | None
    arguments: dict | None
    result: str | None
    content: str | None


@dataclass
class AgentRun:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    stopped_reason: str = "answer"


class Agent:
    """
    The basic tool-using loop, in pseudocode:

        while steps < max_steps:
            reply = llm.chat(messages, tools)
            if there are no tool calls: return reply.content
            for each call: append registry.execute(...) as a role="tool" message

    Observations are cut to `max_obs_chars` so a chatty tool cannot blow up the context.
    """

    def __init__(self, llm, registry: ToolRegistry, system_prompt: str = "You are a helpful assistant. Use tools when needed.",
                 max_steps: int = 6, max_obs_chars: int = 2000, confirm: Callable[[str, dict], bool] | None = None):
        self.llm, self.registry, self.system_prompt = llm, registry, system_prompt
        self.max_steps, self.max_obs_chars, self.confirm = max_steps, max_obs_chars, confirm

    def run(self, user_message: str, verbose: bool = False) -> AgentRun:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}]
        steps: list[AgentStep] = []
        tools = self.registry.to_openai()
        for i in range(self.max_steps):
            reply = self.llm.chat(messages, tools=tools)
            if not reply.get("tool_calls"):
                steps.append(AgentStep(i, None, None, None, reply.get("content")))
                return AgentRun(reply.get("content") or "", steps, messages + [{"role": "assistant", "content": reply.get("content")}])
            messages.append({"role": "assistant", "content": reply.get("content"), "tool_calls": reply["tool_calls"]})
            for call in reply["tool_calls"]:
                name, args = call["name"], call.get("arguments") or {}
                if self.confirm is not None and not self.confirm(name, args):      # human-in-the-loop gate
                    result = "ERROR: action not confirmed by the user"
                else:
                    result = self.registry.execute(name, args)
                result = result[: self.max_obs_chars]
                if verbose:
                    print(f"  [step {i}] {name}({json.dumps(args)}) -> {result[:120]}")
                steps.append(AgentStep(i, name, args, result, None))
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "name": name, "content": result})
        return AgentRun("Stopped: maximum number of steps reached.", steps, messages, stopped_reason="max_steps")


# Structured output

def parse_json_output(text: str) -> Any:
    """Pull the first JSON object or array out of model text, tolerating ```json fences and chatter around it."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found")
    depth, opener = 0, text[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(text)):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON")


def structured_output(llm, prompt: str, schema: dict, max_retries: int = 2) -> dict:
    """
    Ask for JSON matching `schema`, validate it, and on failure ask again with the error message
    included. This is the fallback for servers that do not support constrained decoding
    (response_format=json_schema).
    """
    full = f"{prompt}\n\nReturn JSON only, matching this schema:\n{json.dumps(schema)}"
    last_err = ""
    for attempt in range(max_retries + 1):
        text = llm.complete(full if not last_err else f"{full}\n\nYour previous output was invalid: {last_err}. Fix it.")
        try:
            obj = parse_json_output(text)
            if isinstance(obj, dict) and schema.get("additionalProperties", False) is False:
                obj = {k: v for k, v in obj.items() if k in schema.get("properties", {})}   # silently drop fields the schema does not know
            problems = validate_args(obj, schema) if isinstance(obj, dict) else ["top-level JSON must be an object"]
            if not problems:
                return obj
            last_err = "; ".join(problems)
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"could not obtain valid JSON after {max_retries + 1} attempts: {last_err}")
