"""
A "modern" profile for the mini-swe-agent harness of notebook 27: the techniques current coding harnesses
(Claude Code, Codex CLI, SWE-agent, OpenHands) add around the same agent loop. Each one can be switched off,
so the notebook can measure what it buys (an ablation):

    tools     native function calling with dedicated tools (bash, view_file, edit_file, submit) instead of parsing
              a bash block out of free text; edit_file is an exact string replacement (like Claude Code's
              str_replace tool or SWE-agent's editor), so the model never has to escape a sed expression
    lint      edit_file refuses an edit that leaves a Python file with a syntax error (SWE-agent's lint gate)
    loop      repeated-action detection: the same action with the same result three times in a row gets a warning
    verify    a completion gate ("stop hook"): submit is rejected while the visible tests fail, or when no python
              command ran without error after the last edit; at most two rejections, then the submission is accepted
    mask      observation masking: tool outputs older than the last 6 are replaced by a one-line placeholder in
              the prompt (the trajectory on disk keeps everything), a cheap form of context compaction

Used by scripts/harness_run.py with --profile modern. Runs in the harness environment (.venv-harness).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time

import litellm
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

litellm.drop_params = True          # e.g. a seed the provider does not accept is dropped instead of failing the call

ALL_FEATURES = ("tools", "lint", "loop", "verify", "mask")
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


VIEW_TOOL = _tool("view_file", "Show a text file with line numbers.",
                  {"path": {"type": "string", "description": "file path, relative to the repository root"}}, ["path"])
EDIT_TOOL = _tool(
    "edit_file",
    "Edit a file by exact string replacement. old_str must appear exactly once in the file (include enough "
    "surrounding lines to make it unique); it is replaced by new_str. With an empty old_str, the file is created "
    "(or overwritten) with new_str as its whole content.",
    {"path": {"type": "string"}, "old_str": {"type": "string", "description": "exact text to replace, or empty to create the file"},
     "new_str": {"type": "string", "description": "replacement text"}},
    ["path", "old_str", "new_str"])
SUBMIT_TOOL = _tool("submit", "Finish the task. Call it only after you have verified that your change works.",
                    {"summary": {"type": "string", "description": "one or two sentences on what you changed"}}, ["summary"])
TOOLS = [BASH_TOOL, VIEW_TOOL, EDIT_TOOL, SUBMIT_TOOL]

SYSTEM = """You are an autonomous software engineer working inside a small code repository.
You act only through tool calls: bash (each command runs in a fresh shell in the repository root), view_file,
edit_file and submit. Make one tool call per response, after a short explanation of what you are doing and why."""

INSTANCE = """Your task:

{{task}}

Recommended workflow:
1. Explore: list the files and read the relevant ones with view_file.
2. Reproduce: run the existing tests (python -m pytest -q) or a short python -c snippet that shows the problem.
3. Fix: change the code with edit_file (exact old_str -> new_str). Prefer small, targeted edits. Do not edit the tests.
4. Verify: run the tests or your snippet again, including an edge case.
5. Submit with the submit tool, only when you have seen your change work.

System: {{system}}. Working directory: {{cwd}}."""


def _rate_limit():
    """Free API tiers allow a few requests per minute: wait HARNESS_MIN_INTERVAL seconds between calls."""
    gap = float(os.environ.get("HARNESS_MIN_INTERVAL", "0") or 0)
    last = getattr(_rate_limit, "last", 0.0)
    if gap and time.time() - last < gap:
        time.sleep(gap - (time.time() - last))
    _rate_limit.last = time.time()


class PacedTextbasedModel(LitellmTextbasedModel):
    """The baseline text-based model, with the same pacing for free API tiers."""
    def _query(self, messages, **kwargs):
        _rate_limit()
        return super()._query(messages, **kwargs)


class ModernModel(LitellmModel):
    """Native tool calling with the four tools, and observation masking."""

    def __init__(self, *, mask_keep: int | None = 6, **kwargs):
        super().__init__(**kwargs)
        self.mask_keep = mask_keep

    def _query(self, messages, **kwargs):
        _rate_limit()
        return litellm.completion(model=self.config.model_name, messages=messages, tools=TOOLS,
                                  **(self.config.model_kwargs | kwargs))

    def _prepare_messages_for_api(self, messages):
        prepared = super()._prepare_messages_for_api(messages)
        if self.mask_keep is None:
            return prepared
        obs = [i for i, m in enumerate(prepared) if m.get("role") == "tool"]
        for i in obs[:-self.mask_keep] if len(obs) > self.mask_keep else []:
            c = str(prepared[i].get("content", ""))
            if len(c) > 300:
                prepared[i] = {**prepared[i], "content": f"[output of an earlier step omitted to save context ({len(c)} characters)]"}
        return prepared

    def _parse_actions(self, response):
        calls = response.choices[0].message.tool_calls or []
        if not calls:
            raise FormatError({"role": "user", "content": "No tool call found. Every response must call exactly one tool "
                               "(bash, view_file, edit_file or submit).", "extra": {"interrupt_type": "FormatError"}})
        actions = []
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except Exception as e:
                args, name = {"error": f"arguments are not valid JSON: {e}"}, "invalid"
            if name not in {"bash", "view_file", "edit_file", "submit"}:
                name = "invalid"
                args = {"error": f"unknown tool '{c.function.name}'", **(args if isinstance(args, dict) else {})}
            summary = args.get("command") if name == "bash" else f"{name} {args.get('path', '')}".strip()
            actions.append({"tool": name, "args": args, "command": str(summary), "tool_call_id": c.id})
        return actions


class ModernEnvironmentMixin:
    """Tool dispatch, lint gate, loop detection, verification gate and head+tail truncation.
    Mixed into SandboxedBashEnvironment (harness_run.py) so the permission policy and scrubbed env still apply."""

    def setup_modern(self, features):
        self.features, self.history, self.rejections = set(features), [], 0
        self.edited, self.ran_since_edit = False, False

    # --- helpers -------------------------------------------------------------------------------------------
    def _path(self, p):
        full = os.path.abspath(os.path.join(self.cwd, p))
        if not full.startswith(os.path.abspath(self.cwd)):
            raise ValueError("paths must stay inside the repository")
        return full

    @staticmethod
    def _truncate(text, head=3000, tail=3000):
        if len(text) <= head + tail:
            return text
        return text[:head] + f"\n[... {len(text) - head - tail} characters omitted ...]\n" + text[-tail:]

    def _bash(self, command):
        try:
            return super().execute({"command": command})
        except Submitted:                                        # the text-protocol marker, echoed through bash
            return self._submit("submitted with the echo marker")

    # --- tools ---------------------------------------------------------------------------------------------
    def _view(self, a):
        p = self._path(a["path"])
        if not os.path.isfile(p):
            return {"output": f"No such file: {a['path']}", "returncode": 1, "exception_info": ""}
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        return {"output": "\n".join(f"{i:4d}  {l}" for i, l in enumerate(lines, 1)) or "(empty file)", "returncode": 0, "exception_info": ""}

    def _edit(self, a):
        p, old, new = self._path(a["path"]), a.get("old_str", ""), a.get("new_str", "")
        if old:
            if not os.path.isfile(p):
                return {"output": f"No such file: {a['path']}. To create it, call edit_file with an empty old_str.", "returncode": 1, "exception_info": ""}
            text = open(p, encoding="utf-8").read()
            n = text.count(old)
            if n != 1:
                hint = "not found (check spaces, indentation and quotes; view_file shows the exact text)" if n == 0 else \
                       f"found {n} times; include more surrounding lines so it is unique"
                return {"output": f"edit_file failed: old_str {hint}. The file was not changed.", "returncode": 1, "exception_info": ""}
            result = text.replace(old, new, 1)
        else:
            result = new
        if "lint" in self.features and p.endswith(".py"):
            try:
                compile(result, a["path"], "exec")
            except SyntaxError as e:
                return {"output": f"edit_file rejected: the edit would leave a syntax error ({e.msg}, line {e.lineno}). "
                                  "The file was not changed.", "returncode": 1, "exception_info": ""}
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        open(p, "w", encoding="utf-8", newline="\n").write(result)
        self.edited, self.ran_since_edit = True, False
        start = result[:result.find(new)].count("\n") if new and new in result else 0
        first = max(0, start - 3)                                  # show the edit with 3 lines of context, numbered
        snippet = result.splitlines()[first: start + new.count("\n") + 4]
        return {"output": f"Edited {a['path']}. The changed region now reads:\n" +
                          "\n".join(f"{first + 1 + i:4d}  {l}" for i, l in enumerate(snippet)), "returncode": 0, "exception_info": ""}

    def _submit(self, summary):
        if "verify" in self.features and self.rejections < 2:
            tests = [t for t in glob.glob(os.path.join(self.cwd, "**", "test_*.py"), recursive=True)]
            if tests:
                r = super().execute({"command": "python -m pytest -q -p no:cacheprovider 2>&1 | tail -n 25"})
                if "failed" in r["output"] or "error" in r["output"].lower():
                    self.rejections += 1
                    return {"output": "[harness] Submission rejected: the tests still fail. Fix the code, then submit again.\n"
                                      + r["output"], "returncode": 1, "exception_info": ""}
            elif self.edited and not self.ran_since_edit:
                self.rejections += 1
                return {"output": "[harness] Submission rejected: no code ran successfully since your last edit. Run a short "
                                  "check that imports and exercises your change (for example python -c \"from module import f; "
                                  "print(f(...))\") and look at the result, then submit again.",
                        "returncode": 1, "exception_info": ""}
        raise Submitted({"role": "exit", "content": summary, "extra": {"exit_status": "Submitted", "submission": summary}})

    # --- the Environment interface -------------------------------------------------------------------------
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        tool, args = action.get("tool", "bash"), action.get("args") or {"command": action.get("command", "")}
        if tool == "bash":
            out = self._bash(args.get("command", ""))
            cmd, rc, text = args.get("command", ""), out.get("returncode"), out.get("output", "")
            # A check counts as verification when Python ran and showed no error: no traceback, no failed tests.
            # Requiring exit code 0 from the whole command gave false rejections, found by reading trajectories:
            # `python -c ...; python -m pytest -q` exits 5 in a folder without tests ("no tests ran"), and a final
            # `grep` that finds nothing (here, the old name after a rename: the good outcome) exits 1.
            clean = "Traceback (most recent call last)" not in text and not re.search(r"\b\d+ (failed|errors?)\b", text)
            if ("python" in cmd or "pytest" in cmd) and rc != -1 and clean:
                self.ran_since_edit = True
        elif tool == "view_file":
            out = self._view(args)
        elif tool == "edit_file":
            try:
                out = self._edit(args)
            except (ValueError, OSError) as e:
                out = {"output": f"edit_file failed: {e}", "returncode": 1, "exception_info": ""}
        elif tool == "submit":
            out = self._submit(str(args.get("summary", "")))
        else:
            out = {"output": f"Invalid tool call: {args.get('error', 'unknown')}. Use bash, view_file, edit_file or submit.",
                   "returncode": 1, "exception_info": ""}
        out["output"] = self._truncate(out.get("output", ""))
        if "loop" in self.features:
            sig = (tool, json.dumps(args, sort_keys=True), out["output"][:2000], out.get("returncode"))
            self.history.append(sig)
            if len(self.history) >= 3 and len(set(self.history[-3:])) == 1:
                out["output"] = ("[harness] You have made this exact call three times in a row with the same result. "
                                 "It will not work differently: read the error, look at the file again, and try another approach.\n"
                                 + out["output"])
        return out
