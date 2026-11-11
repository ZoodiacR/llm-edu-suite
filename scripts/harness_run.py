#!/usr/bin/env python
"""
Run the mini-swe-agent harness on one task folder, with a local model through Ollama (or an API model) and a
sandboxed bash environment. Used by notebook 27; also runnable by hand.

mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) is installed in its own virtual environment,
because its terminal UI needs a newer `rich` than the course environment allows:

    python -m venv .venv-harness
    .venv-harness/Scripts/python -m pip install mini-swe-agent==2.4.6 pytest     # (bin/ instead of Scripts/ on Linux/macOS)

Usage (with the harness venv's python):

    .venv-harness/Scripts/python scripts/harness_run.py TASK_DIR MODEL OUT_JSON [--steps 20] [--agents-md]
        [--profile baseline|modern] [--features tools,lint,loop,verify,mask] [--max-tokens 4096] [--wall 300]

MODEL is an Ollama model ("qwen2.5:7b") or a litellm provider/model id for an API model ("gemini/gemini-3.6-flash",
with GEMINI_API_KEY in the environment; the key never reaches the agent's shell, whose environment is scrubbed).
HARNESS_MIN_INTERVAL=7 spaces the calls, for free tiers with a requests-per-minute limit.

--profile baseline is mini-swe-agent as shipped (text protocol, one bash block per reply). --profile modern adds
the techniques of current harnesses, described in scripts/harness_modern.py; --features keeps only some of them.

TASK_DIR must contain TASK.md (the task given to the agent). With --agents-md, the project's AGENTS.md is added
to the task, the way Codex CLI, Claude Code (CLAUDE.md) and other harnesses load project memory files.
The trajectory and its statistics are written to OUT_JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_modern import ALL_FEATURES, INSTANCE, SYSTEM, ModernEnvironmentMixin, ModernModel, PacedTextbasedModel  # noqa: E402

GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
API_PROVIDERS = ("gemini", "anthropic", "openai", "openrouter", "groq", "mistral", "deepseek", "xai")

# A permission policy, as a deny-list of command patterns. Real harnesses use allow/ask/deny rules per tool
# (opencode), exec policies (Codex CLI) or user confirmation (mini-swe-agent's interactive mode).
DENY = {
    "network access": r"\b(curl|wget|ssh|scp|nc|Invoke-WebRequest)\b",
    "pushing code": r"\bgit\s+push\b",
    "deleting outside the task": r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\.\.)",
    "leaving the task folder": r"(^|[;&|(]\s*)cd\s+(/|~|\.\.|[A-Za-z]:)",
    "searching the whole disk":r"\b(find|grep\s+-[a-zA-Z]*r[a-zA-Z]*|ls\s+-[a-zA-Z]*R[a-zA-Z]*)\s+(/|~|[A-Za-z]:[\\/])(\s|$)",
    "reading secrets": r"(\.env\b|id_rsa|\.ssh/|printenv|^\s*env\s*$|/proc/self/environ)",
    "privilege escalation": r"\b(sudo|su)\b",
}


def kill_tree(p: subprocess.Popen) -> None:
    """Kill a process and all its children (taskkill /T on Windows, the process group elsewhere)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        p.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass


class SandboxedBashEnvironment:
    """A mini-swe-agent Environment: the same three methods as its LocalEnvironment (execute, get_template_vars,
    serialize), but every command runs in bash inside the task folder, with a scrubbed environment (no API keys),
    a permission check and a timeout. Like notebook 14's sandbox, it is a teaching layer, not a security boundary:
    production harnesses add OS-level isolation (containers, Seatbelt, Landlock, restricted Windows tokens)."""

    def __init__(self, cwd: str, timeout: int = 60):
        self.cwd, self.timeout = cwd, timeout
        self.bash = os.environ.get("HARNESS_BASH") or (GIT_BASH if os.name == "nt" else shutil.which("bash"))
        keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG")
        self.env = {k: os.environ[k] for k in keep if k in os.environ} | {"PYTHONIOENCODING": "utf-8", "PAGER": "cat", "GIT_PAGER": "cat",
                                                                       "TERM": "dumb", "EDITOR": "false", "VISUAL": "false"}
        self.denied: list[str] = []

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        for reason, pattern in DENY.items():
            if re.search(pattern, command, flags=re.MULTILINE):
                self.denied.append(reason)
                return {"output": f"Permission denied by the harness policy ({reason}).", "returncode": 126, "exception_info": ""}
        # No stdin (an interactive `vim` or `less` fails at once instead of waiting forever), and on a timeout the whole
        # process tree is killed: killing only bash leaves a `find /` or `vim` child alive holding the output pipe open.
        # python3 does not exist on Windows (only a Store shortcut), and models type it by habit
        prelude = 'python3() { python "$@"; }; ' if os.name == "nt" else ""
        p = subprocess.Popen([self.bash, "-c", prelude + command], cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}))
        try:
            stdout, _ = p.communicate(timeout=timeout or self.timeout)
            out = {"output": stdout[-8000:], "returncode": p.returncode, "exception_info": ""}
        except subprocess.TimeoutExpired:
            kill_tree(p)
            out = {"output": "", "returncode": -1, "exception_info": f"The command timed out after {timeout or self.timeout}s and was killed."}
        lines = out["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and out["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission, "extra": {"exit_status": "Submitted", "submission": submission}})
        return out

    def get_template_vars(self, **kwargs) -> dict:
        return {"system": "Linux (bash)", "release": "", "version": "", "machine": "x86_64", "cwd": self.cwd} | kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment_type": type(self).__name__, "cwd": self.cwd}}}


class ModernEnvironment(ModernEnvironmentMixin, SandboxedBashEnvironment):
    """The sandbox above (permission policy, scrubbed environment) plus the modern profile's tools and gates."""

    def __init__(self, cwd: str, features, timeout: int = 60):
        super().__init__(cwd, timeout)
        self.setup_modern(features)


def workspace_context(task_dir: str, max_files: int = 200) -> str:
    """What every current harness tells the model before the first step (Claude Code's environment block, Codex's
    workspace context): where it is, which files exist and how to run things. Without it, a capable model starts by
    searching the disk for the files (`find / -name ...`), which wastes steps and leaves the task folder."""
    files = sorted(p for p in Path(task_dir).rglob("*") if p.is_file() and not any(part.startswith((".", "__pycache__")) for part in p.relative_to(task_dir).parts)
                   and not p.name.startswith("_trajectory"))
    listing = []
    for p in files[:max_files]:
        try:
            n = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        except OSError:
            n = 0
        listing.append(f"  {p.relative_to(task_dir).as_posix()} ({n} lines)")
    more = f"\n  ... and {len(files) - max_files} more" if len(files) > max_files else ""
    return ("Workspace: your shell already starts in the root folder of this repository, and every file of the task is in it. "
            "Work only inside this folder (use relative paths; there is no need to search the rest of the disk). "
            "Run Python as `python` and the tests as `python -m pytest -q`. Files:\n" + "\n".join(listing) + more)


def run(task_dir: str, model: str, out_json: str, steps: int = 20, agents_md: bool = False,
        temperature: float = 0.0, seed: int = 0, profile: str = "baseline", features=ALL_FEATURES,
        max_tokens: int = 4096, wall: int = 300) -> dict:
    features = set(features) if profile == "modern" else set()
    tools = "tools" in features
    cfg = yaml.safe_load((resources.files("minisweagent.config") / ("mini.yaml" if tools else "mini_textbased.yaml")).read_text(encoding="utf-8"))
    agent_cfg = cfg["agent"] | {"step_limit": steps, "cost_limit": 0, "wall_time_limit_seconds": wall}
    if tools:
        agent_cfg |= {"system_template": SYSTEM, "instance_template": INSTANCE}
    api = model.split("/", 1)[0] in API_PROVIDERS
    kwargs = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens, "timeout": 120}   # cap each reply; reasoning models spend part of it thinking, so keep it at a few thousand
    if not api:
        kwargs |= {"api_base": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), "num_ctx": 16384}
    model_cfg = cfg.get("model", {}) | {"model_name": model if api else f"ollama_chat/{model}",
                                        "cost_tracking": "ignore_errors", "model_kwargs": kwargs}   # local models have no price
    task = Path(task_dir, "TASK.md").read_text(encoding="utf-8")
    if agents_md and Path(task_dir, "AGENTS.md").exists():
        task += "\n\nProject instructions (from AGENTS.md):\n" + Path(task_dir, "AGENTS.md").read_text(encoding="utf-8")
    task += "\n\n" + workspace_context(task_dir)
    # Current Claude models check that the history before each of their thinking blocks was not edited ("preserved
    # thinking"): masking old tool outputs edits it, and the API answers 400 ("Invalid signature in thinking block").
    # For them the history must stay append-only; the provider's server-side context editing or compaction does the
    # masking's job instead. So client-side masking is skipped for Anthropic models, and the stats say so.
    mask_skipped = "mask" in features and model.startswith("anthropic/")
    if mask_skipped:
        features.discard("mask")
    env = ModernEnvironment(task_dir, features) if profile == "modern" else SandboxedBashEnvironment(task_dir)
    lm = ModernModel(mask_keep=6 if "mask" in features else None, **model_cfg) if tools else PacedTextbasedModel(**model_cfg)
    agent = DefaultAgent(lm, env, **agent_cfg)
    t0 = time.time()
    try:
        result = agent.run(task)
    except Exception as e:                                     # a crashed run is a failed run, not a crashed evaluation
        result = {"exit_status": f"error: {type(e).__name__}: {str(e)[:200]}"}
    msgs = agent.messages
    usage = [m.get("extra", {}).get("response", {}).get("usage", {}) or {} for m in msgs if m.get("role") == "assistant"]
    text = lambda m: str(m.get("content") or "")
    stats = {
        "exit_status": result.get("exit_status"), "steps": agent.n_calls, "seconds": round(time.time() - t0, 1),
        "format_errors": sum(1 for m in msgs if m.get("role") == "user" and ("EXACTLY ONE action" in text(m) or "No tool call found" in text(m))),
        "denied": env.denied, "profile": profile, "features": sorted(features), "mask_skipped": mask_skipped,
        "gate_rejections": getattr(env, "rejections", 0),
        "loop_warnings": sum(1 for m in msgs if "[harness] You have made this exact call" in text(m)),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usage),
        "messages": [{"role": m.get("role"), "content": text(m)[:4000],
                      "actions": [a.get("command", "") for a in m.get("extra", {}).get("actions", [])]} for m in msgs],
    }
    Path(out_json).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run mini-swe-agent on a task folder with a local Ollama model or an API model.")
    p.add_argument("task_dir"); p.add_argument("model"); p.add_argument("out_json")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--agents-md", action="store_true", help="add the task folder's AGENTS.md to the task")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", choices=["baseline", "modern"], default="baseline")
    p.add_argument("--features", default=",".join(ALL_FEATURES), help="modern profile: a comma-separated subset of " + ",".join(ALL_FEATURES))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--wall", type=int, default=300, help="wall-time limit per run, in seconds")
    a = p.parse_args()
    feats = [f for f in a.features.split(",") if f]
    assert set(feats) <= set(ALL_FEATURES), f"unknown features: {set(feats) - set(ALL_FEATURES)}"
    s = run(a.task_dir, a.model, a.out_json, a.steps, a.agents_md, a.temperature, a.seed, a.profile, feats, a.max_tokens, a.wall)
    print(json.dumps({k: v for k, v in s.items() if k != "messages"}))
