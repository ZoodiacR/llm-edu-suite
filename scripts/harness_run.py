#!/usr/bin/env python
"""
Run the mini-swe-agent harness on one task folder, with a local model through Ollama and a sandboxed bash
environment. Used by notebook 27; also runnable by hand.

mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent) is installed in its own virtual environment,
because its terminal UI needs a newer `rich` than the course environment allows:

    python -m venv .venv-harness
    .venv-harness/Scripts/python -m pip install mini-swe-agent==2.4.6 pytest     # (bin/ instead of Scripts/ on Linux/macOS)

Usage (with the harness venv's python):

    .venv-harness/Scripts/python scripts/harness_run.py TASK_DIR MODEL OUT_JSON [--steps 20] [--agents-md]

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
import time
from importlib import resources
from pathlib import Path

import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"

# A permission policy, as a deny-list of command patterns. Real harnesses use allow/ask/deny rules per tool
# (opencode), exec policies (Codex CLI) or user confirmation (mini-swe-agent's interactive mode).
DENY = {
    "network access": r"\b(curl|wget|ssh|scp|nc|Invoke-WebRequest)\b",
    "pushing code": r"\bgit\s+push\b",
    "deleting outside the task": r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\.\.)",
    "reading secrets": r"(\.env\b|id_rsa|\.ssh/|printenv|^\s*env\s*$|/proc/self/environ)",
    "privilege escalation": r"\b(sudo|su)\b",
}


class SandboxedBashEnvironment:
    """A mini-swe-agent Environment: the same three methods as its LocalEnvironment (execute, get_template_vars,
    serialize), but every command runs in bash inside the task folder, with a scrubbed environment (no API keys),
    a permission check and a timeout. Like notebook 14's sandbox, it is a teaching layer, not a security boundary:
    production harnesses add OS-level isolation (containers, Seatbelt, Landlock, restricted Windows tokens)."""

    def __init__(self, cwd: str, timeout: int = 60):
        self.cwd, self.timeout = cwd, timeout
        self.bash = os.environ.get("HARNESS_BASH") or (GIT_BASH if os.name == "nt" else shutil.which("bash"))
        keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG")
        self.env = {k: os.environ[k] for k in keep if k in os.environ} | {"PYTHONIOENCODING": "utf-8", "PAGER": "cat", "GIT_PAGER": "cat"}
        self.denied: list[str] = []

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        for reason, pattern in DENY.items():
            if re.search(pattern, command, flags=re.MULTILINE):
                self.denied.append(reason)
                return {"output": f"Permission denied by the harness policy ({reason}).", "returncode": 126, "exception_info": ""}
        try:
            p = subprocess.run([self.bash, "-c", command], cwd=self.cwd, env=self.env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout or self.timeout)
            out = {"output": (p.stdout + p.stderr)[-8000:], "returncode": p.returncode, "exception_info": ""}
        except subprocess.TimeoutExpired:
            out = {"output": "", "returncode": -1, "exception_info": f"The command timed out after {self.timeout}s."}
        lines = out["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and out["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission, "extra": {"exit_status": "Submitted", "submission": submission}})
        return out

    def get_template_vars(self, **kwargs) -> dict:
        return {"system": "Linux (bash)", "release": "", "version": "", "machine": "x86_64", "cwd": self.cwd} | kwargs

    def serialize(self) -> dict:
        return {"info": {"config": {"environment_type": "SandboxedBashEnvironment", "cwd": self.cwd}}}


def run(task_dir: str, model: str, out_json: str, steps: int = 20, agents_md: bool = False,
        temperature: float = 0.0, seed: int = 0) -> dict:
    cfg = yaml.safe_load((resources.files("minisweagent.config") / "mini_textbased.yaml").read_text(encoding="utf-8"))
    agent_cfg = cfg["agent"] | {"step_limit": steps, "cost_limit": 0, "wall_time_limit_seconds": 300}
    model_cfg = cfg.get("model", {}) | {
        "model_name": f"ollama_chat/{model}", "cost_tracking": "ignore_errors",     # local models have no price
        "model_kwargs": {"api_base": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), "temperature": temperature, "seed": seed, "num_ctx": 16384,
                         "max_tokens": 1024, "timeout": 120},          # cap each reply: small models can ramble forever
    }
    task = Path(task_dir, "TASK.md").read_text(encoding="utf-8")
    if agents_md and Path(task_dir, "AGENTS.md").exists():
        task += "\n\nProject instructions (from AGENTS.md):\n" + Path(task_dir, "AGENTS.md").read_text(encoding="utf-8")
    env = SandboxedBashEnvironment(task_dir)
    agent = DefaultAgent(LitellmTextbasedModel(**model_cfg), env, **agent_cfg)
    t0 = time.time()
    try:
        result = agent.run(task)
    except Exception as e:                                     # a crashed run is a failed run, not a crashed evaluation
        result = {"exit_status": f"error: {type(e).__name__}: {str(e)[:200]}"}
    msgs = agent.messages
    usage = [m.get("extra", {}).get("response", {}).get("usage", {}) or {} for m in msgs if m.get("role") == "assistant"]
    stats = {
        "exit_status": result.get("exit_status"), "steps": agent.n_calls, "seconds": round(time.time() - t0, 1),
        "format_errors": sum(1 for m in msgs if m.get("role") == "user" and "EXACTLY ONE action" in str(m.get("content", ""))),
        "denied": env.denied,
        "prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usage),
        "messages": [{"role": m.get("role"), "content": str(m.get("content", ""))[:4000]} for m in msgs],
    }
    Path(out_json).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run mini-swe-agent on a task folder with a local Ollama model.")
    p.add_argument("task_dir"); p.add_argument("model"); p.add_argument("out_json")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--agents-md", action="store_true", help="add the task folder's AGENTS.md to the task")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    s = run(a.task_dir, a.model, a.out_json, a.steps, a.agents_md, a.temperature, a.seed)
    print(json.dumps({k: v for k, v in s.items() if k != "messages"}))
