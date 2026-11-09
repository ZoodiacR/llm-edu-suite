"""Tiny dependency-free .ipynb builder used by tools/notebooks/nbXX_*.py."""
import json
import os


class NB:
    def __init__(self):
        self.cells = []

    def _id(self) -> str:
        """A deterministic cell id. nbformat >= 4.5 requires one, and keeping it stable keeps git diffs small."""
        return f"cell-{len(self.cells):03d}"

    def md(self, text: str):
        self.cells.append({
            "id": self._id(),
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        })

    def code(self, src: str):
        self.cells.append({
            "id": self._id(),
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True),
        })

    def save(self, path: str):
        if KEEP_OUTPUTS and os.path.exists(path):
            _merge_outputs(self.cells, path)
        meta = None
        if KEEP_OUTPUTS and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                meta = json.load(f).get("metadata")          # keep the kernel the notebook was executed with
        nb = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        if meta:
            nb["metadata"] = meta
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            if meta:                                          # match nbformat's layout so unchanged notebooks give no diff
                f.write(json.dumps(nb, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            else:
                json.dump(nb, f, ensure_ascii=False, indent=1)


# Rebuilding without losing results.
# With KEEP_OUTPUTS on (build_notebooks.py --keep-outputs), a rebuilt notebook takes the outputs of the
# previous .ipynb for every code cell whose source did not change. Prose edits then need no re-execution;
# cells whose code changed come out empty and are reported, so you know what to run again.

KEEP_OUTPUTS = False


def _merge_outputs(cells: list, old_path: str) -> None:
    import os as _os2
    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)
    pool = {}
    for c in old["cells"]:
        if c["cell_type"] == "code":
            pool.setdefault("".join(c["source"]), []).append(c)
    missing = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        match = pool.get("".join(c["source"]))
        if match:
            prev = match.pop(0)
            c["outputs"], c["execution_count"] = prev.get("outputs", []), prev.get("execution_count")
            c["metadata"] = prev.get("metadata", {})
        elif "".join(c["source"]).strip():
            missing += 1
    name = _os2.path.basename(old_path)
    print(f"  {name}: outputs kept" + (f", {missing} code cell(s) changed and need a re-run" if missing else ", no code changed"))


# Topic layout of notebooks/
# A notebook's number is its place in the overall study order, and its folder is its topic.

import os as _os

NOTEBOOKS_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "notebooks"))
TOPICS = {
    "00_start_here": ["00"],
    "01_architecture": ["01", "02", "03", "04", "17"],
    "02_pretraining_and_inference": ["05", "06", "09", "18", "19"],
    "03_post_training": ["07", "08"],
    "04_applications": ["10", "12", "13"],
    "05_agents_and_mcp": ["11", "14", "16"],
    "06_llmops": ["15"],
}


def topic_of(name: str) -> str:
    num = _os.path.basename(name)[:2]
    for folder, nums in TOPICS.items():
        if num in nums:
            return folder
    raise KeyError(f"notebook {name} is not assigned to a topic in tools/nb_utils.py:TOPICS")


def notebook_path(name: str) -> str:
    """Return notebooks/<topic>/<name>, creating the topic folder if it does not exist yet."""
    folder = _os.path.join(NOTEBOOKS_DIR, topic_of(name))
    _os.makedirs(folder, exist_ok=True)
    return _os.path.join(folder, name)
