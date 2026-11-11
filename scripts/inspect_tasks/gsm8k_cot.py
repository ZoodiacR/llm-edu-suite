"""
GSM8K with chain of thought as an Inspect AI task: the same 80 problems, prompt and answer extraction as
notebook 15, so the hand-written evaluation and the framework can be compared. Used by notebook 29.

Inspect (https://inspect.aisi.org.uk, UK AI Security Institute) runs in its own environment, because it needs a
newer `rich` than the course environment allows:

    python -m venv .venv-eco
    .venv-eco/Scripts/python -m pip install inspect-ai==0.3.289 arize-phoenix==21.0.0 datasets   # bin/ on Linux/macOS

Run (the model through Ollama's OpenAI-compatible endpoint):

    .venv-eco/Scripts/inspect eval scripts/inspect_tasks/gsm8k_cot.py --model ollama/qwen2.5:3b \
        --model-base-url http://localhost:11434/v1 --temperature 0 --max-tokens 512 --max-connections 1
    .venv-eco/Scripts/inspect view          # browse the logs
"""
import re

import datasets
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, match, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

COT = "Solve the problem step by step, then give the final answer as a number on the last line."


def load(n: int) -> MemoryDataset:
    gsm = datasets.load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(n))   # notebook 15's sample
    return MemoryDataset([Sample(input=q, target=a.split("####")[-1].replace(",", "").strip(), id=i)
                          for i, (q, a) in enumerate(zip(gsm["question"], gsm["answer"]))])


def final_number(text: str):
    text = text.split("</think>")[-1].replace("$", "").replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


@scorer(metrics=[accuracy(), stderr()])
def last_number():
    """Notebook 15's rule: the last number in the answer must equal the reference."""
    async def score(state: TaskState, target: Target) -> Score:
        pred = final_number(state.output.completion)
        ok = pred is not None and abs(pred - float(target.text)) < 1e-6
        return Score(value=CORRECT if ok else INCORRECT, answer=None if pred is None else f"{pred:g}")
    return score


@task
def gsm8k_cot(n: int = 80):
    # Two scorers on the same generations: our rule, and Inspect's built-in numeric match at the end of the answer
    return Task(dataset=load(n), solver=[system_message(COT), generate()],
                scorer=[last_number(), match(location="end", numeric=True)])
