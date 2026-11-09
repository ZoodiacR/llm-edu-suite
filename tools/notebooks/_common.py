"""
Shared pieces for the notebook generators in this folder.

Each `nbXX_*.py` file builds exactly one notebook. They all start from the same few setup cells, kept here
so a fix to the setup reaches every notebook. Build them with `python tools/build_notebooks.py`.
"""
import os
import sys

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
sys.path.insert(0, TOOLS_DIR)
from nb_utils import NB, notebook_path  # noqa: E402

# Setup cell for the PyTorch notebooks (00-06): repo-root finder, numpy/torch/matplotlib, seed and device.
SETUP = r'''
import os, sys, math, time, tempfile
ROOT = os.path.abspath(".")
while not os.path.isdir(os.path.join(ROOT, "llm_edu_suite")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)                    # notebooks live in notebooks/<topic>/, so climb up to the repo root
sys.path.insert(0, ROOT)                            # so that `import llm_edu_suite` works from the notebook
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from llm_edu_suite.utils import seed_everything, get_device, count_params, human, show_mask
seed_everything(0)
device = get_device()
print("torch", torch.__version__, "| device", device)
'''

# Setup cell for the fine-tuning notebooks (07-09, 12).
SETUP_FT = r'''
import os, sys, math, time, random, copy
ROOT = os.path.abspath(".")
while not os.path.isdir(os.path.join(ROOT, "llm_edu_suite")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)                    # climb from notebooks/<topic>/ up to the repo root
sys.path.insert(0, ROOT)                            # so that `import llm_edu_suite` works from any notebook
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from llm_edu_suite.utils import seed_everything, get_device, count_params, human
seed_everything(0)
device = get_device()
print("torch", torch.__version__, "| device", device)
'''

# Setup cell for notebooks that do not need PyTorch.
SETUP_NOTORCH = r'''
import os, sys, math, time, random, json
ROOT = os.path.abspath(".")
while not os.path.isdir(os.path.join(ROOT, "llm_edu_suite")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)                    # climb from notebooks/<topic>/ up to the repo root
sys.path.insert(0, ROOT)                            # so that `import llm_edu_suite` works from any notebook
import numpy as np
random.seed(0); np.random.seed(0)
'''

# Shared base model + chat format + toy task for notebooks 07 and 08.
BASE_SFT_SETUP = r'''
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.data import ChatFormat, make_reverse_task, synthetic_structured_corpus, train_val_split
from llm_edu_suite.training import TrainConfig, train_lm

V = 512
fmt = ChatFormat.for_vocab(V)                      # the last 5 ids are reserved for the role tokens SYS, USER, ASSIST, END, PAD
print("role tokens:", fmt)

# Our stand-in for a pretrained base model. A short run on the synthetic corpus teaches it the tokens, but no task.
data = synthetic_structured_corpus(n_tokens=100_000, vocab=V - 5)
train_data, val_data = train_val_split(data)
seed_everything(0)
base = TransformerLM(ModelConfig(vocab_size=V, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, max_seq_len=128))
train_lm(base, train_data, val_data, TrainConfig(steps=200, batch_size=32, block_size=64, max_lr=3e-3, warmup=20, eval_every=200, log=False), device)
print(f"base model: {human(base.num_params())} params")

# The instruction task is to reverse the user's token sequence, which needs attention to copy and reorder.
# We use 8000 examples: with only ~1500 the model memorises them (train loss around 0.1) but generalises poorly.
sft_train = make_reverse_task(fmt, n=8000, seed=0)
sft_test = make_reverse_task(fmt, n=200, seed=99)
inp, lab = sft_train[0]
print("example input :", inp.tolist())
print("example labels:", lab.tolist(), "  <- -100 = no loss (prompt), loss only on the reply")
'''

# Shared evaluation helpers for notebooks 07 and 08.
EVAL_HELPERS = r'''
from llm_edu_suite.finetune import greedy_answer
from llm_edu_suite.evaluation import exact_match_rate, perplexity_on_examples

def reverse_cases(n=150, seed=7):
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        u = [rng.randint(1, 400) for _ in range(rng.randint(2, 5))]
        cases.append((u, u[::-1]))
    return cases

CASES = reverse_cases()          # 150 held-out prompts, shared by every method so the numbers are comparable
RESULTS = {}                      # maps name to (exact match, reply perplexity); evaluate() fills it in

def evaluate(model, name):
    """exact match = the greedy answer is exactly the reversed sequence; reply perplexity on held-out examples."""
    em = exact_match_rate(lambda u: greedy_answer(model, fmt, [10], u, device=device), CASES)
    ppl = perplexity_on_examples(model, sft_test, fmt, device=device)
    RESULTS[name] = (em, ppl)
    print(f"{name:24s} exact match {100*em:5.1f}%   reply perplexity {ppl:8.2f}")
    return em, ppl

def plot_results(title):
    names = list(RESULTS); ems = [100 * RESULTS[n][0] for n in names]; ppls = [RESULTS[n][1] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.45 * len(names) + 1.5))
    ax[0].barh(names, ems, color="tab:blue"); ax[0].set_xlabel("exact match % (higher is better)"); ax[0].set_xlim(0, 100)
    for i, v in enumerate(ems): ax[0].text(v + 1, i, f"{v:.1f}%", va="center")
    ax[1].barh(names, ppls, color="tab:orange"); ax[1].set_xscale("log"); ax[1].set_xlabel("reply perplexity (log, lower is better)")
    for a in ax: a.invert_yaxis(); a.grid(alpha=.3, axis="x")
    ax[1].set_yticklabels([]); fig.suptitle(title); plt.tight_layout(); plt.show()
'''

# Lighter setup used by notebooks 14-16.
SETUP_LIGHT = r'''
import os, sys, json, time, random
ROOT = os.path.abspath(".")
while not os.path.isdir(os.path.join(ROOT, "llm_edu_suite")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)                     # so `llm_edu_suite` imports from any notebooks/ sub-folder
import numpy as np
import matplotlib.pyplot as plt
random.seed(0); np.random.seed(0)
'''
