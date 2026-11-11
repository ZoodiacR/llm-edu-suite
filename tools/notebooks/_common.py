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

# Setup cell for the PyTorch notebooks (02-10): repo-root finder, numpy/torch/matplotlib, seed and device.
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

# Setup cell for the recommender notebook (23).
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



# Lighter setup used by notebooks 00, 15 and 27.
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

# Setup cell for the application notebooks (12, 16-22, 24-26, 28-30) built on Hugging Face, LangChain, LangGraph, LangSmith, RAGAS and MCP.
# NOTEBOOK_PROJECT is replaced per notebook with the LangSmith project name (see setup_apps()).
SETUP_APPS = r'''
import os, sys, json, time, random, warnings
ROOT = os.path.abspath(".")
while not os.path.isdir(os.path.join(ROOT, "llm_edu_suite")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)                    # walk up from notebooks/<topic>/ to the repo root
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)
random.seed(0); np.random.seed(0)

from llm_edu_suite.config import setup, get_llm, get_embeddings
info = setup(project="NOTEBOOK_PROJECT")           # loads .env, picks the LangSmith project, checks that Ollama is up
'''


def setup_apps(project: str) -> str:
    """The SETUP_APPS cell with this notebook's LangSmith project name filled in."""
    return SETUP_APPS.replace("NOTEBOOK_PROJECT", project)
