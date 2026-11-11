"""
Execute notebooks headlessly and report pass/fail and duration for each one. They run in place, so they keep their outputs.

    python tools/run_notebooks.py                       # every notebook under notebooks/ (all topic folders)
    python tools/run_notebooks.py 08_pretraining.ipynb  # only some of them (by file name, wherever they live)
    python tools/run_notebooks.py 05 10 14              # ... or by number

The exit code is 1 if any notebook fails, so the script can be used in CI.
"""
import glob
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

if sys.platform == "win32":                       # avoids the zmq warning about the Proactor event loop on Windows
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

NB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))


ALL = sorted(glob.glob(os.path.join(NB_DIR, "**", "*.ipynb"), recursive=True), key=os.path.basename)
ALL = [p for p in ALL if ".ipynb_checkpoints" not in p]


def resolve(arg: str) -> str:
    matches = [p for p in ALL if os.path.basename(p) == arg or os.path.basename(p).startswith(arg)]
    if len(matches) != 1:
        raise SystemExit(f"'{arg}' matches {len(matches)} notebooks: {[os.path.basename(m) for m in matches]}")
    return matches[0]


def run(path: str) -> bool:
    name = os.path.basename(path)
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    ok, msg = True, "OK"
    try:
        # Use the notebook's own folder as the working directory, just as Jupyter does.
        NotebookClient(nb, timeout=int(os.environ.get("NB_CELL_TIMEOUT", 3600)), kernel_name="python3", resources={"metadata": {"path": os.path.dirname(path)}}).execute()
    except Exception as e:  # keep going so we report every failing notebook, not just the first one
        ok, msg = False, "FAIL\n" + str(e)[-1500:]
    nbformat.write(nb, path)
    print(f"[{time.time() - t0:7.1f}s] {name}: {msg}", flush=True)
    return ok


if __name__ == "__main__":
    paths = [resolve(a) for a in sys.argv[1:]] or ALL
    results = [run(p) for p in paths]
    print(f"\n{sum(results)}/{len(results)} notebooks OK")
    sys.exit(0 if all(results) else 1)
