"""
Build the notebooks from their generators in tools/notebooks/.

    python tools/build_notebooks.py            # every notebook
    python tools/build_notebooks.py 05 14      # only some, by number

Building rewrites the .ipynb files without outputs; run tools/run_notebooks.py afterwards to execute them.
"""
import glob
import os
import runpy
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
sys.path.insert(0, HERE)

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(HERE, "nb[0-9][0-9]_*.py")))
    wanted = sys.argv[1:]
    if wanted:
        files = [f for f in files if any(os.path.basename(f)[2:4] == w.zfill(2) for w in wanted)]
    for f in files:
        runpy.run_path(f, run_name="__main__")
